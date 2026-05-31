from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup
from openai import APIStatusError

from bot.conversation import ConversationStore
from bot.format import md_to_telegram_html
from bot.logging_config import hash_user_id
from bot.pii import PIIDetector
from rag.agent.context import AgentContext
from rag.agent.query_normalization import expand_acronyms_in_query
from rag.agent.runner import run_agent_planned

logger = logging.getLogger(__name__)

router = Router()

TELEGRAM_MESSAGE_LIMIT = 4000

WELCOME_TEXT = (
    "Привет! Я информационный бот по поступлению в бакалавриат "
    "Высшей школы экономики.\n"
    "Помогу узнать про программы, экзамены, сроки и многое другое.\n\n"
    "<i>Московский кампус, 2026</i>"
)

HELP_TEXT = (
    "Я информационный бот по вопросам поступления в бакалавриат "
    "Высшей школы экономики.\n\n"
    "Информация предоставляется из открытых источников — официальный сайт "
    "Высшей школы экономики."
)

GENERIC_ERROR_TEXT = (
    "Что-то пошло не так при обработке запроса. "
    "Попробуйте переформулировать вопрос или повторить чуть позже — "
    "обычно это помогает."
)

BUDGET_ERROR_TEXT = (
    "Сервис временно недоступен — закончился лимит у провайдера API. "
    "Попробуйте через несколько часов."
)

FALLBACK_REPLY = "Не удалось сформировать ответ."

ERROR_REPLY_PREFIXES = (
    "Не удалось сформировать ответ",
    "Что-то пошло не так",
    "Сервис временно недоступен",
    "Запрос потребовал слишком много",
)


def _is_error_reply(text: str) -> bool:
    return any(text.startswith(p) for p in ERROR_REPLY_PREFIXES)


async def _keep_typing(bot: Bot, chat_id: int) -> None:
    while True:
        await bot.send_chat_action(chat_id, action="typing")
        await asyncio.sleep(4)


BTN_START = "/start"
BTN_PROGRAMS = "Список программ"
BTN_HELP = "Помощь"
BTN_RESET = "Очистить историю"

_PROGRAMS_DIR = Path("data/programs/2026")
_PROGRAMS_LIST_CACHE: str | None = None


def _programs_list_text() -> str:
    global _PROGRAMS_LIST_CACHE
    if _PROGRAMS_LIST_CACHE is not None:
        return _PROGRAMS_LIST_CACHE
    names: list[str] = []
    for f in sorted(_PROGRAMS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            name = data.get("name")
            if name:
                names.append(str(name))
        except Exception:
            continue
    names.sort()
    body = "\n".join(f"• {n}" for n in names)
    _PROGRAMS_LIST_CACHE = (
        f"Все {len(names)} программ бакалавриата НИУ ВШЭ (Москва, набор 2026):\n\n{body}"
    )
    return _PROGRAMS_LIST_CACHE


def _main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_START), KeyboardButton(text=BTN_PROGRAMS)],
            [KeyboardButton(text=BTN_HELP), KeyboardButton(text=BTN_RESET)],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Задайте свой вопрос",
    )


@dataclass(frozen=True)
class HandlersDeps:
    agent_context: AgentContext
    conversation_store: ConversationStore
    max_tool_calls: int
    pii_detector: PIIDetector


_deps: HandlersDeps | None = None


def set_deps(deps: HandlersDeps) -> None:
    global _deps
    _deps = deps


def _require_deps() -> HandlersDeps:
    if _deps is None:
        raise RuntimeError("HandlersDeps not initialized; call set_deps() before polling")
    return _deps


def _user_id(message: Message) -> int:
    return int(message.from_user.id) if message.from_user else 0


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    deps = _require_deps()
    deps.conversation_store.clear(_user_id(message))
    await message.answer(WELCOME_TEXT, reply_markup=_main_keyboard(), parse_mode="HTML")


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    deps = _require_deps()
    deps.conversation_store.clear(_user_id(message))
    await message.answer("История диалога очищена. Задавай новый вопрос.")


@router.message(F.text)
async def on_text(message: Message) -> None:
    deps = _require_deps()
    started_at = time.time()

    raw_user_id = _user_id(message)
    user_id_hash = hash_user_id(raw_user_id)

    raw_text = message.text or ""
    if not raw_text.strip():
        return

    if raw_text == BTN_PROGRAMS:
        await message.answer(_programs_list_text(), reply_markup=_main_keyboard())
        return
    if raw_text == BTN_HELP:
        await message.answer(HELP_TEXT, reply_markup=_main_keyboard())
        return
    if raw_text == BTN_RESET:
        deps.conversation_store.clear(raw_user_id)
        await message.answer(
            "История диалога очищена. Задайте новый вопрос.",
            reply_markup=_main_keyboard(),
        )
        return

    masked_text = raw_text
    try:
        det = await deps.pii_detector.detect_async(raw_text)
        if det.has_pii():
            masked_text = det.masked_text

        normalized_query = expand_acronyms_in_query(masked_text, deps.agent_context.acronyms)
        history_context = deps.conversation_store.render_context(raw_user_id)
        full_query = (
            f"{history_context}\n\nНовый вопрос: {normalized_query}"
            if history_context
            else normalized_query
        )

        typing_task: asyncio.Task[None] | None = None
        if message.bot is not None:
            typing_task = asyncio.create_task(_keep_typing(message.bot, message.chat.id))

        try:
            response = await run_agent_planned(
                query=full_query,
                context=deps.agent_context,
                max_tool_calls=deps.max_tool_calls,
            )
        finally:
            if typing_task is not None:
                typing_task.cancel()

        reply_text = response.text or FALLBACK_REPLY
        plan_intent = None
        if response.plan and response.plan.steps:
            plan_intent = response.plan.steps[0].intent.value

        if not _is_error_reply(reply_text):
            deps.conversation_store.add(
                user_id=raw_user_id,
                user_query=masked_text,
                assistant_reply=reply_text,
            )

        html_reply = md_to_telegram_html(reply_text)
        for chunk in _split_for_telegram(html_reply):
            await message.answer(
                chunk,
                disable_web_page_preview=True,
                parse_mode="HTML",
            )

        duration_ms = round((time.time() - started_at) * 1000.0, 1)
        logger.info(
            f"processed user={user_id_hash} intent={plan_intent} "
            f"tools={len(response.tool_calls)} ms={duration_ms}"
        )
    except APIStatusError as exc:
        status_code = getattr(exc, "status_code", None)
        await message.answer(BUDGET_ERROR_TEXT if status_code == 402 else GENERIC_ERROR_TEXT)
        logger.error(f"failed user={user_id_hash} APIStatusError status={status_code}")
    except Exception as exc:
        await message.answer(GENERIC_ERROR_TEXT)
        logger.error(f"failed user={user_id_hash} {type(exc).__name__}: {exc}", exc_info=exc)


def _split_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > limit and current:
            parts.append("".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)
    if current:
        parts.append("".join(current))
    return parts
