import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from rag.agent import tools
from rag.agent.context import AgentContext
from rag.agent.llm_adapter import build_pydantic_ai_model
from rag.agent.planner import Intent, Plan, build_plan, format_plan_for_worker
from rag.agent.schemas import (
    AdmissionDeadline,
    AggregateOperation,
    AnalysisField,
    AnalysisResult,
    ComparisonTable,
    GroupByField,
    MatchResult,
    ProgramDetails,
    ProgramFilter,
    ProgramHit,
    ProgramSummary,
)
from rag.agent.tools import tool_description

_PROMPT_CACHE: dict[tuple[str, Path], str] = {}


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[str] = field(default_factory=list)
    plan: Plan | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0


def _load_system_prompt(path: Path) -> str:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(raw["system"])


def _load_executor_prompt(intent_value: str, prompts_dir: Path) -> str:
    cache_key = (intent_value, prompts_dir)
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]

    shared = _load_system_prompt(prompts_dir / "executor_shared.yaml")
    format_skill = _load_system_prompt(prompts_dir / "executor_format.yaml")
    intent_prompt = _load_system_prompt(prompts_dir / f"executor_{intent_value}.yaml")

    combined = shared.rstrip() + "\n\n" + format_skill.rstrip() + "\n\n" + intent_prompt.strip()
    _PROMPT_CACHE[cache_key] = combined
    return combined


def build_agent(
    intent: str,
    prompts_dir: Path = Path("rag/agent/prompts"),
) -> Agent[AgentContext, str]:
    system_prompt = _load_executor_prompt(intent, prompts_dir)
    model = build_pydantic_ai_model()
    agent: Agent[AgentContext, str] = Agent(
        model=model,
        system_prompt=system_prompt,
        deps_type=AgentContext,
        output_type=str,
    )

    @agent.tool(description=tool_description("search_programs"))
    async def search_programs(
        ctx: RunContext[AgentContext], /, query: str, top_k: int = 5
    ) -> list[ProgramHit]:
        return tools.search_programs(ctx.deps, query=query, top_k=top_k, corpus="programs")

    @agent.tool(description=tool_description("search_knowledge"))
    async def search_knowledge(
        ctx: RunContext[AgentContext], /, query: str, top_k: int = 5
    ) -> list[ProgramHit]:
        return tools.search_programs(ctx.deps, query=query, top_k=top_k, corpus="knowledge")

    @agent.tool(description=tool_description("filter_programs"))
    async def filter_programs(
        ctx: RunContext[AgentContext], /, filter_: ProgramFilter
    ) -> list[ProgramSummary]:
        return tools.filter_programs(ctx.deps, filter_)

    @agent.tool(description=tool_description("get_program_details"))
    async def get_program_details(
        ctx: RunContext[AgentContext], /, slug: str, sections: list[str]
    ) -> ProgramDetails | None:
        return tools.get_program_details(ctx.deps, slug=slug, sections=sections)

    @agent.tool(description=tool_description("compare_programs"))
    async def compare_programs(
        ctx: RunContext[AgentContext], /, slugs: list[str], aspects: list[str]
    ) -> ComparisonTable:
        return tools.compare_programs(ctx.deps, slugs=slugs, aspects=aspects)

    @agent.tool(description=tool_description("get_admission_deadlines"))
    async def get_admission_deadlines(
        ctx: RunContext[AgentContext],
        /,
        has_hse_exam: bool,
    ) -> list[AdmissionDeadline]:
        return tools.get_admission_deadlines(ctx.deps, has_hse_exam=has_hse_exam)

    @agent.tool(description=tool_description("analyze_programs"))
    async def analyze_programs(
        ctx: RunContext[AgentContext],
        /,
        operation: AggregateOperation,
        field: AnalysisField,
        filter_: ProgramFilter | None = None,
        group_by: GroupByField = "none",
    ) -> AnalysisResult:
        return tools.analyze_programs(
            ctx.deps,
            operation=operation,
            field=field,
            filter_=filter_,
            group_by=group_by,
        )

    @agent.tool(description=tool_description("match_by_scores"))
    async def match_by_scores(
        ctx: RunContext[AgentContext],
        /,
        user_scores: dict[str, int],
    ) -> MatchResult:
        return tools.match_by_scores(ctx.deps, user_scores=user_scores)

    return agent


OFF_TOPIC_REPLIES: dict[str, str] = {
    "off_scope": (
        "Этот вопрос выходит за рамки моей темы. Я помогаю с поступлением в "
        "бакалавриат НИУ ВШЭ — программы, экзамены, проходные баллы, стоимость, "
        "сроки, ДВИ, олимпиады и индивидуальные достижения."
    ),
    "subjective": (
        "Я не оцениваю программы по субъективной шкале и не составляю рейтинги "
        "«лучшести» или «престижности». Могу показать факты — экзамены, проходные "
        "баллы, стоимость, карьерные траектории, — а сравните их сами."
    ),
    "personal": (
        "Я не даю личных рекомендаций по выбору программы. Могу показать факты по "
        "нескольким программам — экзамены, проходные баллы, стоимость, карьеру, — "
        "а решение остаётся за вами и приёмной комиссией."
    ),
    "future": (
        "Я не могу предсказать будущие проходные баллы и конкурс — они зависят от "
        "состава абитуриентов конкретного года. Могу показать прошлогодние "
        "результаты приёма для ориентира."
    ),
    "off_task": (
        "Я помогаю с поступлением в бакалавриат НИУ ВШЭ — программы, экзамены, "
        "баллы, сроки. Не выполняю задач вне этой темы (перевод, суммаризация, "
        "рерайтинг и т.п.)."
    ),
    "hallucination": (
        "Я не нашёл программы с таким названием в каталоге бакалавриата ВШЭ "
        "Москва 2026. Уточните название, пожалуйста, или назовите тему — "
        "подскажу близкие программы из актуального списка."
    ),
}


def off_topic_reply(focus: str) -> str:
    return OFF_TOPIC_REPLIES.get(focus.strip().lower(), OFF_TOPIC_REPLIES["off_scope"])


async def run_agent_planned(
    query: str,
    context: AgentContext,
    worker_prompts_dir: Path = Path("rag/agent/prompts"),
    planner_prompt_path: Path = Path("rag/agent/prompts/planner.yaml"),
    max_tool_calls: int = 4,
) -> AgentResponse:
    plan = await build_plan(query, planner_prompt_path)

    first_step = plan.steps[0] if plan.steps else None
    first_intent = first_step.intent if first_step else None
    if first_intent == Intent.off_topic_refuse:
        focus = first_step.focus if first_step else ""
        return AgentResponse(text=off_topic_reply(focus), tool_calls=[], plan=plan)

    plan_hint = format_plan_for_worker(plan)
    enriched_query = f"{plan_hint}\n\n{query}"

    intent_value = first_intent.value if first_intent is not None else "program_search_broad"
    agent = build_agent(intent=intent_value, prompts_dir=worker_prompts_dir)

    limits = UsageLimits(tool_calls_limit=max_tool_calls)
    extra_body_env = os.environ.get("EXECUTOR_EXTRA_BODY")
    model_settings: Any = {"extra_body": json.loads(extra_body_env)} if extra_body_env else None
    try:
        result = await agent.run(
            enriched_query, deps=context, usage_limits=limits, model_settings=model_settings
        )
    except UsageLimitExceeded:
        return AgentResponse(
            text=(
                "Запрос потребовал слишком много обращений к инструментам. "
                "Попробуйте переформулировать."
            ),
            tool_calls=[],
            plan=plan,
        )

    tool_calls: list[str] = []
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            tool_name = getattr(part, "tool_name", None)
            if tool_name:
                tool_calls.append(tool_name)
    usage = result.usage()
    return AgentResponse(
        text=str(result.output),
        tool_calls=tool_calls,
        plan=plan,
        input_tokens=getattr(usage, "input_tokens", 0),
        output_tokens=getattr(usage, "output_tokens", 0),
        requests=getattr(usage, "requests", 0),
    )


def run_agent_planned_sync(
    query: str,
    context: AgentContext,
    worker_prompts_dir: Path = Path("rag/agent/prompts"),
    planner_prompt_path: Path = Path("rag/agent/prompts/planner.yaml"),
    max_tool_calls: int = 4,
) -> AgentResponse:
    return asyncio.run(
        run_agent_planned(
            query,
            context,
            worker_prompts_dir,
            planner_prompt_path,
            max_tool_calls,
        )
    )
