from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

from bot.config import BotConfig
from bot.conversation import ConversationStore
from bot.handlers import HandlersDeps, router, set_deps
from bot.logging_config import setup_logging
from bot.pii import build_default_detector
from rag.agent.context import AgentContext, load_deadlines
from rag.agent.tools import derive_acronyms_from_qdrant
from rag.embedding.embedder import Embedder, EmbedderConfig
from rag.experiments.rerankers import RerankingPipeline, build_llm_reranker
from rag.indexing.indexer import IndexerConfig
from rag.indexing.indexer import _client as _qdrant_client
from rag.retrieval.pipeline import RetrievalPipeline

load_dotenv()

logger = logging.getLogger(__name__)


def _build_agent_context(config: BotConfig) -> AgentContext:
    emb_config = EmbedderConfig.from_yaml(config.embedding_config_path)
    embedder = Embedder(emb_config)
    index_config = IndexerConfig.from_yaml(config.retrieval_config_path, dim=emb_config.dim)
    base_pipeline = RetrievalPipeline(embedder=embedder, index_config=index_config)

    reranker_pipeline = RerankingPipeline(
        base=base_pipeline,
        reranker=build_llm_reranker(
            model_id="deepseek/deepseek-v4-flash",
            extra_body={"reasoning": {"enabled": False}},
        ),
        pool_size=20,
    )

    index_config.qdrant_path.mkdir(parents=True, exist_ok=True)
    qdrant_client = _qdrant_client(index_config.qdrant_path)

    deadlines = load_deadlines(config.knowledge_dir)

    acronyms = derive_acronyms_from_qdrant(
        qdrant_client, index_config.collection_name, config.acronyms_config_path
    )

    return AgentContext(
        retrieval=reranker_pipeline,
        qdrant_client=qdrant_client,
        qdrant_collection=index_config.collection_name,
        acronyms=acronyms,
        deadlines=deadlines,
    )


async def _main() -> None:
    config = BotConfig.from_env()
    setup_logging()

    logger.info(f"bot boot: knowledge_dir={config.knowledge_dir} max_tools={config.max_tool_calls}")
    agent_context = _build_agent_context(config)
    logger.info(
        f"agent context ready: deadlines={len(agent_context.deadlines)} "
        f"acronyms={len(agent_context.acronyms)}"
    )

    conversation_store = ConversationStore(
        db_path=config.conversation_db_path,
        max_messages=config.conversation_max_messages,
        ttl=timedelta(minutes=config.conversation_ttl_minutes),
    )

    pii_detector = build_default_detector(enable_ner=True)
    logger.info(f"pii detector ready: ner_active={pii_detector.ner_pipeline is not None}")

    set_deps(
        HandlersDeps(
            agent_context=agent_context,
            conversation_store=conversation_store,
            max_tool_calls=config.max_tool_calls,
            pii_detector=pii_detector,
        )
    )

    bot = Bot(
        token=config.telegram_token,
        default=DefaultBotProperties(parse_mode=None),
    )
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("bot starting")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


def run() -> None:
    asyncio.run(_main())
