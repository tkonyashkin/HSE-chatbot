import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BotConfig:
    telegram_token: str
    knowledge_dir: Path
    embedding_config_path: Path
    retrieval_config_path: Path
    acronyms_config_path: Path
    conversation_max_messages: int
    conversation_ttl_minutes: int
    max_tool_calls: int
    conversation_db_path: Path

    @classmethod
    def from_env(cls) -> "BotConfig":
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN env var is required")
        db_raw = os.environ.get("HSE_BOT_CONVERSATION_DB")
        if not db_raw:
            raise RuntimeError(
                "HSE_BOT_CONVERSATION_DB env var is required. "
                "Установите путь к SQLite-файлу (например, data/bot/conversations.db)."
            )
        return cls(
            telegram_token=token,
            knowledge_dir=Path(os.environ.get("HSE_KNOWLEDGE_DIR", "data/knowledge")),
            embedding_config_path=Path(
                os.environ.get("HSE_EMBEDDING_CONFIG", "config/embedding.yaml")
            ),
            retrieval_config_path=Path(
                os.environ.get("HSE_RETRIEVAL_CONFIG", "config/retrieval.yaml")
            ),
            acronyms_config_path=Path(
                os.environ.get("HSE_ACRONYMS_CONFIG", "config/acronyms.yaml")
            ),
            conversation_max_messages=int(os.environ.get("HSE_BOT_HISTORY_MESSAGES", "5")),
            conversation_ttl_minutes=int(os.environ.get("HSE_BOT_HISTORY_TTL_MIN", "30")),
            max_tool_calls=int(os.environ.get("HSE_BOT_MAX_TOOL_CALLS", "4")),
            conversation_db_path=Path(db_raw),
        )
