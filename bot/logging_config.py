from __future__ import annotations

import hashlib
import logging
import os


def setup_logging() -> None:
    level = os.environ.get("HSE_BOT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def hash_user_id(user_id: int) -> str:
    digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
    return digest[:16]
