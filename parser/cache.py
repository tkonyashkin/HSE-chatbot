from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

logger = structlog.get_logger("parser.cache")


class CacheKey:
    @staticmethod
    def compute(
        model_id: str,
        system_prompt: str,
        user_message: str,
        response_schema_hash: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model_id": model_id,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "response_schema_hash": response_schema_hash,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_schema_hash(schema: type[BaseModel]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(schema_json.encode("utf-8")).hexdigest()


def model_safe_id(model_id: str) -> str:
    return model_id.replace("/", "_")


class Cache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str, model_id: str) -> Path:
        return self.cache_dir / model_safe_id(model_id) / f"{key}.json"

    def get(self, key: str, model_id: str) -> dict[str, Any] | None:
        path = self.path_for(key, model_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return payload

    def put(self, key: str, model_id: str, value: dict[str, Any]) -> Path:
        path = self.path_for(key, model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


def build_cache_payload(
    cache_meta: dict[str, Any],
    openrouter_response: dict[str, Any],
    extracted_object: dict[str, Any],
) -> dict[str, Any]:
    enriched_meta = {**cache_meta, "cached_at": datetime.now(UTC).isoformat()}
    return {
        "cache_meta": enriched_meta,
        "openrouter_response": openrouter_response,
        "extracted_object": extracted_object,
    }
