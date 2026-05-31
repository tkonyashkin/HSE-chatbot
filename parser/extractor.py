from __future__ import annotations

import time
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from parser.cache import Cache, CacheKey, build_cache_payload, compute_schema_hash
from parser.llm import LLMClient, build_parser_model
from parser.prompt_loader import load_extraction_prompt
from parser.quoted_value import QuotedFloat, QuotedInt

logger = structlog.get_logger("parser.extractor")

SchemaT = TypeVar("SchemaT", bound=BaseModel)

SOURCE_QUOTE_FIELDS = {"raw_quote", "row_quote"}


class QuoteNotInSourceError(ValueError):
    pass


def extract(
    markdown_text: str,
    schema_type: type[SchemaT],
    prompt_name: str,
    *,
    llm: LLMClient | None = None,
    cache: Cache | None = None,
    model_id: str = "anthropic/claude-sonnet-4.6",
    prompts_dir: Path = Path("parser/prompts"),
) -> SchemaT:
    prompt_meta = load_extraction_prompt(prompt_name, prompts_dir)
    user_message = str(prompt_meta["user_template"]).format(markdown_text=markdown_text)
    model_hints = prompt_meta.get("model_hints", {})
    temperature = model_hints.get("temperature", 0)
    max_tokens = model_hints.get("max_tokens", 4096)
    schema_hash = compute_schema_hash(schema_type)
    cache_key = CacheKey.compute(
        model_id=model_id,
        system_prompt=str(prompt_meta["system"]),
        user_message=user_message,
        response_schema_hash=schema_hash,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    start = time.time()
    if cache is not None:
        cached = cache.get(cache_key, model_id)
        if cached is not None:
            result = validate_result(schema_type, cached["extracted_object"], markdown_text)
            log_extraction(
                "hit",
                model_id,
                schema_type.__name__,
                cache_key,
                int((time.time() - start) * 1000),
            )
            return result

    result_instance, openrouter_response = run_extraction(
        schema_type=schema_type,
        prompt_meta=prompt_meta,
        user_message=user_message,
        llm=llm,
        model_id=model_id,
    )
    result = validate_result(schema_type, result_instance.model_dump(mode="json"), markdown_text)
    latency_ms = int((time.time() - start) * 1000)

    if cache is not None:
        payload = build_cache_payload(
            cache_meta={
                "model_id": model_id,
                "response_schema_hash": schema_hash,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            openrouter_response=openrouter_response,
            extracted_object=result.model_dump(mode="json"),
        )
        cache.put(cache_key, model_id, payload)

    log_extraction("miss", model_id, schema_type.__name__, cache_key, latency_ms)
    return result


def run_extraction(
    *,
    schema_type: type[SchemaT],
    prompt_meta: dict[str, Any],
    user_message: str,
    llm: LLMClient | None,
    model_id: str,
) -> tuple[SchemaT, dict[str, Any]]:
    if llm is not None:
        return llm.extract(str(prompt_meta["system"]), user_message, schema_type), {}

    model = build_parser_model(model_id)
    model_hints = prompt_meta.get("model_hints", {})
    agent: Agent[None, SchemaT] = Agent(
        model=model,
        system_prompt=str(prompt_meta["system"]),
        output_type=schema_type,
        model_settings=ModelSettings(
            temperature=float(model_hints.get("temperature", 0)),
            max_tokens=int(model_hints.get("max_tokens", 4096)),
        ),
    )
    run_result = agent.run_sync(user_message)
    return run_result.output, serialize_run_result(run_result)


def validate_result(
    schema_type: type[SchemaT],
    data: Any,
    source_markdown: str,
) -> SchemaT:
    result = schema_type.model_validate(data)
    validate_quotes_in_source(result, source_markdown)
    return result


def validate_quotes_in_source(value: Any, source_markdown: str, path: str = "") -> None:
    if isinstance(value, QuotedInt | QuotedFloat):
        check_literal_quote(value.quote, source_markdown, path or "quote")
        return

    if isinstance(value, BaseModel):
        for field_name, field_value in value:
            child_path = f"{path}.{field_name}" if path else field_name
            if field_name in SOURCE_QUOTE_FIELDS and isinstance(field_value, str):
                check_literal_quote(field_value, source_markdown, child_path)
            validate_quotes_in_source(field_value, source_markdown, child_path)
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_quotes_in_source(item, source_markdown, f"{path}[{index}]")
        return

    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            validate_quotes_in_source(item, source_markdown, child_path)


def check_literal_quote(quote: str, source_markdown: str, path: str) -> None:
    if quote not in source_markdown:
        raise QuoteNotInSourceError(f"quote not found in source_markdown at {path}: {quote!r}")


def serialize_run_result(run_result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    usage_attr = getattr(run_result, "usage", None)
    if usage_attr is not None:
        try:
            usage = usage_attr() if callable(usage_attr) else usage_attr
            payload["usage"] = to_jsonable(usage)
        except Exception:
            payload["usage"] = {}

    messages_attr = getattr(run_result, "all_messages", None)
    if callable(messages_attr):
        with suppress(Exception):
            payload["messages_count"] = len(messages_attr())
    return payload


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return {
            key: to_jsonable(item) for key, item in vars(value).items() if not key.startswith("_")
        }
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value


def log_extraction(
    cache_status: str,
    model_id: str,
    schema_name: str,
    cache_key: str,
    latency_ms: int,
) -> None:
    logger.info(
        "parser_extraction_complete",
        cache_status=cache_status,
        model_id=model_id,
        schema_name=schema_name,
        cache_key=cache_key[:16],
        latency_ms=latency_ms,
    )
