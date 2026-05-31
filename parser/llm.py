from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel
from pydantic_ai.models import Model
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

T = TypeVar("T", bound=BaseModel)


class LLMResponseError(Exception):
    def __init__(self, message: str, raw_content: str = ""):
        super().__init__(message)
        self.raw_content = raw_content


class LLMClient(Protocol):
    def extract(self, prompt: str, text: str, schema: type[T]) -> T: ...


@dataclass
class MockLLMClient:
    canned: dict[str, BaseModel]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def extract(self, prompt: str, text: str, schema: type[T]) -> T:
        self.calls.append({"prompt": prompt, "text": text, "schema": schema.__name__})
        key = "any" if "any" in self.canned else schema.__name__
        if key not in self.canned:
            raise KeyError(f"No canned response for {schema.__name__}")
        result = self.canned[key]
        if not isinstance(result, schema):
            raise TypeError(f"Expected {schema.__name__}, got {type(result).__name__}")
        return result


def build_parser_model(model_override: str | None = None) -> Model:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    model_name = (
        model_override or os.environ.get("OPENROUTER_MODEL") or "anthropic/claude-sonnet-4.6"
    )
    provider = OpenRouterProvider(api_key=api_key)
    return OpenRouterModel(model_name, provider=provider)
