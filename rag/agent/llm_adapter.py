from __future__ import annotations

import os

from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def build_pydantic_ai_model(model_override: str | None = None) -> Model:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    model_name = (
        model_override or os.environ.get("OPENROUTER_MODEL") or "anthropic/claude-haiku-4.5"
    )
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    provider = OpenAIProvider(openai_client=client)
    return OpenAIChatModel(model_name, provider=provider)
