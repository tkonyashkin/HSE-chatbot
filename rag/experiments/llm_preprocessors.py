from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import yaml
from openai import OpenAI

from rag.experiments.tracking import CallTracker, now_ms


def load_prompt_template(path: Path) -> str:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(raw["prompt"])


def default_llm_call(
    model_id: str | None = None,
    tracker: CallTracker | None = None,
    extra_body: dict[str, object] | None = None,
) -> Callable[[str], str]:
    api_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    model = model_id or os.environ.get("OPENROUTER_MODEL") or "anthropic/claude-haiku-4.5"

    def call(prompt: str) -> str:
        t0 = now_ms()
        if extra_body is not None:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                extra_body=extra_body,
            )
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        latency = now_ms() - t0
        if tracker is not None and resp.usage is not None:
            tracker.record(
                latency_ms=latency,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
            )
        return resp.choices[0].message.content or ""

    return call
