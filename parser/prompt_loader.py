from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PROMPT_CACHE: dict[tuple[str, Path], dict[str, Any]] = {}


def load_extraction_prompt(
    name: str,
    prompts_dir: Path = Path("parser/prompts"),
) -> dict[str, Any]:
    cache_key = (name, prompts_dir)
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]

    shared = yaml.safe_load((prompts_dir / "shared.yaml").read_text(encoding="utf-8"))
    specific = yaml.safe_load((prompts_dir / f"{name}.yaml").read_text(encoding="utf-8"))

    merged: dict[str, Any] = {
        "system": shared["system"].rstrip() + "\n\n" + specific["system"].strip(),
        "user_template": specific["user_template"],
        "model_hints": {
            **shared.get("model_hints", {}),
            **specific.get("model_hints", {}),
        },
    }
    _PROMPT_CACHE[cache_key] = merged
    return merged


def clear_prompt_cache() -> None:
    _PROMPT_CACHE.clear()
