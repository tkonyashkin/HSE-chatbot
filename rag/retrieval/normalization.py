from __future__ import annotations

import re
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9.]+")
CYRILLIC_RE = re.compile(r"[а-я]")
_MORPH: Any | None = None


def get_morph() -> Any:
    global _MORPH
    if _MORPH is None:
        import pymorphy3

        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH


def normalize_ru_tokens(
    text: str,
    stopwords: set[str] | None = None,
    normalizer: str = "pymorphy3",
) -> list[str]:
    blocked = stopwords or set()
    tokens = [match.group(0).lower().replace("ё", "е") for match in TOKEN_RE.finditer(text)]
    tokens = [token for token in tokens if token and token not in blocked]
    if normalizer == "raw":
        return tokens
    if normalizer != "pymorphy3":
        raise ValueError(f"unsupported normalizer: {normalizer}")

    morph = get_morph()
    normalized: list[str] = []
    for token in tokens:
        if CYRILLIC_RE.search(token):
            normalized.append(morph.parse(token)[0].normal_form.replace("ё", "е"))
        else:
            normalized.append(token)
    return normalized
