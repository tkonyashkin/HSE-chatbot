from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from rag.agent.query_normalization import expand_acronyms_in_query


def raw(query: str) -> str:
    return query


def expand_acronyms(text: str, acronyms: dict[str, str]) -> str:
    return expand_acronyms_in_query(text, acronyms)


@lru_cache(maxsize=1)
def morph_analyzer() -> Any:
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


def lemma(text: str) -> str:
    morph = morph_analyzer()
    pieces: list[str] = []
    for token in re.findall(r"\w+|\W+", text, re.UNICODE):
        if token.isalpha() and not token.isupper():
            pieces.append(morph.parse(token)[0].normal_form)
        else:
            pieces.append(token)
    return "".join(pieces)


def lexical(query: str, acronyms: dict[str, str]) -> str:
    return expand_acronyms(lemma(query), acronyms)


def load_acronyms(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data.get("universal", {}))
