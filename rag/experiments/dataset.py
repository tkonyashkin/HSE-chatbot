from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    query_type: str
    expected_refusal: bool
    expected_answer_facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceLabel:
    query_id: str
    chunk_key: str


def load_queries(path: Path) -> list[Query]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Query(
            query_id=item.get("id", item.get("query_id", "")),
            text=item.get("text", item.get("query_text", "")),
            query_type=item.get("type", item.get("query_type", "")),
            expected_refusal=bool(item.get("refuse", item.get("expected_refusal", False))),
            expected_answer_facts=tuple(
                item.get("facts", item.get("expected_answer_facts", [])) or []
            ),
        )
        for item in data["queries"]
    ]


def load_source_labels(path: Path) -> dict[str, list[SourceLabel]]:
    grouped: dict[str, list[SourceLabel]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        query_id, chunk_key, *_ = line.split("\t")
        grouped.setdefault(query_id, []).append(SourceLabel(query_id=query_id, chunk_key=chunk_key))
    return grouped
