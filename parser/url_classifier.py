from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit

from parser.schemas import KnowledgeDocType, TableType

UrlRole = KnowledgeDocType | TableType | tuple[TableType, int]


def coerce_role(value: str | UrlRole | None) -> UrlRole | None:
    if value is None or isinstance(value, KnowledgeDocType | TableType | tuple):
        return value
    try:
        return KnowledgeDocType(value)
    except ValueError:
        pass
    try:
        return TableType(value)
    except ValueError:
        pass
    if value.startswith("results:"):
        try:
            return (TableType.results, int(value.split(":", 1)[1]))
        except ValueError:
            return None
    return None


def classify_url(url: str, classifier: Mapping[str, str | UrlRole]) -> UrlRole | None:
    path = urlsplit(url).path.rstrip("/")
    return coerce_role(classifier.get(path))


__all__ = ["UrlRole", "classify_url"]
