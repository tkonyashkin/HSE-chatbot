from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

SCROLL_BATCH = 256


def list_unique_programs(client: QdrantClient, collection: str) -> list[dict[str, Any]]:
    program_filter = Filter(
        must=[FieldCondition(key="chunk_type", match=MatchValue(value="overview"))]
    )
    records: list[dict[str, Any]] = []
    next_offset: Any = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=program_filter,
            limit=SCROLL_BATCH,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            if point.payload is not None:
                records.append(point.payload)
        if next_offset is None:
            break
    return records


def get_program_facts(client: QdrantClient, collection: str, slug: str) -> dict[str, Any] | None:
    program_filter = Filter(
        must=[
            FieldCondition(key="program_slug", match=MatchValue(value=slug)),
            FieldCondition(key="chunk_type", match=MatchValue(value="overview")),
        ]
    )
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=program_filter,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if points and points[0].payload is not None:
        return points[0].payload
    return None


def get_program_section_text(
    client: QdrantClient, collection: str, slug: str, chunk_type: str
) -> str | None:
    program_filter = Filter(
        must=[
            FieldCondition(key="program_slug", match=MatchValue(value=slug)),
            FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)),
        ]
    )
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=program_filter,
        limit=SCROLL_BATCH,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    payloads: list[dict[str, Any]] = [
        point.payload for point in points if point.payload is not None
    ]
    payloads.sort(key=lambda pl: pl.get("section_index") or 0)
    return "\n\n".join(pl.get("text", "") for pl in payloads).strip() or None
