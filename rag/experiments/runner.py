from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rag.experiments.dataset import Query, SourceLabel
from rag.experiments.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.schemas import RetrievalHit


class RetrievalPipeline(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]: ...


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    query_type: str
    preprocessed_query: str
    retrieved_keys: list[str]
    hit_at_5: int
    mrr_at_5: float
    recall_at_20: float


@dataclass(frozen=True)
class VariantResult:
    variant: str
    rows: list[QueryResult]
    mean_hit_at_5: float
    mean_mrr_at_5: float
    mean_recall_at_20: float


def chunk_key_from_hit(hit: RetrievalHit) -> str:
    chunk = hit.chunk
    if chunk.chunk_type.value == "knowledge":
        return chunk.program_slug
    return f"{chunk.program_slug}:{chunk.chunk_type.value}"


def run_variant(
    *,
    variant: str,
    queries: list[Query],
    source_labels: dict[str, list[SourceLabel]],
    preprocess: Callable[[str], str],
    pipeline: RetrievalPipeline,
    pool_size: int = 20,
) -> VariantResult:
    rows: list[QueryResult] = []
    for query in queries:
        if query.expected_refusal:
            continue
        relevant_keys = {label.chunk_key for label in source_labels.get(query.query_id, [])}
        preprocessed = preprocess(query.text)
        hits = pipeline.search(preprocessed, top_k=pool_size)
        retrieved_keys = [chunk_key_from_hit(hit) for hit in hits]
        rows.append(
            QueryResult(
                query_id=query.query_id,
                query_type=query.query_type,
                preprocessed_query=preprocessed,
                retrieved_keys=retrieved_keys,
                hit_at_5=hit_at_k(retrieved_keys, relevant_keys, k=5),
                mrr_at_5=reciprocal_rank(retrieved_keys, relevant_keys, k=5),
                recall_at_20=recall_at_k(retrieved_keys, relevant_keys, k=20),
            )
        )
    if not rows:
        return VariantResult(
            variant=variant, rows=[], mean_hit_at_5=0.0, mean_mrr_at_5=0.0, mean_recall_at_20=0.0
        )
    return VariantResult(
        variant=variant,
        rows=rows,
        mean_hit_at_5=sum(r.hit_at_5 for r in rows) / len(rows),
        mean_mrr_at_5=sum(r.mrr_at_5 for r in rows) / len(rows),
        mean_recall_at_20=sum(r.recall_at_20 for r in rows) / len(rows),
    )
