from __future__ import annotations

from rag.experiments.runner import RetrievalPipeline, chunk_key_from_hit
from rag.schemas import RetrievalHit


class DedupingPipeline:
    def __init__(self, *, base: RetrievalPipeline, pool_multiplier: int = 4) -> None:
        self._base = base
        self._pool_multiplier = pool_multiplier

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        raw = self._base.search(query, top_k=top_k * self._pool_multiplier)
        seen: set[str] = set()
        unique: list[RetrievalHit] = []
        for hit in raw:
            key = chunk_key_from_hit(hit)
            if key in seen:
                continue
            seen.add(key)
            unique.append(hit)
            if len(unique) >= top_k:
                break
        return unique
