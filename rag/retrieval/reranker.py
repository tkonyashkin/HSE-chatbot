from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from rag.schemas import RetrievalHit


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 5
    device: str | None = "cpu"


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig | None = None, model: Any | None = None):
        self.config = config or RerankerConfig()
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = CrossEncoder(self.config.model_name, device=self.config.device)
        return self._model

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.embedding_input) for hit in hits]
        raw_scores = np.asarray(self.model.predict(pairs), dtype=np.float32).reshape(-1)
        scored = list(zip(hits, raw_scores, strict=True))
        scored.sort(key=lambda item: (-float(item[1]), item[0].chunk.chunk_id))
        reranked: list[RetrievalHit] = []
        for rank, (hit, score) in enumerate(scored[:top_k], start=1):
            reranked.append(hit.model_copy(update={"score": max(float(score), 0.0), "rank": rank}))
        return reranked
