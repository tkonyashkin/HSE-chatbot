from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag.retrieval.normalization import normalize_ru_tokens
from rag.schemas import Chunk, RetrievalHit


@dataclass
class BM25Retriever:
    chunks: list[Chunk]
    tokenized_corpus: list[list[str]]
    stopwords: set[str]
    normalizer: str = "pymorphy3"

    def __post_init__(self) -> None:
        self._bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        stopwords: set[str] | None = None,
        normalizer: str = "pymorphy3",
    ) -> BM25Retriever:
        blocked = stopwords or set()
        tokenized = [
            normalize_ru_tokens(chunk.embedding_input or chunk.text, blocked, normalizer)
            for chunk in chunks
        ]
        return cls(
            chunks=chunks,
            tokenized_corpus=tokenized,
            stopwords=blocked,
            normalizer=normalizer,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if self._bm25 is None or not self.chunks:
            return []
        query_tokens = normalize_ru_tokens(query, self.stopwords, self.normalizer)
        if not query_tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float32)
        order = sorted(range(len(self.chunks)), key=lambda index: (-float(scores[index]), index))
        hits: list[RetrievalHit] = []
        for rank, index in enumerate(order[:top_k], start=1):
            hits.append(
                RetrievalHit(
                    chunk=self.chunks[index],
                    score=max(float(scores[index]), 0.0),
                    rank=rank,
                )
            )
        return hits
