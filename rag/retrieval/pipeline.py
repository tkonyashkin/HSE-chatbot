from dataclasses import dataclass

from rag.embedding.embedder import Embedder
from rag.indexing.indexer import IndexerConfig
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.dense import dense_search
from rag.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from rag.retrieval.rrf import reciprocal_rank_fusion
from rag.schemas import RetrievalHit


@dataclass
class RetrievalPipeline:
    embedder: Embedder
    index_config: IndexerConfig
    bm25_retriever: BM25Retriever | None = None
    reranker: CrossEncoderReranker | None = None

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        mode = self.index_config.mode
        if mode == "dense_only":
            return self.dense(query, top_k)
        if mode == "dense_rerank":
            dense_hits = self.dense(query, self.index_config.prefetch_limit)
            return self.rerank(query, dense_hits, top_k)
        if mode == "hybrid_rrf":
            return self.hybrid(query, top_k)
        if mode == "hybrid_rrf_rerank":
            fused_hits = self.hybrid(query, self.index_config.prefetch_limit)
            return self.rerank(query, fused_hits, top_k)
        raise ValueError(f"invalid retrieval mode: {mode}")

    def dense(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_vec = self.embedder.encode_query(query)
        return dense_search(query_vec, top_k=top_k, config=self.index_config)

    def hybrid(self, query: str, top_k: int) -> list[RetrievalHit]:
        bm25 = self.require_bm25()
        dense_hits = self.dense(query, self.index_config.prefetch_limit)
        bm25_hits = bm25.search(query, top_k=self.index_config.prefetch_limit)
        return reciprocal_rank_fusion(
            [dense_hits, bm25_hits],
            k=self.index_config.rrf_k,
            top_k=top_k,
        )

    def require_bm25(self) -> BM25Retriever:
        if self.bm25_retriever is None:
            raise ValueError("BM25 retriever is required for hybrid retrieval modes")
        return self.bm25_retriever

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        reranker = self.reranker
        if reranker is None:
            reranker = CrossEncoderReranker(
                RerankerConfig(
                    model_name=self.index_config.reranker_model_name,
                    top_n=self.index_config.reranker_top_n,
                    device=self.index_config.reranker_device,
                )
            )
            self.reranker = reranker
        return reranker.rerank(query, hits, top_k=top_k)
