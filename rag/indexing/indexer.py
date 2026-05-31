from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.embedding.embedder import Embedder, EmbedderConfig
from rag.schemas import Chunk, RetrievalHit

RetrievalMode = Literal["dense_only", "dense_rerank", "hybrid_rrf", "hybrid_rrf_rerank"]


@dataclass
class IndexerConfig:
    qdrant_path: Path
    collection_name: str
    dim: int = 1024
    top_k: int = 5
    mode: RetrievalMode = "dense_only"
    prefetch_limit: int = 20
    rrf_k: int = 60
    bm25_normalizer: str = "pymorphy3"
    bm25_stopwords: list[str] = field(default_factory=list)
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_n: int = 5
    reranker_device: str | None = "cpu"

    @classmethod
    def from_yaml(cls, path: Path, dim: int = 1024) -> "IndexerConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = raw["retrieval"]
        mode = cfg.get("mode", "dense_only")
        if mode not in ("dense_only", "dense_rerank", "hybrid_rrf", "hybrid_rrf_rerank"):
            raise ValueError(f"invalid retrieval mode: {mode}")
        bm25 = cfg.get("bm25", {}) or {}
        reranker = cfg.get("reranker", {}) or {}
        return cls(
            qdrant_path=Path(cfg["qdrant_path"]),
            collection_name=cfg["collection_name"],
            dim=dim,
            top_k=int(cfg.get("top_k", 5)),
            mode=mode,
            prefetch_limit=int(cfg.get("prefetch_limit", 20)),
            rrf_k=int(cfg.get("rrf_k", 60)),
            bm25_normalizer=str(bm25.get("normalizer", "pymorphy3")),
            bm25_stopwords=[str(item) for item in bm25.get("stopwords", [])],
            reranker_model_name=str(reranker.get("model_name", "BAAI/bge-reranker-v2-m3")),
            reranker_top_n=int(reranker.get("top_n", 5)),
            reranker_device=reranker.get("device", "cpu"),
        )


@dataclass(frozen=True)
class IndexBuildResult:
    chunk_count: int
    collection_name: str


_CLIENT_CACHE: dict[str, QdrantClient] = {}


def _client(qdrant_path: Path) -> QdrantClient:
    qdrant_path.mkdir(parents=True, exist_ok=True)
    key = str(qdrant_path)
    cached = _CLIENT_CACHE.get(key)
    if cached is None:
        cached = QdrantClient(path=key)
        _CLIENT_CACHE[key] = cached
    return cached


def reset_client_cache() -> None:
    import contextlib

    for client in _CLIENT_CACHE.values():
        with contextlib.suppress(Exception):
            client.close()
    _CLIENT_CACHE.clear()


def build_index(chunks: list[Chunk], embeddings: np.ndarray, config: IndexerConfig) -> None:
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) count mismatch"
        )
    client = _client(config.qdrant_path)
    if client.collection_exists(config.collection_name):
        client.delete_collection(config.collection_name)
    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=VectorParams(size=config.dim, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload=chunk.model_dump(mode="json"),
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=config.collection_name, points=points, wait=True)


def load_chunks_jsonl(chunks_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk.model_validate_json(line))
    return chunks


def run_index(
    chunks_path: Path,
    embedding_config_path: Path,
    retrieval_config_path: Path,
) -> IndexBuildResult:
    emb_config = EmbedderConfig.from_yaml(embedding_config_path)
    embedder = Embedder(emb_config)
    chunks = load_chunks_jsonl(chunks_path)
    embeddings = embedder.encode_passages([c.embedding_input for c in chunks])
    index_config = IndexerConfig.from_yaml(retrieval_config_path, dim=emb_config.dim)
    build_index(chunks, embeddings, index_config)
    return IndexBuildResult(
        chunk_count=len(chunks),
        collection_name=index_config.collection_name,
    )


def search_index(query_vec: np.ndarray, top_k: int, config: IndexerConfig) -> list[RetrievalHit]:
    client = _client(config.qdrant_path)
    result = client.query_points(
        collection_name=config.collection_name,
        query=query_vec.tolist(),
        limit=top_k,
        with_payload=True,
    )
    hits: list[RetrievalHit] = []
    for rank, point in enumerate(result.points, start=1):
        chunk = Chunk.model_validate(point.payload)
        hits.append(RetrievalHit(chunk=chunk, score=point.score, rank=rank))
    return hits
