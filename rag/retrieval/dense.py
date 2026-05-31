import numpy as np

from rag.indexing.indexer import IndexerConfig, search_index
from rag.schemas import RetrievalHit


def dense_search(query_vec: np.ndarray, top_k: int, config: IndexerConfig) -> list[RetrievalHit]:
    return search_index(query_vec, top_k=top_k, config=config)
