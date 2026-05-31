from dataclasses import dataclass, field
from typing import Protocol

from qdrant_client import QdrantClient

from parser.schemas import DeadlineEntry
from rag.schemas import RetrievalHit


class SearchPipeline(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]: ...


@dataclass
class AgentContext:
    retrieval: SearchPipeline
    qdrant_client: QdrantClient
    qdrant_collection: str
    acronyms: dict[str, str]
    deadlines: list[DeadlineEntry] = field(default_factory=list)
