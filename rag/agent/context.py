from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter
from qdrant_client import QdrantClient

from parser.knowledge_schemas import KnowledgeDoc
from parser.schemas import DeadlineEntry
from rag.schemas import RetrievalHit

_KNOWLEDGE_DOC_ADAPTER: TypeAdapter[KnowledgeDoc] = TypeAdapter(KnowledgeDoc)


class SearchPipeline(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]: ...


@dataclass
class AgentContext:
    retrieval: SearchPipeline
    qdrant_client: QdrantClient
    qdrant_collection: str
    acronyms: dict[str, str]
    deadlines: list[DeadlineEntry] = field(default_factory=list)


def load_deadlines(knowledge_dir: Path) -> list[DeadlineEntry]:
    out: list[DeadlineEntry] = []
    if not knowledge_dir.exists():
        return out
    for json_file in sorted(knowledge_dir.glob("*.json")):
        doc = _KNOWLEDGE_DOC_ADAPTER.validate_json(json_file.read_text(encoding="utf-8"))
        out.extend(getattr(doc, "deadlines", []))
    return out
