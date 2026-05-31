from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from parser.knowledge_schemas import KnowledgeDoc
from parser.schemas import Program
from rag.chunking.chunker import ChunkerConfig, chunk_knowledge_doc, chunk_program

KNOWLEDGE_DOC_ADAPTER: TypeAdapter[KnowledgeDoc] = TypeAdapter(KnowledgeDoc)


@dataclass(frozen=True)
class ChunkWriteResult:
    chunk_count: int
    program_chunk_count: int
    knowledge_chunk_count: int
    profile_name: str
    output_path: Path
    bytes_written: int


def write_chunks(
    *,
    programs_dir: Path,
    knowledge_dir: Path,
    output_path: Path,
    config_path: Path,
    chunk_profile: str | None = None,
) -> ChunkWriteResult:
    config = ChunkerConfig.from_yaml(config_path, profile_name=chunk_profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    program_count = 0
    knowledge_count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for json_file in sorted(programs_dir.glob("*.json")):
            program = Program.model_validate_json(json_file.read_text(encoding="utf-8"))
            for chunk in chunk_program(program, config):
                out.write(chunk.model_dump_json() + "\n")
                program_count += 1
        if knowledge_dir.exists():
            for json_file in sorted(knowledge_dir.glob("*.json")):
                doc = KNOWLEDGE_DOC_ADAPTER.validate_json(json_file.read_text(encoding="utf-8"))
                for chunk in chunk_knowledge_doc(doc, config):
                    out.write(chunk.model_dump_json() + "\n")
                    knowledge_count += 1
    return ChunkWriteResult(
        chunk_count=program_count + knowledge_count,
        program_chunk_count=program_count,
        knowledge_chunk_count=knowledge_count,
        profile_name=config.profile_name,
        output_path=output_path,
        bytes_written=output_path.stat().st_size,
    )
