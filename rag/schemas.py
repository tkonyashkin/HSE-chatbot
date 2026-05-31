from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ChunkType(StrEnum):
    overview = "overview"
    curriculum = "curriculum"
    advantages = "advantages"
    career = "career"
    admission = "admission"
    knowledge = "knowledge"


class Chunk(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=200)
    chunk_type: ChunkType
    text: str = Field(min_length=1)
    embedding_input: str = Field(min_length=1)
    program_slug: str
    program_name: str
    program_code: str
    faculty: str
    admission_year: int = Field(ge=2020, le=2100)
    score_years: list[int] = Field(default_factory=list)
    extracted_facts: dict[str, Any] = Field(default_factory=dict)
    source_doc_type: str | None = None
    profile_name: str | None = None
    section_index: int | None = Field(default=None, ge=0)
    url: HttpUrl
    raw_html_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    created_at: datetime | None = None

    @field_validator("score_years")
    @classmethod
    def _validate_years(cls, v: list[int]) -> list[int]:
        for y in v:
            if not (2000 <= y <= 2100):
                raise ValueError(f"score year {y} outside [2000, 2100]")
        return v


class RetrievalHit(BaseModel):
    chunk: Chunk
    score: float = Field(ge=0.0)
    rank: int = Field(ge=1)


class Answer(BaseModel):
    text: str = Field(min_length=1)
    source_urls: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
