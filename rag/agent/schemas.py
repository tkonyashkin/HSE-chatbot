from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.schemas import ChunkType

MatchStatus = Literal["ok", "incompatible", "no_data"]

AggregateOperation = Literal["count", "mean", "median", "min", "max", "sum"]
AnalysisField = Literal[
    "tuition",
    "passing_score_budget",
    "budget_places",
    "paid_places",
]
GroupByField = Literal["none", "faculty"]


class AnalysisBucket(BaseModel):
    key: str
    value: float | None
    n_programs: int
    top_examples: list[str] = Field(default_factory=list, max_length=5)


class AnalysisResult(BaseModel):
    operation: AggregateOperation
    field: AnalysisField
    group_by: GroupByField
    n_programs_total: int
    n_programs_missing_field: int
    buckets: list[AnalysisBucket] = Field(default_factory=list)
    notes: str = ""


class ProgramHit(BaseModel):
    program_slug: str
    program_name: str
    faculty: str
    chunk_type: ChunkType
    text_preview: str = Field(max_length=500)
    source_url: str
    score: float


class ProgramSummary(BaseModel):
    program_slug: str
    program_name: str
    faculty: str
    code: str
    passing_score_budget: int | None
    tuition_rub: int | None
    budget_places: int | None
    exams: list[str]
    source_url: str


class ProgramFilter(BaseModel):
    faculty_query: str | None = None
    min_passing_score_budget: int | None = None
    max_passing_score_budget: int | None = None
    max_tuition_rub: int | None = None
    has_subject: str | None = None
    year: int = 2025


class ProgramDetails(BaseModel):
    program: ProgramSummary
    sections: dict[str, str]
    extracted_facts: dict[str, Any]
    source_url: str


Aspect = str


class ComparisonTable(BaseModel):
    programs: list[ProgramSummary]
    comparison: dict[str, dict[str, str]]


class YearScore(BaseModel):
    year: int
    budget: int | None
    paid: int | None = None


class MatchHit(BaseModel):
    program_slug: str
    program_name: str
    faculty: str
    status: MatchStatus
    user_sum: int | None
    passing_score_budget: int | None
    passing_year: int | None
    passing_score_history: list[YearScore] = Field(default_factory=list)
    reason: str | None
    notes: list[str] = Field(default_factory=list)
    source_url: str


class MatchResult(BaseModel):
    based_on_year: int | None
    disclaimer: str
    results: list[MatchHit] = Field(default_factory=list)


class AdmissionDeadline(BaseModel):
    financing: Literal["budget", "paid"]
    category: Literal["ege_only", "with_hse_exams"]
    start_date: str
    end_date: str
    description: str
    source_url: str
