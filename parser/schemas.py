from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from parser.quoted_value import QuotedInt


class DegreeType(StrEnum):
    bachelor = "bachelor"
    specialitet = "specialitet"


class ProgramFlag(StrEnum):
    dual_degree = "dual_degree"
    english_taught = "english_taught"
    quota_only = "quota_only"
    cooperative = "cooperative"


class TableType(StrEnum):
    minkrit = "minkrit"
    kolmest = "kolmest"
    price = "price"
    results = "results"


class Exam(BaseModel):
    subject: str = Field(min_length=2, max_length=300)
    min_score: QuotedInt


class PassingScore(BaseModel):
    year: int = Field(ge=2000, le=2100)
    budget: QuotedInt | None = None
    paid: QuotedInt | None = None


class Places(BaseModel):
    budget: QuotedInt | None = None
    paid: QuotedInt | None = None
    target: QuotedInt | None = None
    special: QuotedInt | None = None
    separate: QuotedInt | None = None


class YearlyPlaces(BaseModel):
    year: int = Field(ge=2000, le=2100)
    places: Places


class YearlyTuition(BaseModel):
    year: int = Field(ge=2000, le=2100)
    tuition_fee_rub_per_year: QuotedInt


class TableProgramIdentity(BaseModel):
    program_url: str = Field(min_length=1, max_length=1000)
    program_slug: str = Field(min_length=1, max_length=100)
    program_label: str = Field(min_length=1, max_length=300)
    okso_code: str | None = Field(default=None, pattern=r"^\d{2}\.\d{2}\.\d{2}$")
    okso_label: str | None = Field(default=None, min_length=1, max_length=300)
    row_quote: str = Field(min_length=1, max_length=4000)


class TableMinkritRow(TableProgramIdentity):
    year: int = Field(ge=2000, le=2100)
    exams: list[Exam] = Field(default_factory=list)


class TableKolmestRow(TableProgramIdentity):
    year: int = Field(ge=2000, le=2100)
    places: Places = Field(default_factory=Places)


class TablePriceRow(TableProgramIdentity):
    year: int = Field(ge=2000, le=2100)
    tuition_fee_rub_per_year: QuotedInt


class TableResultsRow(TableProgramIdentity):
    passing_scores: list[PassingScore] = Field(default_factory=list)


class ProgramMetadata(BaseModel):
    flags: list[ProgramFlag] = Field(default_factory=list)


class ExtractionMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: str = Field(min_length=1)
    model_version_alias: str = Field(min_length=1)
    extraction_date: datetime


class KnowledgeDocType(StrEnum):
    information = "information"
    olympiads = "olympiads"
    achievements = "achievements"
    exam_programs = "exam_programs"
    faq = "faq"


class FinancingType(StrEnum):
    budget = "budget"
    paid = "paid"


class DeadlineCategory(StrEnum):
    ege_only = "ege_only"
    with_hse_exams = "with_hse_exams"


class DeadlineEntry(BaseModel):
    financing: FinancingType
    category: DeadlineCategory
    start_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    description: str = Field(min_length=2, max_length=500)
    source_url: str = Field(min_length=1)


class QAPair(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    answer: str = Field(min_length=2, max_length=4000)


class ProgramDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=2, max_length=300)
    codes: list[str] = Field(default_factory=list)
    faculty: str = Field(min_length=2, max_length=300)
    degree_type: DegreeType = DegreeType.bachelor

    duration: str | None = None
    form: str | None = None
    language: str | None = None

    description: str | None = None
    what_to_study: str | None = None
    advantages: str | None = None
    career: str | None = None
    admission_info: str | None = None

    exams: list[Exam] = Field(default_factory=list)
    places: Places = Field(default_factory=Places)
    passing_scores: list[PassingScore] = Field(default_factory=list)
    tuition_fee_rub_per_year: QuotedInt | None = None
    metadata: ProgramMetadata = Field(default_factory=ProgramMetadata)


class Program(ProgramDraft):
    slug: str = Field(min_length=1, max_length=100)
    url: HttpUrl
    admission_year: int = Field(ge=2020, le=2100)

    generation_metadata: ExtractionMetadata

    retrieved_at: datetime
    raw_html_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    places_history: list[YearlyPlaces] = Field(default_factory=list)
    tuition_history: list[YearlyTuition] = Field(default_factory=list)
