from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from parser.quoted_value import QuotedInt
from parser.schemas import DeadlineEntry, ExtractionMetadata, KnowledgeDocType, QAPair


class RuleCategory(StrEnum):
    eligibility = "eligibility"
    documents = "documents"
    deadlines = "deadlines"
    grading = "grading"
    other = "other"


class OlympiadLevel(StrEnum):
    level_1 = "1"
    level_2 = "2"
    level_3 = "3"
    vsosh = "vsosh"


class OlympiadBenefit(StrEnum):
    bvi = "bvi"
    score_100 = "score_100"
    score_other = "score_other"


class ExamFormat(StrEnum):
    written = "written"
    oral = "oral"
    test = "test"
    creative = "creative"
    mixed = "mixed"


class Rule(BaseModel):
    text: str = Field(min_length=10, max_length=2000)
    category: RuleCategory
    raw_quote: str = Field(min_length=1, max_length=2000)


class OlympiadEntry(BaseModel):
    name: str = Field(min_length=2, max_length=300)
    level: OlympiadLevel
    subjects: list[str] = Field(default_factory=list)
    benefit: OlympiadBenefit
    eligible_programs: list[str] = Field(default_factory=list)
    raw_quote: str = Field(min_length=1, max_length=2000)


class PortfolioComponent(BaseModel):
    name: str = Field(min_length=2, max_length=500)
    max_points: QuotedInt
    criteria_text: str = Field(min_length=10, max_length=2000)
    evidence_required: list[str] = Field(default_factory=list)
    applicable_programs: list[str] = Field(default_factory=list)
    raw_quote: str = Field(min_length=1, max_length=2000)


class ExamDescription(BaseModel):
    name: str = Field(min_length=2, max_length=300)
    subjects: list[str] = Field(default_factory=list)
    duration_min: QuotedInt
    format: ExamFormat
    topics: list[str] = Field(default_factory=list)
    has_program_pdf: bool = Field(default=False)
    has_demo_version: bool = Field(default=False)
    raw_quote: str = Field(min_length=1, max_length=2000)


class KnowledgeDocBase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doc_id: str = Field(default="knowledge", min_length=1, max_length=100)
    title: str = Field(default="Knowledge document", min_length=2, max_length=300)
    url: HttpUrl = HttpUrl("https://ba.hse.ru/")
    admission_year: int = Field(default=2026, ge=2020, le=2100)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_html_hash: str = Field(
        default="sha256:" + "0" * 64,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    generation_metadata: ExtractionMetadata


class InformationDoc(KnowledgeDocBase):
    doc_type: Literal[KnowledgeDocType.information] = KnowledgeDocType.information
    deadlines: list[DeadlineEntry] = Field(default_factory=list)
    general_rules: list[Rule] = Field(default_factory=list)
    text: str = Field(min_length=50)


class FAQDoc(KnowledgeDocBase):
    doc_type: Literal[KnowledgeDocType.faq] = KnowledgeDocType.faq
    qa_pairs: list[QAPair] = Field(min_length=1)


class OlympiadsDoc(KnowledgeDocBase):
    doc_type: Literal[KnowledgeDocType.olympiads] = KnowledgeDocType.olympiads
    olympiads: list[OlympiadEntry] = Field(min_length=1)
    text: str = Field(min_length=50)


class AchievementsDoc(KnowledgeDocBase):
    doc_type: Literal[KnowledgeDocType.achievements] = KnowledgeDocType.achievements
    portfolio_components: list[PortfolioComponent] = Field(min_length=1)
    aggregate_cap: QuotedInt
    text: str = Field(min_length=50)


class ExamProgramsDoc(KnowledgeDocBase):
    doc_type: Literal[KnowledgeDocType.exam_programs] = KnowledgeDocType.exam_programs
    exam_descriptions: list[ExamDescription] = Field(min_length=1)
    text: str = Field(min_length=50)


KnowledgeDoc = Annotated[
    InformationDoc | FAQDoc | OlympiadsDoc | AchievementsDoc | ExamProgramsDoc,
    Field(discriminator="doc_type"),
]
