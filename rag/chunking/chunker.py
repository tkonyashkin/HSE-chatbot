from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from parser.knowledge_schemas import FAQDoc, KnowledgeDoc
from parser.quoted_value import QuotedInt
from parser.schemas import Places, Program
from rag.schemas import Chunk, ChunkType

KNOWLEDGE_MAX_CHARS = 1200
KNOWLEDGE_FACULTY = "Приёмная комиссия"

ContextHeaderMode = Literal["none", "simple", "llm"]
ShortSectionMode = Literal["merge_into_overview", "preserve"]


@dataclass
class ChunkerConfig:
    min_text_chars: int = 100
    context_header_mode: ContextHeaderMode = "simple"
    max_tokens_per_chunk: int = 460
    profile_name: str = "structural_context_header"
    short_section_mode: ShortSectionMode = "merge_into_overview"

    @classmethod
    def from_yaml(cls, path: Path, profile_name: str | None = None) -> "ChunkerConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = raw["chunking"]
        if "profiles" in cfg:
            profiles = cfg["profiles"] or {}
            selected = profile_name or cfg.get("default_profile", "structural_context_header")
            if selected not in profiles:
                raise ValueError(f"unknown chunking profile: {selected}")
            cfg = {**profiles[selected], "profile_name": selected}
        else:
            cfg = {**cfg, "profile_name": profile_name or cfg.get("profile_name", "default")}
        mode = cfg.get("context_header_mode", "simple")
        if mode not in ("none", "simple", "llm"):
            raise ValueError(f"invalid context_header_mode: {mode}")
        short_section_mode = cfg.get("short_section_mode", "merge_into_overview")
        if short_section_mode not in ("merge_into_overview", "preserve"):
            raise ValueError(f"invalid short_section_mode: {short_section_mode}")
        return cls(
            min_text_chars=int(cfg.get("min_text_chars", 100)),
            context_header_mode=mode,
            max_tokens_per_chunk=int(cfg.get("max_tokens_per_chunk", 460)),
            profile_name=str(cfg.get("profile_name", "default")),
            short_section_mode=short_section_mode,
        )


TYPE_TO_FIELD: dict[ChunkType, str] = {
    ChunkType.curriculum: "what_to_study",
    ChunkType.advantages: "advantages",
    ChunkType.career: "career",
    ChunkType.admission: "admission_info",
}


def build_simple_header(program: Program, chunk_type: ChunkType) -> str:
    primary_code = program.codes[0] if program.codes else ""
    return (
        f"Программа: «{program.name}» ({primary_code}) | "
        f"Факультет: {program.faculty} | "
        f"Раздел: {chunk_type.value} | "
        f"Год приёма: {program.admission_year}"
    )


def build_overview_text(program: Program, extras: list[str]) -> str:
    code = program.codes[0] if program.codes else ""
    parts = [f"{program.name} ({code}). {program.faculty}."]
    if program.description:
        parts.append(program.description)
    parts.extend(extras)
    return "\n\n".join(parts)


def make_chunk_id(slug: str, chunk_type: ChunkType, section_index: int | None = None) -> str:
    if section_index is not None:
        return f"{slug}_{chunk_type.value}_{section_index + 1:02d}"
    return f"{slug}_{chunk_type.value}"


def quoted_value(value: QuotedInt | None) -> int | None:
    return value.value if value is not None else None


def format_places(places: Places) -> dict[str, int | None]:
    return {
        "budget": quoted_value(places.budget),
        "paid": quoted_value(places.paid),
        "target": quoted_value(places.target),
        "special": quoted_value(places.special),
        "separate": quoted_value(places.separate),
    }


def format_extracted_facts(program: Program) -> dict[str, Any]:
    return {
        "exams": [
            {"subject": exam.subject, "min_score": exam.min_score.value} for exam in program.exams
        ],
        "passing_scores": [
            {
                "year": score.year,
                "budget": quoted_value(score.budget),
                "paid": quoted_value(score.paid),
            }
            for score in program.passing_scores
        ],
        "min_scores_by_subject": {exam.subject: exam.min_score.value for exam in program.exams},
        "places": format_places(program.places),
        "tuition_fee_rub_per_year": quoted_value(program.tuition_fee_rub_per_year),
        "codes": list(program.codes),
        "places_history": [
            {"year": yp.year, "places": format_places(yp.places)} for yp in program.places_history
        ],
        "tuition_history": [
            {
                "year": yt.year,
                "tuition_fee_rub_per_year": quoted_value(yt.tuition_fee_rub_per_year),
            }
            for yt in program.tuition_history
        ],
    }


def make_chunk(
    program: Program,
    chunk_type: ChunkType,
    text: str,
    config: ChunkerConfig,
    score_years: list[int],
    facts: dict[str, Any],
    now: datetime,
    section_index: int | None = None,
) -> Chunk:
    primary_code = program.codes[0] if program.codes else ""
    if config.context_header_mode == "simple":
        header = build_simple_header(program, chunk_type)
        embedding_input = f"{header}\n\n{text}"
    else:
        embedding_input = text
    return Chunk(
        chunk_id=make_chunk_id(program.slug, chunk_type, section_index),
        chunk_type=chunk_type,
        text=text,
        embedding_input=embedding_input,
        program_slug=program.slug,
        program_name=program.name,
        program_code=primary_code,
        faculty=program.faculty,
        admission_year=program.admission_year,
        score_years=score_years,
        extracted_facts=facts,
        source_doc_type="program",
        profile_name=config.profile_name,
        section_index=section_index,
        url=program.url,
        raw_html_hash=program.raw_html_hash,
        created_at=now,
    )


def max_chars_from_tokens(max_tokens: int) -> int:
    return max(300, max_tokens * 4)


def split_text_by_soft_cap(text: str, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text]
    return batch_paragraphs(paragraphs, max_chars)


def chunk_program(program: Program, config: ChunkerConfig) -> list[Chunk]:
    score_years = sorted({s.year for s in program.passing_scores})
    facts = format_extracted_facts(program)
    now = datetime.now(UTC)

    typed_chunks: list[Chunk] = []
    short_extras: list[str] = []

    if program.places_history:
        history_pairs = []
        for yp in sorted(program.places_history, key=lambda x: x.year, reverse=True)[:3]:
            budget = yp.places.budget.value if yp.places.budget else "—"
            history_pairs.append(f"{budget} бюджетных мест ({yp.year})")
        short_extras.append(f"История мест: {'; '.join(history_pairs)}")

    if program.tuition_history:
        history_pairs = []
        for yt in sorted(program.tuition_history, key=lambda x: x.year, reverse=True)[:3]:
            thousand = yt.tuition_fee_rub_per_year.value // 1000
            history_pairs.append(f"{thousand} тыс. ₽/год ({yt.year})")
        short_extras.append(f"История стоимости: {'; '.join(history_pairs)}")

    past_scores = [ps for ps in program.passing_scores if ps.year < program.admission_year]
    if past_scores:
        past_pairs = []
        for ps in sorted(past_scores, key=lambda s: s.year, reverse=True)[:3]:
            b = ps.budget.value if ps.budget else "—"
            p = ps.paid.value if ps.paid else "—"
            past_pairs.append(f"{ps.year}: бюджет {b}, платно {p}")
        short_extras.append(f"Проходные баллы: {'; '.join(past_pairs)}")

    for chunk_type, field in TYPE_TO_FIELD.items():
        text = (getattr(program, field) or "").strip()
        if not text:
            continue

        if (
            len(text.encode("utf-8")) < config.min_text_chars
            and config.short_section_mode == "merge_into_overview"
        ):
            short_extras.append(f"{chunk_type.value.capitalize()}: {text}")
            continue
        sections = split_text_by_soft_cap(text, max_chars_from_tokens(config.max_tokens_per_chunk))
        for index, section in enumerate(sections):
            section_index = index if len(sections) > 1 else None
            typed_chunks.append(
                make_chunk(
                    program,
                    chunk_type,
                    section,
                    config,
                    score_years,
                    facts,
                    now,
                    section_index=section_index,
                )
            )

    overview_text = build_overview_text(program, short_extras)
    overview_chunk = make_chunk(
        program, ChunkType.overview, overview_text, config, score_years, facts, now
    )
    return [overview_chunk, *typed_chunks]


def batch_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    batches: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        if current and current_len + para_len + 2 > max_chars:
            batches.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len + 2
    if current:
        batches.append("\n\n".join(current))
    return batches


def chunk_knowledge_doc(doc: KnowledgeDoc, config: ChunkerConfig) -> list[Chunk]:
    now = datetime.now(UTC)

    if isinstance(doc, FAQDoc) and doc.qa_pairs:
        return chunk_faq_qa_pairs(doc, config, now)

    paragraphs = [p.strip() for p in getattr(doc, "text", "").split("\n") if p.strip()]
    if not paragraphs:
        return []
    batches = batch_paragraphs(paragraphs, KNOWLEDGE_MAX_CHARS)
    chunks: list[Chunk] = []
    for i, batch_text in enumerate(batches):
        if len(batch_text.encode("utf-8")) < config.min_text_chars:
            continue
        chunks.append(make_knowledge_chunk(doc, f"chunk_{i:02d}", batch_text, config, now, i))
    return chunks


def chunk_faq_qa_pairs(doc: FAQDoc, config: ChunkerConfig, now: datetime) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, pair in enumerate(doc.qa_pairs):
        text = f"Вопрос: {pair.question}\nОтвет: {pair.answer}"
        if len(text.encode("utf-8")) < config.min_text_chars:
            continue
        chunk = make_knowledge_chunk(doc, f"qa_{i:03d}", text, config, now, i)
        chunk.extracted_facts.update(
            {
                "question": pair.question,
                "answer": pair.answer,
                "qa_index": i,
            }
        )
        chunks.append(chunk)
    return chunks


def make_knowledge_chunk(
    doc: KnowledgeDoc,
    suffix: str,
    text: str,
    config: ChunkerConfig,
    now: datetime,
    section_index: int | None = None,
) -> Chunk:
    chunk_id = f"{doc.doc_id}_{suffix}"
    header = f"Документ: «{doc.title}» | Тип: {doc.doc_type.value} | Год: {doc.admission_year}"
    embedding_input = f"{header}\n\n{text}" if config.context_header_mode == "simple" else text
    return Chunk(
        chunk_id=chunk_id,
        chunk_type=ChunkType.knowledge,
        text=text,
        embedding_input=embedding_input,
        program_slug=doc.doc_id,
        program_name=doc.title,
        program_code="",
        faculty=KNOWLEDGE_FACULTY,
        admission_year=doc.admission_year,
        score_years=[],
        extracted_facts={"doc_type": doc.doc_type.value},
        source_doc_type=doc.doc_type.value,
        profile_name=config.profile_name,
        section_index=section_index,
        url=doc.url,
        raw_html_hash=doc.raw_html_hash,
        created_at=now,
    )
