import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from qdrant_client import QdrantClient

from rag.agent.context import AgentContext
from rag.agent.qdrant_lookup import (
    get_program_facts,
    get_program_section_text,
    list_unique_programs,
)
from rag.agent.schemas import (
    AdmissionDeadline,
    AggregateOperation,
    AnalysisBucket,
    AnalysisField,
    AnalysisResult,
    ComparisonTable,
    GroupByField,
    MatchHit,
    MatchResult,
    ProgramDetails,
    ProgramFilter,
    ProgramHit,
    ProgramSummary,
    YearScore,
)

Corpus = Literal["programs", "knowledge"]


TOOLS_YAML = Path(__file__).parent / "prompts" / "tools.yaml"
TOOL_DESCRIPTIONS: dict[str, str] = {
    name: str(spec["description"]).strip()
    for name, spec in (yaml.safe_load(TOOLS_YAML.read_text(encoding="utf-8")) or {}).items()
}


def tool_description(name: str) -> str:
    return TOOL_DESCRIPTIONS[name]


PROGRAM_CHUNK_TYPES: frozenset[str] = frozenset(
    {"overview", "curriculum", "advantages", "career", "admission"}
)
KNOWLEDGE_CHUNK_TYPES: frozenset[str] = frozenset({"knowledge"})

MATCH_DISCLAIMER = (
    "Это сравнение с последним известным проходным баллом, а не прогноз поступления. "
    "В новом наборе проходной балл может быть другим."
)


NON_EGE_EXAM_MARKERS = ("творческое", "профессиональный", "дви")


def payload_to_summary(payload: dict[str, Any]) -> ProgramSummary:
    facts = payload.get("extracted_facts", {}) or {}
    codes = facts.get("codes", []) or []
    code = codes[0] if codes else ""
    passing_scores = facts.get("passing_scores", []) or []
    latest_passing = max(passing_scores, key=lambda s: s["year"]) if passing_scores else None
    places = facts.get("places", {}) or {}
    exams_raw = facts.get("exams", []) or []
    return ProgramSummary(
        program_slug=payload["program_slug"],
        program_name=payload["program_name"],
        faculty=payload.get("faculty", ""),
        code=code,
        passing_score_budget=(latest_passing or {}).get("budget"),
        tuition_rub=facts.get("tuition_fee_rub_per_year"),
        budget_places=places.get("budget"),
        exams=[e["subject"] for e in exams_raw],
        source_url=str(payload.get("url", "")),
    )


def search_programs(
    ctx: AgentContext,
    query: str,
    top_k: int = 5,
    corpus: Corpus = "programs",
) -> list[ProgramHit]:
    allowed = PROGRAM_CHUNK_TYPES if corpus == "programs" else KNOWLEDGE_CHUNK_TYPES

    raw_hits = ctx.retrieval.search(query, top_k=top_k * 3)
    out: list[ProgramHit] = []
    for h in raw_hits:
        c = h.chunk
        if c.chunk_type not in allowed:
            continue
        out.append(
            ProgramHit(
                program_slug=c.program_slug,
                program_name=c.program_name,
                faculty=c.faculty,
                chunk_type=c.chunk_type,
                text_preview=c.text[:500],
                source_url=str(c.url),
                score=h.score,
            )
        )
    return out[:top_k]


def filter_programs(ctx: AgentContext, filter_: ProgramFilter) -> list[ProgramSummary]:
    result: list[ProgramSummary] = []
    for payload in list_unique_programs(ctx.qdrant_client, ctx.qdrant_collection):
        faculty = payload.get("faculty", "")
        if (
            filter_.faculty_query is not None
            and filter_.faculty_query.lower() not in faculty.lower()
        ):
            continue
        summary = payload_to_summary(payload)
        if filter_.min_passing_score_budget is not None and (
            summary.passing_score_budget is None
            or summary.passing_score_budget < filter_.min_passing_score_budget
        ):
            continue
        if filter_.max_passing_score_budget is not None and (
            summary.passing_score_budget is None
            or summary.passing_score_budget > filter_.max_passing_score_budget
        ):
            continue
        if filter_.max_tuition_rub is not None and (
            summary.tuition_rub is None or summary.tuition_rub > filter_.max_tuition_rub
        ):
            continue
        if filter_.has_subject is not None:
            target = filter_.has_subject.lower()
            if not any(target in s.lower() for s in summary.exams):
                continue
        result.append(summary)
    return result


def get_program_details(ctx: AgentContext, slug: str, sections: list[str]) -> ProgramDetails | None:
    payload = get_program_facts(ctx.qdrant_client, ctx.qdrant_collection, slug)
    if payload is None:
        return None
    facts = payload.get("extracted_facts", {}) or {}
    summary = payload_to_summary(payload)
    section_map: dict[str, str] = {}
    for section in sections:
        if section not in PROGRAM_CHUNK_TYPES:
            continue
        text = get_program_section_text(ctx.qdrant_client, ctx.qdrant_collection, slug, section)
        if text:
            section_map[section] = text
    return ProgramDetails(
        program=summary,
        sections=section_map,
        extracted_facts={
            "passing_scores": facts.get("passing_scores", []),
            "min_scores_by_subject": facts.get("min_scores_by_subject", {}),
            "places": facts.get("places", {}),
            "tuition_fee_rub_per_year": facts.get("tuition_fee_rub_per_year"),
            "places_history": facts.get("places_history", []),
            "tuition_history": facts.get("tuition_history", []),
        },
        source_url=str(payload.get("url", "")),
    )


def compare_programs(ctx: AgentContext, slugs: list[str], aspects: list[str]) -> ComparisonTable:
    summaries: list[ProgramSummary] = []
    for slug in slugs:
        payload = get_program_facts(ctx.qdrant_client, ctx.qdrant_collection, slug)
        if payload is None:
            continue
        summaries.append(payload_to_summary(payload))

    comparison: dict[str, dict[str, str]] = {}
    for aspect in aspects:
        per_slug: dict[str, str] = {}
        for s in summaries:
            value = ""
            if aspect == "exams":
                value = ", ".join(s.exams)
            elif aspect == "passing_score":
                value = str(s.passing_score_budget) if s.passing_score_budget else "—"
            elif aspect == "tuition":
                value = f"{s.tuition_rub} ₽/год" if s.tuition_rub else "—"
            elif aspect == "places":
                value = str(s.budget_places) if s.budget_places else "—"
            elif aspect == "career":
                career_text = get_program_section_text(
                    ctx.qdrant_client, ctx.qdrant_collection, s.program_slug, "career"
                )
                value = career_text or "—"
            per_slug[s.program_slug] = value
        comparison[aspect] = per_slug
    return ComparisonTable(programs=summaries, comparison=comparison)


RU_STOPWORDS = {"и", "в", "на", "из", "о", "по", "к", "с", "у", "от"}


def load_universal_acronyms(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    universal = raw.get("universal", {})
    if not isinstance(universal, dict):
        raise ValueError("acronyms config must contain a universal map")
    return {str(key): str(value) for key, value in universal.items() if str(value).strip()}


def derive_acronyms_from_qdrant(
    qdrant_client: QdrantClient,
    collection: str,
    acronyms_config_path: Path,
) -> dict[str, str]:
    acronyms = load_universal_acronyms(acronyms_config_path)
    payloads = list_unique_programs(qdrant_client, collection)
    sorted_payloads = sorted(
        payloads,
        key=lambda payload: (payload.get("program_name", ""), payload.get("program_slug", "")),
    )
    for payload in sorted_payloads:
        name = payload.get("program_name", "")
        if not name:
            continue
        words = [
            w
            for w in re.findall(r"[А-Яа-я]+", name)
            if len(w) >= 3 and w.lower() not in RU_STOPWORDS
        ]
        if 2 <= len(words) <= 5:
            acronym = "".join(w[0].upper() for w in words)
            if 2 <= len(acronym) <= 5 and acronym not in acronyms:
                acronyms[acronym] = name
    return acronyms


def load_program_acronyms_from_disk(
    programs_dir: Path,
    acronyms_config_path: Path,
) -> dict[str, str]:
    acronyms = load_universal_acronyms(acronyms_config_path)
    names: list[str] = []
    if not programs_dir.exists():
        return acronyms
    for jf in sorted(programs_dir.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        name = str(data.get("name") or "").strip()
        if name:
            names.append(name)
    for name in sorted(names):
        words = [
            w
            for w in re.findall(r"[А-Яа-я]+", name)
            if len(w) >= 3 and w.lower() not in RU_STOPWORDS
        ]
        if 2 <= len(words) <= 5:
            acronym = "".join(w[0].upper() for w in words)
            if 2 <= len(acronym) <= 5 and acronym not in acronyms:
                acronyms[acronym] = name
    return acronyms


def get_admission_deadlines(
    ctx: AgentContext,
    has_hse_exam: bool,
) -> list[AdmissionDeadline]:
    target_category = "with_hse_exams" if has_hse_exam else "ege_only"
    out: list[AdmissionDeadline] = []
    for e in ctx.deadlines:
        if e.category.value != target_category:
            continue
        out.append(
            AdmissionDeadline(
                financing=e.financing.value,
                category=e.category.value,
                start_date=e.start_date,
                end_date=e.end_date,
                description=e.description,
                source_url=e.source_url,
            )
        )

    out.sort(key=lambda x: 0 if x.financing == "budget" else 1)
    return out


def matches_subject(user_subject: str, exam_alternative: str) -> bool:
    u = user_subject.lower().strip()
    e = exam_alternative.lower().strip()
    return bool(u) and bool(e) and (u in e or e in u)


def resolve_user_subject(exam_subject: str, user_scores: dict[str, int]) -> tuple[str, int] | None:
    alternatives = [a.strip() for a in exam_subject.split("/") if a.strip()]
    for alt in alternatives:
        for user_subject, score in user_scores.items():
            if matches_subject(user_subject, alt):
                return alt, score
    return None


def match_by_scores(
    ctx: AgentContext,
    user_scores: dict[str, int],
) -> MatchResult:
    payloads = list_unique_programs(ctx.qdrant_client, ctx.qdrant_collection)

    latest_year: int | None = None
    for payload in payloads:
        facts = payload.get("extracted_facts", {}) or {}
        for s in facts.get("passing_scores", []) or []:
            if s.get("budget") is not None and (latest_year is None or s["year"] > latest_year):
                latest_year = s["year"]
    result = MatchResult(based_on_year=latest_year, disclaimer=MATCH_DISCLAIMER)
    for payload in payloads:
        hit = evaluate_program_payload(payload, user_scores)
        result.results.append(hit)

    order = {"ok": 0, "no_data": 1, "incompatible": 2}
    result.results.sort(
        key=lambda h: (
            order[h.status],
            h.passing_score_budget if h.passing_score_budget is not None else 10_000,
            h.program_name,
        )
    )
    return result


def payload_field_value(payload: dict[str, Any], field: AnalysisField) -> int | None:
    facts = payload.get("extracted_facts", {}) or {}
    if field == "tuition":
        return facts.get("tuition_fee_rub_per_year")
    if field == "passing_score_budget":
        scores = sorted(
            facts.get("passing_scores", []) or [],
            key=lambda s: s["year"],
            reverse=True,
        )
        for s in scores:
            if s.get("budget") is not None:
                return int(s["budget"])
        return None
    places = facts.get("places", {}) or {}
    if field == "budget_places":
        return places.get("budget")
    if field == "paid_places":
        return places.get("paid")
    return None


def aggregate_values(values: list[int], operation: AggregateOperation) -> float | None:
    if operation == "count":
        return float(len(values))
    if not values:
        return None
    if operation == "sum":
        return float(sum(values))
    if operation == "mean":
        return sum(values) / len(values)
    if operation == "min":
        return float(min(values))
    if operation == "max":
        return float(max(values))
    if operation == "median":
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        return (
            (sorted_vals[mid - 1] + sorted_vals[mid]) / 2 if n % 2 == 0 else float(sorted_vals[mid])
        )
    return None


def analyze_programs(
    ctx: AgentContext,
    operation: AggregateOperation,
    field: AnalysisField,
    filter_: ProgramFilter | None = None,
    group_by: GroupByField = "none",
) -> AnalysisResult:
    payloads = list_unique_programs(ctx.qdrant_client, ctx.qdrant_collection)
    if filter_ is not None:
        included_summaries = filter_programs(ctx, filter_)
        keep = {s.program_slug for s in included_summaries}
        payloads = [p for p in payloads if p.get("program_slug") in keep]

    n_total = len(payloads)
    n_missing = 0
    grouped: dict[str, list[tuple[str, int]]] = {}
    slug_to_name = {p.get("program_slug", ""): p.get("program_name", "") for p in payloads}
    for payload in payloads:
        slug = payload.get("program_slug", "")
        value = payload_field_value(payload, field)
        if value is None:
            n_missing += 1
            if operation == "count":
                key = payload.get("faculty", "all") if group_by == "faculty" else "all"
                grouped.setdefault(key, []).append((slug, 0))
            continue
        key = payload.get("faculty", "all") if group_by == "faculty" else "all"
        grouped.setdefault(key, []).append((slug, value))

    buckets: list[AnalysisBucket] = []
    for key, items in grouped.items():
        values = [v for _, v in items]
        result_value = aggregate_values(values, operation)
        sorted_items = sorted(items, key=lambda iv: iv[1], reverse=True)[:5]
        top_examples = [f"{slug_to_name.get(slug, slug)} ({slug})" for slug, _ in sorted_items]
        buckets.append(
            AnalysisBucket(
                key=key,
                value=result_value,
                n_programs=len(items),
                top_examples=top_examples,
            )
        )

    buckets.sort(key=lambda b: b.value if b.value is not None else -1.0, reverse=True)
    notes = (
        f"{operation}({field}) по {n_total} программам"
        + (f", группировка по {group_by}" if group_by != "none" else "")
        + (f"; пропущено {n_missing} (нет данных)" if n_missing else "")
    )
    return AnalysisResult(
        operation=operation,
        field=field,
        group_by=group_by,
        n_programs_total=n_total,
        n_programs_missing_field=n_missing,
        buckets=buckets,
        notes=notes,
    )


def evaluate_program_payload(payload: dict[str, Any], user_scores: dict[str, int]) -> MatchHit:
    facts = payload.get("extracted_facts", {}) or {}
    exams = facts.get("exams", []) or []
    passing_scores = facts.get("passing_scores", []) or []
    program_slug = payload.get("program_slug", "")
    program_name = payload.get("program_name", "")
    faculty = payload.get("faculty", "")
    source_url = str(payload.get("url", ""))

    notes: list[str] = []
    has_non_ege = any(
        any(marker in str(e.get("subject", "")).lower() for marker in NON_EGE_EXAM_MARKERS)
        for e in exams
    )
    if has_non_ege:
        notes.append("программа включает ДВИ/творческое/профессиональное испытание")

    user_sum = 0
    reason: str | None = None
    for exam in exams:
        subject = str(exam.get("subject", ""))
        min_score = int(exam.get("min_score", 0))
        if any(marker in subject.lower() for marker in NON_EGE_EXAM_MARKERS):
            continue
        resolved = resolve_user_subject(subject, user_scores)
        if resolved is None:
            reason = f"нет ЕГЭ по предмету: {subject}"
            break
        _alt, score = resolved
        if score < min_score:
            reason = f"балл по «{subject}» ({score}) ниже минимального ({min_score})"
            break
        user_sum += score

    history = [
        YearScore(year=s["year"], budget=s.get("budget"), paid=s.get("paid"))
        for s in sorted(passing_scores, key=lambda x: x["year"])
    ]
    latest_with_budget = next(
        ((s.year, s.budget) for s in reversed(history) if s.budget is not None),
        (None, None),
    )
    passing_year, passing_latest = latest_with_budget

    if reason is not None:
        return MatchHit(
            program_slug=program_slug,
            program_name=program_name,
            faculty=faculty,
            source_url=source_url,
            notes=notes,
            status="incompatible",
            user_sum=None,
            passing_score_budget=None,
            passing_year=None,
            passing_score_history=history,
            reason=reason,
        )

    if has_non_ege or passing_latest is None:
        no_data_reason = (
            "программа требует ДВИ/творческого испытания — оценка по ЕГЭ-сумме некорректна"
            if has_non_ege
            else "нет проходного балла по бюджету за последние годы"
        )
        return MatchHit(
            program_slug=program_slug,
            program_name=program_name,
            faculty=faculty,
            source_url=source_url,
            notes=notes,
            status="no_data",
            user_sum=user_sum,
            passing_score_budget=passing_latest,
            passing_year=passing_year,
            passing_score_history=history,
            reason=no_data_reason,
        )

    return MatchHit(
        program_slug=program_slug,
        program_name=program_name,
        faculty=faculty,
        source_url=source_url,
        notes=notes,
        status="ok",
        user_sum=user_sum,
        passing_score_budget=passing_latest,
        passing_year=passing_year,
        passing_score_history=history,
        reason=None,
    )
