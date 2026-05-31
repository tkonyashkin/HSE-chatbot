from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar
from urllib.parse import urlsplit

from parser.schemas import (
    Exam,
    PassingScore,
    Program,
    TableKolmestRow,
    TableMinkritRow,
    TablePriceRow,
    TableProgramIdentity,
    TableResultsRow,
    YearlyPlaces,
    YearlyTuition,
)

RowT = TypeVar("RowT", bound=TableProgramIdentity)


def join_program(
    program_draft: Program,
    *,
    kolmest_rows: Sequence[TableKolmestRow] | None = None,
    minkrit_rows: Sequence[TableMinkritRow] | None = None,
    price_rows: Sequence[TablePriceRow] | None = None,
    result_rows: Sequence[TableResultsRow] | None = None,
) -> Program:
    matching_kolmest = matching_rows(kolmest_rows, program_draft)
    matching_minkrit = matching_rows(minkrit_rows, program_draft)
    matching_prices = matching_rows(price_rows, program_draft)
    matching_results = matching_rows(result_rows, program_draft)

    identity_rows: list[TableProgramIdentity] = [
        *matching_kolmest,
        *matching_minkrit,
        *matching_prices,
        *matching_results,
    ]
    codes = merge_codes(program_draft.codes, identity_rows)

    current_year = program_draft.admission_year

    current_kolmest = next((row for row in matching_kolmest if row.year == current_year), None)
    places = current_kolmest.places if current_kolmest else program_draft.places
    places_history = [
        YearlyPlaces(year=row.year, places=row.places)
        for row in sorted(matching_kolmest, key=lambda r: r.year, reverse=True)
        if row.year != current_year
    ]

    current_price = next((row for row in matching_prices if row.year == current_year), None)
    tuition = (
        current_price.tuition_fee_rub_per_year
        if current_price
        else program_draft.tuition_fee_rub_per_year
    )
    tuition_history = [
        YearlyTuition(
            year=row.year,
            tuition_fee_rub_per_year=row.tuition_fee_rub_per_year,
        )
        for row in sorted(matching_prices, key=lambda r: r.year, reverse=True)
        if row.year != current_year
    ]

    merged_exams = merge_exams(program_draft.exams, matching_minkrit)
    passing_scores = merge_passing_scores(program_draft.passing_scores, matching_results)

    return program_draft.model_copy(
        update={
            "codes": codes,
            "places": places,
            "places_history": places_history,
            "exams": merged_exams,
            "tuition_fee_rub_per_year": tuition,
            "tuition_history": tuition_history,
            "passing_scores": passing_scores,
        }
    )


def matching_rows(rows: Sequence[RowT] | None, program: Program) -> list[RowT]:
    return [row for row in rows or [] if row_matches_program(row, program)]


def row_matches_program(row: TableProgramIdentity, program: Program) -> bool:
    program_url = canonical_url(str(program.url))
    row_url = canonical_url(row.program_url)
    if row_url and row_url == program_url:
        return True

    program_slug = normalize_slug(program.slug)
    row_slug = normalize_slug(row.program_slug)
    row_url_slug = slug_from_url(row.program_url)
    return program_slug in {row_slug, row_url_slug}


def canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.rstrip("/")
        return parsed._replace(path=path, query="", fragment="").geturl()
    return value.split("#", 1)[0].split("?", 1)[0].rstrip("/")


def slug_from_url(value: str) -> str:
    path = urlsplit(value).path or value
    slug = path.rstrip("/").rsplit("/", 1)[-1]
    return normalize_slug(slug)


def normalize_slug(value: str) -> str:
    return value.strip().strip("/")


def merge_codes(
    draft_codes: Sequence[str],
    rows: Sequence[TableProgramIdentity],
) -> list[str]:
    codes = list(draft_codes)
    for row in rows:
        if row.okso_code and row.okso_code not in codes:
            codes.append(row.okso_code)
    return codes


def merge_exams(
    draft_exams: Sequence[Exam],
    rows: Sequence[TableMinkritRow],
) -> list[Exam]:
    exams = list(draft_exams)
    seen = {value_key(exam) for exam in exams}
    for row in rows:
        for exam in row.exams:
            key = value_key(exam)
            if key not in seen:
                seen.add(key)
                exams.append(exam)
    return exams


def merge_passing_scores(
    draft_scores: Sequence[PassingScore],
    rows: Sequence[TableResultsRow],
) -> list[PassingScore]:
    scores = list(draft_scores)
    seen = {value_key(score) for score in scores}
    for row in rows:
        for score in row.passing_scores:
            key = value_key(score)
            if key not in seen:
                seen.add(key)
                scores.append(score)
    return sorted(scores, key=lambda score: score.year)


def value_key(value: object) -> str:
    if hasattr(value, "model_dump_json"):
        return str(value.model_dump_json())
    return repr(value)
