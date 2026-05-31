from __future__ import annotations

import re
from typing import Any

from parser.quoted_value import QuotedInt
from parser.schemas import (
    Exam,
    PassingScore,
    Places,
    TableKolmestRow,
    TableMinkritRow,
    TablePriceRow,
    TableResultsRow,
)

PROGRAM_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*\[([^\]]+)\]\(([^)\s]+)[^)]*\)\s*\|(.+)\|\s*$",
)


PROGRAM_ROW_NO_INDEX_RE = re.compile(
    r"^\|\s*\[([^\]]+)\]\(([^)\s]+)[^)]*\)[^|]*\|(.+)\|\s*$",
)

SECTION_HEADER_RE = re.compile(
    r"^\|\s*(?:Направление подготовки|Специальность)\s+(\d{2}\.\d{2}\.\d{2})\s+(.+?)\s*\|",
)


def slug_from_url(url: str) -> str:
    if "/ba/" in url:
        return url.split("/ba/", 1)[1].rstrip("/").split("/", 1)[0].split("?", 1)[0]
    host_match = re.match(r"^https?://([^/]+)", url)
    if host_match is None:
        return ""
    host = host_match.group(1)
    if not host.endswith(".hse.ru"):
        return ""
    subdomain = host.rsplit(".hse.ru", 1)[0]
    if subdomain in {"www", "ba"}:
        return ""
    return subdomain.split(".", 1)[0]


def normalize_url(url: str) -> str:
    return url.split("#", 1)[0].split("?", 1)[0].rstrip("/")


def parse_minkrit_table(markdown: str, year: int) -> list[TableMinkritRow]:
    rows: list[TableMinkritRow] = []
    current_okso: tuple[str, str] | None = None
    pending: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        rows.append(
            TableMinkritRow(
                year=year,
                program_url=pending["url"],
                program_slug=pending["slug"],
                program_label=pending["label"],
                okso_code=pending["okso"][0] if pending["okso"] else None,
                okso_label=pending["okso"][1] if pending["okso"] else None,
                row_quote=pending["row_quote"],
                exams=pending["exams"],
            )
        )
        pending = None

    for line in markdown.split("\n"):
        sec_match = SECTION_HEADER_RE.match(line)
        if sec_match:
            flush()
            current_okso = (sec_match.group(1), sec_match.group(2).strip())
            continue

        prog_match = PROGRAM_ROW_RE.match(line)
        if prog_match:
            flush()
            label = prog_match.group(1).strip()
            url = normalize_url(prog_match.group(2).strip())
            slug = slug_from_url(url)
            if not slug:
                continue
            rest_cells = [c.strip() for c in prog_match.group(3).split("|") if c.strip()]
            first_exam = parse_minkrit_exam_cells(rest_cells[-3:]) if len(rest_cells) >= 3 else None
            pending = {
                "url": url,
                "slug": slug,
                "label": label,
                "okso": current_okso,
                "row_quote": line.strip(),
                "exams": [first_exam] if first_exam else [],
            }
            continue

        if pending is not None and line.startswith("|"):
            cells = [c.strip() for c in line.strip().split("|") if c.strip()]
            if len(cells) >= 3:
                exam = parse_minkrit_exam_cells(cells[-3:])
                if exam is not None:
                    pending["exams"].append(exam)

    flush()
    return rows


def parse_minkrit_exam_cells(cells: list[str]) -> Exam | None:
    if len(cells) < 3:
        return None
    subject, _priority, score_str = cells[0], cells[1], cells[2]
    try:
        score = int(score_str.strip())
    except ValueError:
        return None
    return Exam(subject=subject, min_score=QuotedInt(value=score, quote=score_str))


def parse_kolmest_table(markdown: str, year: int) -> list[TableKolmestRow]:
    rows: list[TableKolmestRow] = []
    current_okso: tuple[str, str] | None = None

    for line in markdown.split("\n"):
        sec_match = SECTION_HEADER_RE.match(line)
        if sec_match:
            current_okso = (sec_match.group(1), sec_match.group(2).strip())
            continue

        prog_match = PROGRAM_ROW_NO_INDEX_RE.match(line)
        if not prog_match:
            continue

        label = prog_match.group(1).strip()
        url = normalize_url(prog_match.group(2).strip())
        slug = slug_from_url(url)
        if not slug:
            continue
        cells = [c.strip() for c in prog_match.group(3).split("|")]
        places = parse_kolmest_cells(cells)
        rows.append(
            TableKolmestRow(
                year=year,
                program_url=url,
                program_slug=slug,
                program_label=label,
                okso_code=current_okso[0] if current_okso else None,
                okso_label=current_okso[1] if current_okso else None,
                row_quote=line.strip(),
                places=places,
            )
        )
    return rows


def parse_kolmest_cells(cells: list[str]) -> Places:
    def cell_to_qi(cell: str) -> QuotedInt | None:
        cell = cell.strip()
        if not cell or cell == "-":
            return None
        m = re.search(r"\d+", cell)
        if not m:
            return None
        return QuotedInt(value=int(m.group()), quote=cell[:200])

    numeric_start = 1 if cells and not re.search(r"\d", cells[0]) else 0

    def at(idx: int) -> QuotedInt | None:
        actual = numeric_start + idx
        if actual >= len(cells):
            return None
        return cell_to_qi(cells[actual])

    return Places(
        budget=at(0),
        special=at(1),
        target=at(2),
        separate=at(3),
        paid=at(4),
    )


def parse_price_table(markdown: str, year: int) -> list[TablePriceRow]:
    rows: list[TablePriceRow] = []
    current_okso: tuple[str, str] | None = None

    for line in markdown.split("\n"):
        sec_match = SECTION_HEADER_RE.match(line)
        if sec_match:
            current_okso = (sec_match.group(1), sec_match.group(2).strip())
            continue

        prog_match = PROGRAM_ROW_NO_INDEX_RE.match(line)
        if not prog_match:
            continue

        label = prog_match.group(1).strip()
        url = normalize_url(prog_match.group(2).strip())
        slug = slug_from_url(url)
        if not slug:
            continue
        cells = [c.strip() for c in prog_match.group(3).split("|") if c.strip()]
        digits = re.search(r"\d+", cells[-1] if cells else "")
        if not digits:
            continue

        tuition = QuotedInt(value=int(digits.group()) * 1000, quote=cells[-1])
        rows.append(
            TablePriceRow(
                year=year,
                program_url=url,
                program_slug=slug,
                program_label=label,
                okso_code=current_okso[0] if current_okso else None,
                okso_label=current_okso[1] if current_okso else None,
                row_quote=line.strip(),
                tuition_fee_rub_per_year=tuition,
            )
        )
    return rows


def parse_result_table(markdown: str, year: int) -> list[TableResultsRow]:
    rows: list[TableResultsRow] = []
    current_okso: tuple[str, str] | None = None

    for line in markdown.split("\n"):
        sec_match = SECTION_HEADER_RE.match(line)
        if sec_match:
            current_okso = (sec_match.group(1), sec_match.group(2).strip())
            continue

        prog_match = PROGRAM_ROW_NO_INDEX_RE.match(line)
        if not prog_match:
            continue

        label = prog_match.group(1).strip()
        url = normalize_url(prog_match.group(2).strip())
        slug = slug_from_url(url)
        if not slug:
            continue
        cells = [c.strip() for c in prog_match.group(3).split("|") if c.strip()]
        score = parse_result_cells(cells, year)
        rows.append(
            TableResultsRow(
                program_url=url,
                program_slug=slug,
                program_label=label,
                okso_code=current_okso[0] if current_okso else None,
                okso_label=current_okso[1] if current_okso else None,
                row_quote=line.strip(),
                passing_scores=[score] if score else [],
            )
        )
    return rows


def parse_result_cells(cells: list[str], year: int) -> PassingScore | None:
    def parse_score(cell: str) -> QuotedInt | None:
        cell = cell.strip()

        if not cell or cell.startswith("-"):
            return None
        m = re.search(r"\d+", cell)
        if not m:
            return None
        return QuotedInt(value=int(m.group()), quote=cell)

    budget = parse_score(cells[0]) if len(cells) > 0 else None
    paid = parse_score(cells[1]) if len(cells) > 1 else None
    return PassingScore(year=year, budget=budget, paid=paid)
