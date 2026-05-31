from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from parser.cache import Cache
from parser.config import ParserConfig, aux_slugs_for, load_parser_config
from parser.crawler import CatalogCrawler, ProgramListing
from parser.extractor import QuoteNotInSourceError, extract
from parser.fetcher import Fetcher, html_to_markdown
from parser.joiner import join_program
from parser.knowledge_schemas import (
    AchievementsDoc,
    ExamProgramsDoc,
    FAQDoc,
    InformationDoc,
    OlympiadsDoc,
)
from parser.llm import LLMClient
from parser.schemas import ExtractionMetadata, Program, ProgramDraft

load_dotenv()

app = typer.Typer(help="Парсер программ бакалавриата НИУ ВШЭ (Москва), LLM-first, year-stamped.")


def resolve_year(year: int | None) -> int:
    return year if year is not None else int(os.environ.get("HSE_ADMISSION_YEAR", "2026"))


def program_base_url(pattern: str) -> str:
    return pattern.split("{slug}", 1)[0]


def json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def write_failed_program(
    *,
    slug: str,
    source_url: str,
    raw_markdown: str,
    exc: Exception,
    failed_dir: Path,
    model_id: str,
) -> Path:
    payload = {
        "slug": slug,
        "source_url": source_url,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "model_id": model_id,
        "raw_markdown": raw_markdown,
    }
    path = failed_dir / f"{slug}.json"
    write_json(path, payload)
    return path


def load_failed_entries(failed_dir: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in sorted(failed_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_url = str(payload.get("source_url", ""))
        if source_url:
            entries.append({"slug": path.stem, "url": source_url})
    return entries


def empty_tables_payload() -> dict[str, Any]:
    return {
        "kolmest_rows": [],
        "minkrit_rows": [],
        "price_rows": [],
        "result_rows": [],
    }


def _extract_tables(
    config: ParserConfig,
    fetcher: Fetcher,
    cache: Cache,
    year: int,
    model_id: str,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    from parser.table_parser import (
        parse_kolmest_table,
        parse_minkrit_table,
        parse_price_table,
        parse_result_table,
    )

    tables: dict[str, Any] = empty_tables_payload()
    table_specs = [
        ("minkrit_rows", config.tables.minkrit, parse_minkrit_table),
        ("kolmest_rows", config.tables.kolmest, parse_kolmest_table),
        ("price_rows", config.tables.price, parse_price_table),
        ("result_rows", config.tables.results, parse_result_table),
    ]
    for output_key, year_url_map, parse_fn in table_specs:
        for year_str, url in sorted(year_url_map.items()):
            year_int = int(year_str)
            try:
                fetched = fetcher.fetch(str(url), year=year_int)
                markdown = html_to_markdown(fetched.html)
                rows = parse_fn(markdown, year=year_int)
                tables[output_key].extend(rows)
            except Exception as exc:
                typer.echo(f"{output_key} {year_str}: ERROR {type(exc).__name__}: {exc}", err=True)
    return tables


def raw_fetcher(config: ParserConfig) -> Fetcher:
    return Fetcher(cache_dir=config.data_root / "raw")


def parse_slugs_option(value: str | None) -> list[str] | None:
    if value is None:
        return None
    slugs = [slug.strip() for slug in value.split(",") if slug.strip()]
    return slugs or None


def url_from_pattern(pattern: str, slug: str) -> str:
    return pattern.format(slug=slug)


def content_hash_from_page(page: Any) -> str:
    content_hash = getattr(page, "content_hash", None)
    if isinstance(content_hash, str) and content_hash:
        return content_hash
    html = str(getattr(page, "html", ""))
    return "sha256:" + hashlib.sha256(html.encode("utf-8")).hexdigest()


def parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def preserved_provenance(
    output_path: Path,
    *,
    raw_html_hash: str,
    model_id: str,
) -> tuple[datetime, datetime] | None:
    payload = read_json_object(output_path)
    if payload is None or payload.get("raw_html_hash") != raw_html_hash:
        return None
    generation_metadata = payload.get("generation_metadata")
    if not isinstance(generation_metadata, dict):
        return None
    if generation_metadata.get("model_name") != model_id:
        return None
    retrieved_at = parse_datetime(payload.get("retrieved_at"))
    extraction_date = parse_datetime(generation_metadata.get("extraction_date"))
    if retrieved_at is None or extraction_date is None:
        return None
    return retrieved_at, extraction_date


def extraction_metadata(
    model_id: str,
    *,
    extraction_date: datetime | None = None,
) -> ExtractionMetadata:
    return ExtractionMetadata(
        model_name=model_id,
        model_version_alias=model_id,
        extraction_date=extraction_date or datetime.now(UTC),
    )


def enrich_program(
    draft: ProgramDraft,
    *,
    slug: str,
    url: str,
    year: int,
    page: Any,
    model_id: str,
    output_path: Path,
) -> Program:
    payload = draft.model_dump(mode="json")
    raw_html_hash = content_hash_from_page(page)
    preserved = preserved_provenance(
        output_path,
        raw_html_hash=raw_html_hash,
        model_id=model_id,
    )
    retrieved_at = preserved[0] if preserved is not None else datetime.now(UTC)
    extraction_date = preserved[1] if preserved is not None else None
    payload.update(
        {
            "slug": slug,
            "url": url,
            "admission_year": year,
            "retrieved_at": retrieved_at,
            "raw_html_hash": raw_html_hash,
            "generation_metadata": extraction_metadata(
                model_id,
                extraction_date=extraction_date,
            ).model_dump(mode="json"),
        }
    )
    return Program.model_validate(payload)


def write_crawl_meta(
    *,
    config: ParserConfig,
    year: int,
    listings: list[ProgramListing],
) -> Path:
    raw_dir = config.raw_dir(year)
    current_slugs = [item.slug for item in listings]
    payload = {
        "admission_year": year,
        "catalog_url": str(config.catalog_url),
        "discovered_at": datetime.now(UTC).isoformat(),
        "total": len(current_slugs),
        "slugs": current_slugs,
        "programs": [{"slug": item.slug, "name": item.name, "url": item.url} for item in listings],
    }
    path = raw_dir / "_meta.json"
    write_json(path, payload)
    return path


@app.command()
def crawl(
    config_path: Path = typer.Option(Path("config/parser.yaml"), help="Путь к parser.yaml"),
    year: int | None = typer.Option(None, "--year"),
    output: Path = typer.Option(Path("data/programs_listing.json"), help="Куда записать JSON"),
) -> None:
    config = load_parser_config(config_path)
    resolved_year = resolve_year(year)
    fetcher = raw_fetcher(config)
    crawler = CatalogCrawler(
        aux_slug_blacklist=aux_slugs_for(config),
        url_classifier=config.url_classifier,
    )
    fetched = fetcher.fetch(str(config.catalog_url), year=resolved_year)
    listings = crawler.parse_catalog(
        fetched.html, base_url=program_base_url(config.program_url_pattern)
    )
    meta_path = write_crawl_meta(
        config=config,
        year=resolved_year,
        listings=listings,
    )
    write_json(
        output, [{"slug": item.slug, "name": item.name, "url": item.url} for item in listings]
    )
    typer.echo(f"Найдено программ: {len(listings)}; записано в {output}; meta: {meta_path}")


@app.command("parse-programs")
def parse_programs(
    config_path: Path = typer.Option(Path("config/parser.yaml"), help="Путь к parser.yaml"),
    year: int | None = typer.Option(None, "--year"),
    rerun_failed: bool = typer.Option(False, "--rerun-failed"),
    limit: int | None = typer.Option(None, "--limit"),
    slugs: str | None = typer.Option(None, "--slugs", help="Comma-separated program slugs"),
    model_id: str = typer.Option("anthropic/claude-sonnet-4.6", "--model-id", "--model"),
    skip_tables: bool = typer.Option(False, "--skip-tables"),
) -> None:
    config = load_parser_config(config_path)
    resolved_year = resolve_year(year)
    programs_dir = config.programs_dir(resolved_year)
    failed_dir = config.failed_dir(resolved_year)
    fetcher = raw_fetcher(config)
    cache = Cache(cache_dir=config.cache_dir(resolved_year))
    crawler = CatalogCrawler(
        aux_slug_blacklist=aux_slugs_for(config),
        url_classifier=config.url_classifier,
    )
    llm: LLMClient | None = None
    selected_slugs = parse_slugs_option(slugs)

    if rerun_failed:
        if selected_slugs is not None:
            raise typer.BadParameter("--slugs cannot be combined with --rerun-failed")
        failed_dir.mkdir(parents=True, exist_ok=True)
        entries = load_failed_entries(failed_dir)
        typer.echo(f"Повторный прогон failed programs: {len(entries)} из {failed_dir}")
    else:
        catalog = fetcher.fetch(str(config.catalog_url), year=resolved_year)
        listings = crawler.parse_catalog(
            catalog.html, base_url=program_base_url(config.program_url_pattern)
        )
        listing_by_slug = {item.slug: item for item in listings}
        if selected_slugs is not None:
            selected = [
                listing_by_slug.get(slug)
                or ProgramListing(
                    slug=slug,
                    name=slug,
                    url=url_from_pattern(config.program_url_pattern, slug),
                )
                for slug in selected_slugs
            ]
        else:
            selected = listings[:limit] if limit else listings
        entries = [{"slug": item.slug, "url": item.url} for item in selected]

    programs_dir.mkdir(parents=True, exist_ok=True)
    tables_payload = (
        empty_tables_payload()
        if skip_tables
        else _extract_tables(config, fetcher, cache, resolved_year, model_id, llm)
    )
    success_count = 0
    for entry in entries:
        slug = entry["slug"]
        url = entry["url"]
        raw_markdown = ""
        try:
            page = fetcher.fetch(url, year=resolved_year)
            raw_markdown = html_to_markdown(page.html)
            draft = extract(
                raw_markdown,
                schema_type=ProgramDraft,
                prompt_name="program",
                llm=llm,
                cache=cache,
                model_id=model_id,
            )
            draft = enrich_program(
                draft,
                slug=slug,
                url=url,
                year=resolved_year,
                page=page,
                model_id=model_id,
                output_path=programs_dir / f"{slug}.json",
            )
            joined = join_program(draft, **tables_payload)
            (programs_dir / f"{slug}.json").write_text(
                joined.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (failed_dir / f"{slug}.json").unlink(missing_ok=True)
            typer.echo(f"{slug}: OK")
            success_count += 1
        except (ValidationError, QuoteNotInSourceError) as exc:
            written = write_failed_program(
                slug=slug,
                source_url=url,
                raw_markdown=raw_markdown,
                exc=exc,
                failed_dir=failed_dir,
                model_id=model_id,
            )
            typer.echo(f"{slug}: FAIL {written}", err=True)
        except Exception as exc:
            written = write_failed_program(
                slug=slug,
                source_url=url,
                raw_markdown=raw_markdown,
                exc=exc,
                failed_dir=failed_dir,
                model_id=model_id,
            )
            typer.echo(f"{slug}: ERROR {type(exc).__name__}: {exc} {written}", err=True)

    remaining_failed = len(list(failed_dir.glob("*.json"))) if failed_dir.exists() else 0
    typer.echo(f"Успешно: {success_count}/{len(entries)}")
    typer.echo(f"Осталось в _failed/: {remaining_failed}")


@app.command("parse-knowledge")
def parse_knowledge(
    config_path: Path = typer.Option(Path("config/parser.yaml"), help="Путь к parser.yaml"),
    year: int | None = typer.Option(None, "--year"),
    model_id: str = typer.Option("anthropic/claude-sonnet-4.6", "--model-id"),
) -> None:
    config = load_parser_config(config_path)
    resolved_year = resolve_year(year)
    output_dir = config.knowledge_dir(resolved_year)
    output_dir.mkdir(parents=True, exist_ok=True)
    fetcher = raw_fetcher(config)
    cache = Cache(cache_dir=config.cache_dir(resolved_year))
    specs: list[tuple[str, str, type[BaseModel], Any | None]] = [
        ("information", "information", InformationDoc, config.knowledge_urls.get("information")),
        ("olympiads", "olympiads", OlympiadsDoc, config.knowledge_urls.get("olympiads")),
        (
            "achievements",
            "achievements",
            AchievementsDoc,
            config.knowledge_urls.get("achievements"),
        ),
        (
            "exam_programs",
            "exam_programs",
            ExamProgramsDoc,
            config.knowledge_urls.get("exam_programs"),
        ),
        ("faq", "faq", FAQDoc, config.knowledge_urls.get("faq")),
    ]
    for doc_name, prompt_name, schema_type, url_value in specs:
        if url_value is None:
            typer.echo(f"{doc_name}: skip, URL не настроен")
            continue
        try:
            page = fetcher.fetch(str(url_value), year=resolved_year)
            markdown = html_to_markdown(page.html)
            doc: BaseModel = extract(
                markdown,
                schema_type=schema_type,
                prompt_name=prompt_name,
                cache=cache,
                model_id=model_id,
            )
            output_path = output_dir / f"{doc_name}.json"
            raw_html_hash = "sha256:" + hashlib.sha256(page.html.encode("utf-8")).hexdigest()
            preserved = preserved_provenance(
                output_path,
                raw_html_hash=raw_html_hash,
                model_id=model_id,
            )
            retrieved_at = preserved[0] if preserved is not None else datetime.now(UTC)
            extraction_date = preserved[1] if preserved is not None else None
            doc = doc.model_copy(
                update={
                    "doc_id": f"{doc_name}_{resolved_year}",
                    "title": config.knowledge_titles.get(doc_name, doc_name),
                    "url": url_value,
                    "admission_year": resolved_year,
                    "retrieved_at": retrieved_at,
                    "raw_html_hash": raw_html_hash,
                    "generation_metadata": extraction_metadata(
                        model_id,
                        extraction_date=extraction_date,
                    ),
                }
            )
            output_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            (output_dir / "_failed" / f"{doc_name}.json").unlink(missing_ok=True)
            typer.echo(f"{doc_name}: OK")
        except Exception as exc:
            failed_dir = output_dir / "_failed"
            write_json(
                failed_dir / f"{doc_name}.json",
                {"error": str(exc), "type": type(exc).__name__},
            )
            typer.echo(f"{doc_name}: FAIL {exc}", err=True)


@app.command("parse-tables")
def parse_tables(
    config_path: Path = typer.Option(Path("config/parser.yaml"), help="Путь к parser.yaml"),
    year: int | None = typer.Option(None, "--year"),
    model_id: str = typer.Option("anthropic/claude-sonnet-4.6", "--model-id"),
) -> None:
    config = load_parser_config(config_path)
    resolved_year = resolve_year(year)
    fetcher = raw_fetcher(config)
    cache = Cache(cache_dir=config.cache_dir(resolved_year))
    tables = _extract_tables(config, fetcher, cache, resolved_year, model_id, llm=None)
    tables_dir = config.data_root / "tables" / str(resolved_year)
    for name, payload in tables.items():
        write_json(tables_dir / f"{name}.json", payload)
    typer.echo(f"Готово: {len(tables)} таблиц в {tables_dir}")


if __name__ == "__main__":
    app()
