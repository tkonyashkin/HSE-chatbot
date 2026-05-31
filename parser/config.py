from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, HttpUrl


class TablesConfig(BaseModel):
    minkrit: dict[str, HttpUrl] = Field(default_factory=dict)
    kolmest: dict[str, HttpUrl] = Field(default_factory=dict)
    price: dict[str, HttpUrl] = Field(default_factory=dict)
    results: dict[str, HttpUrl] = Field(default_factory=dict)


class ParserConfig(BaseModel):
    data_root: Path = Field(default=Path("data"))
    year_default: int = Field(default=2026, ge=2020, le=2100)
    catalog_url: HttpUrl
    program_url_pattern: str
    tables: TablesConfig
    information_url: HttpUrl | None = None
    knowledge_urls: dict[str, HttpUrl] = Field(default_factory=dict)
    knowledge_titles: dict[str, str] = Field(default_factory=dict)
    url_classifier: dict[str, str] = Field(default_factory=dict)
    auxiliary_slugs: set[str] = Field(default_factory=set)

    def raw_dir(self, year: int | None = None) -> Path:
        return self.data_root / "raw" / str(year or self.year_default)

    def programs_dir(self, year: int | None = None) -> Path:
        return self.data_root / "programs" / str(year or self.year_default)

    def knowledge_dir(self, year: int | None = None) -> Path:
        return self.data_root / "knowledge" / str(year or self.year_default)

    def cache_dir(self, year: int | None = None) -> Path:
        return self.data_root / "llm_cache" / str(year or self.year_default)

    def failed_dir(self, year: int | None = None) -> Path:
        return self.programs_dir(year) / "_failed"


def load_parser_config(path: Path) -> ParserConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid parser config at {path}: expected mapping")
    return ParserConfig.model_validate(raw)


def aux_slugs_for(config: ParserConfig) -> frozenset[str]:
    slugs: set[str] = set(config.auxiliary_slugs)
    for url_map in (
        config.tables.minkrit,
        config.tables.kolmest,
        config.tables.price,
        config.tables.results,
    ):
        for url in url_map.values():
            path = urlsplit(str(url)).path.rstrip("/")
            last = path.rsplit("/", 1)[-1]
            if last:
                slugs.add(last)
    return frozenset(slugs)
