from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import ATX, BACKSLASH, markdownify
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

USER_AGENT = "HSEAdmissionBot-Research/0.1 (coursework; +https://github.com/example/hse-bot)"

NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "footer",
    "header",
    "[role=banner]",
    "[role=navigation]",
    "[role=contentinfo]",
    "form[role=search]",
    "iframe",
    "object",
    "embed",
]


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    md: str = markdownify(
        str(soup),
        heading_style=ATX,
        newline_style=BACKSLASH,
        bullets="-",
        table_infer_header=True,
        autolinks=False,
        keep_inline_images_in=[],
        escape_asterisks=False,
        escape_underscores=False,
    )
    return md.replace("\xa0", " ").replace("\u202f", " ")


def resolve_year() -> int:
    return int(os.environ.get("HSE_ADMISSION_YEAR", "2026"))


@dataclass
class FetchResult:
    url: str
    html: str
    content_hash: str
    from_cache: bool
    fetched_at: float


class Fetcher:
    def __init__(
        self,
        cache_dir: Path,
        rate_limit_seconds: float = 1.0,
        timeout: float = 30.0,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout = timeout
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def fetch(
        self,
        url: str,
        force_refresh: bool = False,
        year: int | None = None,
    ) -> FetchResult:
        cache_path = self._cache_path(url, year=year)
        if not force_refresh and cache_path.exists():
            html = cache_path.read_text(encoding="utf-8")
            return FetchResult(
                url=url,
                html=html,
                content_hash=hash_html(html),
                from_cache=True,
                fetched_at=cache_path.stat().st_mtime,
            )
        self._wait_for_rate_limit()
        html = self._download(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")
        return FetchResult(
            url=url,
            html=html,
            content_hash=hash_html(html),
            from_cache=False,
            fetched_at=time.time(),
        )

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self.last_request_time = time.time()

    @retry(
        retry=retry_if_exception_type(
            (requests.HTTPError, requests.ConnectionError, requests.Timeout)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _download(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def _cache_path(self, url: str, year: int | None = None) -> Path:
        year_value = year if year is not None else resolve_year()
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / str(year_value) / f"{key}.html"


def hash_html(html: str) -> str:
    return "sha256:" + hashlib.sha256(html.encode("utf-8")).hexdigest()
