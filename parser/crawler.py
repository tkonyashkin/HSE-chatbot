import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from parser.schemas import KnowledgeDocType, TableType
from parser.url_classifier import UrlRole, classify_url


@dataclass
class ProgramListing:
    slug: str
    name: str
    url: str


class CatalogCrawler:
    def __init__(
        self,
        aux_slug_blacklist: frozenset[str] | None = None,
        url_classifier: Mapping[str, str | UrlRole] | None = None,
    ):
        self.aux_slugs = aux_slug_blacklist or frozenset()
        self.url_classifier = url_classifier or {}

    def parse_catalog(self, html: str, base_url: str) -> list[ProgramListing]:
        soup = BeautifulSoup(html, "html.parser")
        listings: dict[str, ProgramListing] = {}
        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"]
            absolute = urljoin(base_url, href)
            slug = self.extract_slug(absolute, base_url)
            if slug is None:
                continue
            text = anchor.get_text(strip=True)
            if not text or len(text) < 3:
                continue
            if slug in listings:
                continue
            listings[slug] = ProgramListing(slug=slug, name=text, url=absolute)
        return list(listings.values())

    def discover_knowledge_urls(self, html: str, base_url: str) -> dict[KnowledgeDocType, str]:
        soup = BeautifulSoup(html, "html.parser")
        discovered: dict[KnowledgeDocType, str] = {}
        for anchor in soup.find_all("a", href=True):
            absolute = urljoin(base_url, anchor["href"])
            role = classify_url(absolute, self.url_classifier)
            if isinstance(role, KnowledgeDocType) and role not in discovered:
                discovered[role] = absolute
        return discovered

    def discover_table_urls(
        self,
        html: str,
        base_url: str,
    ) -> dict[TableType | tuple[TableType, int], str]:
        soup = BeautifulSoup(html, "html.parser")
        discovered: dict[TableType | tuple[TableType, int], str] = {}
        for anchor in soup.find_all("a", href=True):
            absolute = urljoin(base_url, anchor["href"])
            role = classify_url(absolute, self.url_classifier)
            if isinstance(role, TableType | tuple):
                key: TableType | tuple[TableType, int] = role
                if key not in discovered:
                    discovered[key] = absolute
        return discovered

    def extract_slug(self, url: str, base: str) -> str | None:
        parts = urlsplit(url)
        url_clean = f"{parts.scheme}://{parts.netloc}{parts.path}"
        if not url_clean.startswith(base):
            return None
        suffix = url_clean[len(base) :].rstrip("/")
        if not suffix or "/" in suffix:
            return None
        if suffix in self.aux_slugs:
            return None
        if not re.match(r"^[a-z0-9_-]+$", suffix):
            return None
        return suffix
