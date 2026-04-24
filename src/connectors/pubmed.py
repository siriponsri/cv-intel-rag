"""
PubMed connector scoped to Cardiovascular + Diabetes + CKD.

Uses MeSH terms for precision. Pulls last N days.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Iterable, Optional

import httpx

from ..config.domain import PUBMED_MESH_QUERIES
from ..config.settings import settings
from ..models.record import Category, Record, SourceType
from .base import BaseConnector, ConnectorConfig

log = logging.getLogger(__name__)


class PubMedConnector(BaseConnector):
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(
        self,
        queries: Optional[list[str]] = None,
        days_back: int = 90,
        cfg: Optional[ConnectorConfig] = None,
    ):
        super().__init__(cfg or ConnectorConfig(
            source_name="PubMed",
            country_or_region="Global",
            rate_limit_per_sec=3.0 if not settings.ncbi_api_key else 10.0,
            max_records_per_run=100,
        ))
        self.queries = queries or PUBMED_MESH_QUERIES
        self.days_back = days_back

    def fetch(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Iterable[Record]:
        cap = limit or self.cfg.max_records_per_run
        seen_pmids: set[str] = set()

        for query in self.queries:
            if len(seen_pmids) >= cap:
                break
            try:
                pmids = self._esearch(query, retmax=min(25, cap - len(seen_pmids)))
            except httpx.HTTPError as exc:
                log.warning("[PubMed] esearch failed for %s: %s", query, exc)
                continue

            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)
            if not new_pmids:
                continue

            try:
                xml = self._efetch(new_pmids)
                yield from self._parse_efetch_xml(xml, query)
            except httpx.HTTPError as exc:
                log.warning("[PubMed] efetch failed: %s", exc)

    # ── HTTP ────────────────────────────────────────────────────────

    def _auth_params(self) -> dict:
        params = {}
        if settings.ncbi_api_key:
            params["api_key"] = settings.ncbi_api_key
        if settings.ncbi_email:
            params["email"] = settings.ncbi_email
        return params

    def _esearch(self, query: str, retmax: int = 25) -> list[str]:
        self.rate_limit()
        params = {
            "db": "pubmed",
            "term": f"{query} AND last {self.days_back} days[dp]",
            "retmode": "json",
            "retmax": retmax,
            "sort": "date",
            **self._auth_params(),
        }
        log.info("[PubMed] esearch: %s", query[:50])
        with httpx.Client(timeout=30.0) as client:
            r = client.get(self.ESEARCH_URL, params=params)
            r.raise_for_status()
            return r.json().get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmids: list[str]) -> str:
        self.rate_limit()
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            **self._auth_params(),
        }
        log.info("[PubMed] efetch: %d PMIDs", len(pmids))
        with httpx.Client(timeout=60.0) as client:
            r = client.get(self.EFETCH_URL, params=params)
            r.raise_for_status()
            return r.text

    # ── parsing ─────────────────────────────────────────────────────

    def _parse_efetch_xml(self, xml_text: str, query: str) -> Iterable[Record]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            log.error("[PubMed] XML parse error: %s", exc)
            return

        for article in root.findall(".//PubmedArticle"):
            try:
                pmid = article.findtext(".//PMID", default="").strip()
                title = article.findtext(".//ArticleTitle", default="(no title)").strip()
                journal = article.findtext(".//Journal/Title", default="").strip()
                pub_date = self._extract_pub_date(article)

                abstract_parts = [
                    (t.text or "").strip()
                    for t in article.findall(".//Abstract/AbstractText")
                    if t.text
                ]
                abstract = " ".join(abstract_parts)[:5000]    # keep longer for RAG chunking

                mesh_tags = [
                    (d.text or "").strip()
                    for d in article.findall(".//MeshHeading/DescriptorName")
                    if d.text
                ][:15]

                # Full text for RAG: title + abstract + journal
                full_text = f"{title}\n\n{abstract}\n\nJournal: {journal}\nMeSH: {', '.join(mesh_tags)}"

                yield Record(
                    title=title[:2000],
                    source_name="PubMed",
                    source_type=SourceType.API,
                    country_or_region="Global",
                    published_date=pub_date,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    category=Category.RESEARCH,
                    topic_tags=mesh_tags,
                    entity_tags=[journal] if journal else [],
                    summary_short=abstract[:500] if abstract else None,
                    raw_text=full_text,
                    external_id=pmid,
                )
            except Exception as exc:                                          # noqa: BLE001
                log.warning("[PubMed] skipped article: %s", exc)

    @staticmethod
    def _extract_pub_date(article: ET.Element) -> Optional[date]:
        for path in (".//PubDate", ".//ArticleDate"):
            node = article.find(path)
            if node is None:
                continue
            y = node.findtext("Year")
            m = node.findtext("Month") or "1"
            d = node.findtext("Day") or "1"
            if not y:
                continue
            try:
                months = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
                mi = months.get(m, int(m) if m.isdigit() else 1)
                return date(int(y), mi, int(d))
            except (ValueError, TypeError):
                continue
        return None
