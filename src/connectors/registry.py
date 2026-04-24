"""
Connector registry for CV + DM + CKD domain.

Minimal set for MVP: 3 structured APIs + 4 RSS feeds.
Expand (e.g. Thai FDA scraper) later once RAG quality is solid.
"""
from __future__ import annotations

from typing import Callable

from ..models.record import Category
from .base import BaseConnector
from .clinical_trials import ClinicalTrialsConnector
from .openfda import OpenFDAConnector
from .pubmed import PubMedConnector
from .rss import RSSConnector


CONNECTORS: dict[str, Callable[[], BaseConnector]] = {
    # ── Structured APIs (Layer 1) ──────────────────────────────
    "pubmed":          PubMedConnector,
    "clinicaltrials":  ClinicalTrialsConnector,
    "openfda":         OpenFDAConnector,

    # ── RSS feeds ──────────────────────────────────────────────
    "rss_medwatch":    lambda: RSSConnector(feed_key="fda_medwatch",   category=Category.SAFETY),
    "rss_ema":         lambda: RSSConnector(feed_key="ema_whats_new",  category=Category.REGULATOR),
    "rss_esc":         lambda: RSSConnector(feed_key="escardio_news",  category=Category.GUIDELINE),
    "rss_aha":         lambda: RSSConnector(feed_key="ahajournals",    category=Category.RESEARCH),
}


def get_connector(key: str) -> BaseConnector:
    if key not in CONNECTORS:
        raise KeyError(f"Unknown connector: {key}. Available: {sorted(CONNECTORS)}")
    return CONNECTORS[key]()


def all_connectors() -> list[BaseConnector]:
    return [factory() for factory in CONNECTORS.values()]
