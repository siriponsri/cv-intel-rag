"""
openFDA connector — CV drug-class scoped.

Pulls enforcement (recalls) and drugsfda (approvals) and filters by
pharmacological class keywords for CV/DM/CKD relevance.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import httpx

from ..config.domain import OPENFDA_DRUG_CLASSES
from ..config.settings import settings
from ..models.record import Category, Record, SourceType
from .base import BaseConnector, ConnectorConfig

log = logging.getLogger(__name__)


class OpenFDAConnector(BaseConnector):
    BASE = "https://api.fda.gov"
    ENDPOINTS = [
        ("/drug/enforcement.json", Category.SAFETY,        "report_date"),
        ("/drug/drugsfda.json",    Category.DRUG_APPROVAL, "submissions.submission_status_date"),
    ]

    def __init__(
        self,
        days_back: int = 30,
        cfg: Optional[ConnectorConfig] = None,
    ):
        super().__init__(cfg or ConnectorConfig(
            source_name="openFDA",
            country_or_region="US",
            rate_limit_per_sec=4.0,
            max_records_per_run=200,
        ))
        self.days_back = days_back

    def fetch(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Iterable[Record]:
        cap = limit or self.cfg.max_records_per_run
        yielded = 0
        cutoff = since.date() if since else date.today() - timedelta(days=self.days_back)

        for path, category, date_field in self.ENDPOINTS:
            if yielded >= cap:
                break
            for record in self._fetch_endpoint(path, category, date_field, cutoff, min(50, cap - yielded)):
                if yielded >= cap:
                    break
                # Domain filter — only yield if CV-relevant
                if self._is_cv_relevant(record):
                    yielded += 1
                    yield record

    def _is_cv_relevant(self, record: Record) -> bool:
        """Cheap keyword filter — title + summary + tags."""
        blob = " ".join([
            record.title or "",
            record.summary_short or "",
            " ".join(record.topic_tags),
            " ".join(record.entity_tags),
        ]).lower()
        return any(cls in blob for cls in OPENFDA_DRUG_CLASSES)

    def _fetch_endpoint(
        self,
        path: str,
        category: Category,
        date_field: str,
        cutoff: date,
        limit: int,
    ) -> Iterable[Record]:
        self.rate_limit()
        cutoff_str = cutoff.strftime("%Y%m%d")
        params = {
            "search": f"{date_field}:[{cutoff_str}+TO+99991231]",
            "limit": limit,
        }
        if settings.openfda_api_key:
            params["api_key"] = settings.openfda_api_key

        url = self.BASE + path
        log.info("[openFDA] %s", path)

        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(url, params=params)
                if r.status_code == 404:
                    return
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as exc:
            log.warning("[openFDA] failed: %s", exc)
            return

        for item in data.get("results", []):
            try:
                if "enforcement" in path:
                    yield self._parse_enforcement(item)
                elif "drugsfda" in path:
                    yield self._parse_drugsfda(item)
            except Exception as exc:                                           # noqa: BLE001
                log.warning("[openFDA] parse error: %s", exc)

    @staticmethod
    def _safe_date(raw: str | None) -> Optional[date]:
        if not raw:
            return None
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw, fmt).date()
            except (ValueError, TypeError):
                continue
        return None

    def _parse_enforcement(self, item: dict) -> Record:
        rec_num = item.get("recall_number", "") or item.get("event_id", "")
        product = item.get("product_description", "")[:500]
        reason = item.get("reason_for_recall", "")[:500]
        firm = item.get("recalling_firm", "")
        classification = item.get("classification", "")

        full_text = (
            f"Product: {product}\n\n"
            f"Reason for recall: {reason}\n\n"
            f"Classification: {classification}\n"
            f"Recalling firm: {firm}"
        )[:5000]

        return Record(
            title=f"Recall: {product[:200]}",
            source_name="openFDA",
            source_type=SourceType.API,
            country_or_region="US",
            published_date=self._safe_date(item.get("report_date")),
            url="https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts",
            category=Category.SAFETY,
            topic_tags=["recall", classification] if classification else ["recall"],
            entity_tags=[firm] if firm else [],
            summary_short=f"[{classification}] {reason}"[:500] if classification else reason[:500],
            raw_text=full_text,
            external_id=rec_num or None,
        )

    def _parse_drugsfda(self, item: dict) -> Record:
        app_num = item.get("application_number", "")
        sponsor = item.get("sponsor_name", "")
        submissions = item.get("submissions") or []
        latest = sorted(
            submissions, key=lambda s: s.get("submission_status_date") or "0", reverse=True
        )[0] if submissions else {}

        status = latest.get("submission_status", "")
        status_date = self._safe_date(latest.get("submission_status_date"))

        products = item.get("products") or []
        brand_names = [p.get("brand_name") for p in products if p.get("brand_name")]
        active_ingredients = []
        for p in products:
            for ai in p.get("active_ingredients", []) or []:
                if ai.get("name"):
                    active_ingredients.append(ai["name"])
        active_ingredients = list(dict.fromkeys(active_ingredients))[:5]

        title = f"FDA Application {app_num}: " + (brand_names[0] if brand_names else "(no brand)")
        full_text = (
            f"{title}\n\n"
            f"Status: {status}\n"
            f"Sponsor: {sponsor}\n"
            f"Active ingredients: {', '.join(active_ingredients)}\n"
            f"Brand names: {', '.join(brand_names)}"
        )[:5000]

        return Record(
            title=title[:2000],
            source_name="openFDA",
            source_type=SourceType.API,
            country_or_region="US",
            published_date=status_date,
            url=f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={app_num}",
            category=Category.DRUG_APPROVAL,
            topic_tags=[status] if status else [],
            entity_tags=[sponsor] + active_ingredients,
            summary_short=f"Status: {status}. Active: {', '.join(active_ingredients) or 'N/A'}"[:500],
            raw_text=full_text,
            external_id=app_num or None,
        )
