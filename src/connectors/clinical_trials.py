"""
ClinicalTrials.gov v2 — CV + Diabetes + CKD scope.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import httpx

from ..config.domain import CTGOV_CONDITIONS
from ..models.record import Category, Record, SourceType
from .base import BaseConnector, ConnectorConfig

log = logging.getLogger(__name__)


class ClinicalTrialsConnector(BaseConnector):
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(
        self,
        conditions: Optional[list[str]] = None,
        days_back: int = 14,
        cfg: Optional[ConnectorConfig] = None,
    ):
        super().__init__(cfg or ConnectorConfig(
            source_name="ClinicalTrials.gov",
            country_or_region="Global",
            rate_limit_per_sec=2.0,
            max_records_per_run=100,
        ))
        self.conditions = conditions or CTGOV_CONDITIONS
        self.days_back = days_back

    def fetch(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Iterable[Record]:
        cap = limit or self.cfg.max_records_per_run
        yielded = 0
        seen_nct: set[str] = set()

        cutoff = since.date() if since else date.today() - timedelta(days=self.days_back)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        for term in self.conditions:
            if yielded >= cap:
                break
            self.rate_limit()
            params = {
                "query.cond": term,
                "filter.advanced": f"AREA[LastUpdatePostDate]RANGE[{cutoff_str},MAX]",
                "pageSize": min(25, cap - yielded),
                "format": "json",
            }
            log.info("[ClinicalTrials] cond=%s since=%s", term, cutoff_str)
            try:
                with httpx.Client(timeout=30.0) as client:
                    r = client.get(self.BASE_URL, params=params)
                    r.raise_for_status()
                    data = r.json()
            except httpx.HTTPError as exc:
                log.warning("[ClinicalTrials] failed: %s", exc)
                continue

            for study in data.get("studies", []):
                if yielded >= cap:
                    break
                try:
                    record = self._parse_study(study, term)
                    if record and record.external_id not in seen_nct:
                        seen_nct.add(record.external_id)
                        yielded += 1
                        yield record
                except Exception as exc:                                       # noqa: BLE001
                    log.warning("[ClinicalTrials] parse error: %s", exc)

    def _parse_study(self, study: dict, query_term: str) -> Optional[Record]:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        cond = proto.get("conditionsModule", {})
        design = proto.get("designModule", {})
        sponsor = proto.get("sponsorCollaboratorsModule", {})
        desc = proto.get("descriptionModule", {})

        nct_id = ident.get("nctId", "").strip()
        if not nct_id:
            return None

        title = (ident.get("briefTitle") or ident.get("officialTitle") or "").strip()[:2000]

        post_date_raw = status.get("lastUpdatePostDateStruct", {}).get("date")
        pub_date = None
        if post_date_raw:
            try:
                pub_date = datetime.strptime(post_date_raw, "%Y-%m-%d").date()
            except ValueError:
                try:
                    pub_date = datetime.strptime(post_date_raw, "%Y-%m").date()
                except ValueError:
                    pass

        brief = desc.get("briefSummary", "").strip()
        detailed = desc.get("detailedDescription", "").strip()
        conditions = cond.get("conditions", []) or []
        phases = design.get("phases", []) or []
        lead_sponsor = sponsor.get("leadSponsor", {}).get("name", "")

        # Full text for RAG
        full_text = (
            f"{title}\n\n"
            f"Brief Summary: {brief}\n\n"
            f"Detailed Description: {detailed}\n\n"
            f"Conditions: {', '.join(conditions)}\n"
            f"Phases: {', '.join(phases)}\n"
            f"Sponsor: {lead_sponsor}"
        )[:10000]

        return Record(
            title=title or f"Clinical Trial {nct_id}",
            source_name="ClinicalTrials.gov",
            source_type=SourceType.API,
            country_or_region="Global",
            published_date=pub_date,
            url=f"https://clinicaltrials.gov/study/{nct_id}",
            category=Category.CLINICAL_TRIAL,
            topic_tags=conditions[:5] + phases[:2] + [query_term],
            entity_tags=[lead_sponsor] if lead_sponsor else [],
            summary_short=brief[:500] if brief else None,
            raw_text=full_text,
            external_id=nct_id,
        )
