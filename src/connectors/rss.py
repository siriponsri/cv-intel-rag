"""
Generic RSS connector — one feed per instance.

Filters entries by CV_TERM_REGEX to stay on-domain (CV + DM + CKD).
"""
from __future__ import annotations

import logging
import re
from datetime import date
from datetime import datetime
from time import struct_time
from typing import Iterable, Optional

import feedparser

from ..config.domain import CV_RSS_FEEDS, CV_TERM_REGEX
from ..models.record import Category, Record, SourceType
from .base import BaseConnector, ConnectorConfig

log = logging.getLogger(__name__)

_CV_RE = re.compile(CV_TERM_REGEX)


class RSSConnector(BaseConnector):
    def __init__(
        self,
        feed_key: Optional[str] = None,
        *,
        url: Optional[str] = None,
        source_name: Optional[str] = None,
        region: Optional[str] = None,
        category: Category = Category.OTHER,
        apply_cv_filter: bool = True,
        cfg: Optional[ConnectorConfig] = None,
    ):
        if feed_key and feed_key in CV_RSS_FEEDS:
            meta = CV_RSS_FEEDS[feed_key]
            url = url or meta["url"]
            source_name = source_name or meta["source_name"]
            region = region or meta["region"]

        if not (url and source_name and region):
            raise ValueError("RSSConnector requires url, source_name, region")

        self.url = url
        self.category = category
        self.apply_cv_filter = apply_cv_filter
        super().__init__(cfg or ConnectorConfig(
            source_name=source_name,
            country_or_region=region,
            rate_limit_per_sec=2.0,
            max_records_per_run=50,
        ))

    def fetch(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Iterable[Record]:
        cap = limit or self.cfg.max_records_per_run
        self.rate_limit()
        log.info("[RSS %s] fetching %s", self.cfg.source_name, self.url)
        parsed = feedparser.parse(self.url)
        if parsed.bozo and not parsed.entries:
            log.warning("[RSS %s] parse error: %s", self.cfg.source_name, parsed.bozo_exception)
            return

        yielded = 0
        for entry in parsed.entries:
            if yielded >= cap:
                break
            try:
                record = self._parse_entry(entry)
            except Exception as exc:                                              # noqa: BLE001
                log.warning("[RSS] skipped entry: %s", exc)
                continue

            if self.apply_cv_filter and not self._is_cv_relevant(record):
                continue
            yielded += 1
            yield record

    def _is_cv_relevant(self, record: Record) -> bool:
        blob = " ".join([record.title or "", record.summary_short or ""])
        return bool(_CV_RE.search(blob))

    def _parse_entry(self, entry) -> Record:
        title = (getattr(entry, "title", None) or "(no title)")[:2000]
        link = getattr(entry, "link", None)
        summary = (getattr(entry, "summary", None) or getattr(entry, "description", "") or "")[:2000]
        pub_date = self._entry_date(entry)
        ext_id = getattr(entry, "id", None) or link

        full_text = f"{title}\n\n{summary}"

        return Record(
            title=title,
            source_name=self.cfg.source_name,
            source_type=SourceType.RSS,
            country_or_region=self.cfg.country_or_region,
            published_date=pub_date,
            url=link,
            category=self.category,
            topic_tags=[t.get("term") for t in getattr(entry, "tags", []) if t.get("term")][:10],
            entity_tags=[],
            summary_short=summary[:500] if summary else None,
            raw_text=full_text,
            external_id=ext_id,
        )

    @staticmethod
    def _entry_date(entry) -> Optional[date]:
        for attr in ("published_parsed", "updated_parsed"):
            tm: struct_time | None = getattr(entry, attr, None)
            if tm:
                try:
                    return date(tm.tm_year, tm.tm_mon, tm.tm_mday)
                except (ValueError, TypeError):
                    continue
        return None
