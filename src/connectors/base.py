"""
Base connector class for Layer 1 data sources.

Every connector (PubMed, ClinicalTrials.gov, openFDA, THFDA Catalog, …)
subclasses `BaseConnector` and implements `fetch()` → iterable of `Record`.

Provides uniform:
  • ingestion_run bookkeeping
  • dedup via (source_name, external_id, external_version)
  • raw_payload storage
  • basic retry + rate limiting
"""
from __future__ import annotations

import abc
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional

from ..models.record import IngestionRun, Record

log = logging.getLogger(__name__)


# ─── Config struct ────────────────────────────────────────────────────────

@dataclass
class ConnectorConfig:
    """Runtime config for a connector."""
    source_name:          str                        # e.g. 'PubMed', 'openFDA'
    country_or_region:    str                        # 'Global', 'US', 'Thailand', ...
    rate_limit_per_sec:   float = 2.0                # requests per second cap
    max_records_per_run:  int = 500                  # safety cap for MVP
    retry_attempts:       int = 3
    retry_backoff_sec:    float = 2.0
    user_agent:           str = "PharmaRegIntel/0.1 (internal)"


# ─── Result container ─────────────────────────────────────────────────────

@dataclass
class FetchResult:
    """What a single connector execution produced."""
    records:         List[Record] = field(default_factory=list)
    raw_payloads:    List[dict]   = field(default_factory=list)   # list of {record_id, content_type, raw_text, checksum}
    errors:          List[str]    = field(default_factory=list)


# ─── Base connector ──────────────────────────────────────────────────────

class BaseConnector(abc.ABC):
    """
    Abstract base for all Layer 1 connectors.

    Subclass contract:
      • implement `fetch(since=..., limit=...)` — yields Record objects
      • optionally override `rate_limit()` if the source needs special pacing
    """

    def __init__(self, cfg: ConnectorConfig):
        self.cfg = cfg
        self._last_request_at: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────

    @abc.abstractmethod
    def fetch(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Iterable[Record]:
        """
        Yield Record objects from the source.

        Args:
            since: only return records published/updated after this timestamp.
            limit: max records to yield (overrides cfg.max_records_per_run if smaller).
        """
        ...

    def run(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> tuple[IngestionRun, FetchResult]:
        """
        Execute the connector, wrap in an IngestionRun audit record, and
        return both so callers can persist them.
        """
        run = IngestionRun(source_name=self.cfg.source_name, status="running")
        result = FetchResult()

        cap = min(limit or self.cfg.max_records_per_run, self.cfg.max_records_per_run)
        log.info("[%s] starting ingestion run (cap=%d, since=%s)", self.cfg.source_name, cap, since)

        try:
            for idx, record in enumerate(self.fetch(since=since, limit=cap)):
                result.records.append(record)
                run.records_fetched += 1
                if (idx + 1) >= cap:
                    log.warning("[%s] hit max_records cap %d", self.cfg.source_name, cap)
                    break
            run.status = "success"
        except Exception as exc:                                    # noqa: BLE001 (we want the catch-all for audit)
            log.exception("[%s] connector failed", self.cfg.source_name)
            run.status = "failed"
            run.error_message = str(exc)
            result.errors.append(str(exc))
        finally:
            run.finished_at = datetime.utcnow()

        log.info(
            "[%s] run finished: status=%s fetched=%d errors=%d",
            self.cfg.source_name, run.status, run.records_fetched, len(result.errors),
        )
        return run, result

    # ── Helpers for subclasses ──────────────────────────────────────────

    def rate_limit(self) -> None:
        """Block until at least 1/rate_limit_per_sec has passed since last call."""
        min_interval = 1.0 / max(self.cfg.rate_limit_per_sec, 0.01)
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_at = time.monotonic()

    @staticmethod
    def checksum(text: str) -> str:
        """SHA-256 checksum for raw payload dedup."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    def build_raw_payload(
        self,
        record_id: str,
        raw_text: str,
        content_type: str = "application/json",
    ) -> dict:
        """Build a raw_payload dict ready for persistence."""
        return {
            "payload_id":  self.new_id(),
            "record_id":   record_id,
            "content_type": content_type,
            "raw_text":    raw_text,
            "fetched_at":  datetime.utcnow(),
            "checksum":    self.checksum(raw_text),
        }
