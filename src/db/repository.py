"""
RecordRepository — the only way connectors and the API talk to the `records` table.

Responsibilities:
  • upsert(Record, raw_text=...)   — idempotent insert/update based on (source_name, external_id, external_version)
  • find_by_external_id(...)       — source-specific lookup
  • list_since(datetime)           — records ingested after a timestamp
  • export_csv(path, ...)          — Layer 1 CSV dump
  • record_ingestion_run(...)      — persist an IngestionRun audit row
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.record import IngestionRun, Record, record_to_csv_row
from .models import (
    IngestionRunORM, RawPayloadORM, RecordORM, RecordTagORM, TagORM,
)

log = logging.getLogger(__name__)


class RecordRepository:
    """Persistence boundary between Pydantic Records and the DB."""

    def __init__(self, session: Session):
        self.session = session

    # ─── upsert ─────────────────────────────────────────────────────

    def upsert(
        self,
        record: Record,
        raw_text: Optional[str] = None,
        content_type: str = "application/json",
        checksum: Optional[str] = None,
    ) -> tuple[RecordORM, bool]:
        """
        Insert or update a record. Returns (orm, is_new).

        Dedup key: (source_name, external_id, external_version).
        If no external_id is provided, a new row is always created.
        """
        existing: Optional[RecordORM] = None
        if record.external_id:
            stmt = select(RecordORM).where(
                RecordORM.source_name == record.source_name,
                RecordORM.external_id == record.external_id,
                RecordORM.external_version == (record.external_version or None),
            )
            existing = self.session.execute(stmt).scalar_one_or_none()

        is_new = existing is None
        orm = existing or RecordORM(record_id=record.record_id)

        # project scalar fields
        orm.title             = record.title
        orm.source_name       = record.source_name
        orm.source_type       = record.source_type if isinstance(record.source_type, str) else record.source_type.value
        orm.layer             = record.layer
        orm.country_or_region = record.country_or_region
        orm.published_date    = record.published_date
        orm.ingested_at       = record.ingested_at
        orm.url               = record.url
        orm.category          = record.category if isinstance(record.category, str) else record.category.value
        orm.summary_short     = record.summary_short
        orm.confidence        = record.confidence if isinstance(record.confidence, str) else (record.confidence.value if record.confidence else None)
        orm.review_status     = record.review_status if isinstance(record.review_status, str) else record.review_status.value
        orm.exportable_to_csv = record.exportable_to_csv
        orm.external_id       = record.external_id
        orm.external_version  = record.external_version
        orm.language          = record.language
        orm.updated_at        = datetime.utcnow()

        if is_new:
            self.session.add(orm)
            self.session.flush()                          # get the PK before we attach tags

        # tags ───
        self._replace_tags(orm.record_id, record.topic_tags, tag_type="topic")
        self._replace_tags(orm.record_id, record.entity_tags, tag_type="entity")

        # raw payload ───
        if raw_text:
            # Dedup on (record_id, checksum) — only insert if new content
            if checksum:
                from sqlalchemy import select as _select
                existing_payload = self.session.execute(
                    _select(RawPayloadORM).where(
                        RawPayloadORM.record_id == orm.record_id,
                        RawPayloadORM.checksum == checksum,
                    )
                ).scalar_one_or_none()
                if not existing_payload:
                    self.session.add(RawPayloadORM(
                        payload_id=self._new_payload_id(),
                        record_id=orm.record_id,
                        content_type=content_type,
                        raw_text=raw_text,
                        fetched_at=datetime.utcnow(),
                        checksum=checksum,
                    ))
            else:
                self.session.add(RawPayloadORM(
                    payload_id=self._new_payload_id(),
                    record_id=orm.record_id,
                    content_type=content_type,
                    raw_text=raw_text,
                    fetched_at=datetime.utcnow(),
                    checksum=None,
                ))

        return orm, is_new

    @staticmethod
    def _new_payload_id() -> str:
        import uuid as _uuid
        return str(_uuid.uuid4())

    # ─── tags ───────────────────────────────────────────────────────

    def _replace_tags(self, record_id: str, values: Sequence[str], *, tag_type: str) -> None:
        # drop old associations of this type
        stale = self.session.execute(
            select(RecordTagORM, TagORM)
            .join(TagORM, TagORM.tag_id == RecordTagORM.tag_id)
            .where(RecordTagORM.record_id == record_id, TagORM.tag_type == tag_type)
        ).all()
        for assoc, _ in stale:
            self.session.delete(assoc)

        for value in values:
            tag = self._get_or_create_tag(tag_type=tag_type, tag_value=value)
            self.session.add(RecordTagORM(record_id=record_id, tag_id=tag.tag_id))

    def _get_or_create_tag(self, *, tag_type: str, tag_value: str) -> TagORM:
        existing = self.session.execute(
            select(TagORM).where(TagORM.tag_type == tag_type, TagORM.tag_value == tag_value)
        ).scalar_one_or_none()
        if existing:
            return existing
        tag = TagORM(tag_type=tag_type, tag_value=tag_value, normalized=tag_value.lower().strip())
        self.session.add(tag)
        self.session.flush()
        return tag

    # ─── queries ────────────────────────────────────────────────────

    def find_by_external_id(
        self,
        source_name: str,
        external_id: str,
        external_version: Optional[str] = None,
    ) -> Optional[RecordORM]:
        stmt = select(RecordORM).where(
            RecordORM.source_name == source_name,
            RecordORM.external_id == external_id,
            RecordORM.external_version == (external_version or None),
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_since(self, since: datetime, limit: int = 100) -> list[RecordORM]:
        stmt = (
            select(RecordORM)
            .where(RecordORM.ingested_at >= since)
            .order_by(RecordORM.ingested_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def list_by_source(self, source_name: str, limit: int = 100) -> list[RecordORM]:
        stmt = (
            select(RecordORM)
            .where(RecordORM.source_name == source_name)
            .order_by(RecordORM.ingested_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def tags_for(self, record_id: str, tag_type: Optional[str] = None) -> list[str]:
        stmt = (
            select(TagORM.tag_value)
            .join(RecordTagORM, RecordTagORM.tag_id == TagORM.tag_id)
            .where(RecordTagORM.record_id == record_id)
        )
        if tag_type:
            stmt = stmt.where(TagORM.tag_type == tag_type)
        return [row[0] for row in self.session.execute(stmt).all()]

    # ─── ingestion_runs ─────────────────────────────────────────────

    def record_ingestion_run(self, run: IngestionRun) -> IngestionRunORM:
        orm = IngestionRunORM(
            run_id          = run.run_id,
            source_name     = run.source_name,
            started_at      = run.started_at,
            finished_at     = run.finished_at,
            status          = run.status,
            records_fetched = run.records_fetched,
            records_new     = run.records_new,
            records_updated = run.records_updated,
            error_message   = run.error_message,
            notes           = run.notes,
        )
        self.session.add(orm)
        return orm

    # ─── CSV export (Layer 1 only) ──────────────────────────────────

    def export_csv(
        self,
        output_path: Path,
        since: Optional[datetime] = None,
        source_name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> int:
        """
        Dump Layer 1 records to CSV. Returns row count written.

        Tags are aggregated and written as pipe-delimited strings
        (matches the `csv_export` view in schema.sql).
        """
        stmt = select(RecordORM).where(RecordORM.exportable_to_csv == True)  # noqa: E712
        if since:
            stmt = stmt.where(RecordORM.ingested_at >= since)
        if source_name:
            stmt = stmt.where(RecordORM.source_name == source_name)
        if category:
            stmt = stmt.where(RecordORM.category == category)
        stmt = stmt.order_by(RecordORM.ingested_at.desc())

        orm_rows = list(self.session.execute(stmt).scalars().all())

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(record_to_csv_row(self._orm_to_pydantic_stub()).keys())
        n = 0
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in orm_rows:
                topic = self.tags_for(r.record_id, "topic")
                entity = self.tags_for(r.record_id, "entity")
                writer.writerow({
                    "record_id":         r.record_id,
                    "title":             r.title,
                    "source_name":       r.source_name,
                    "source_type":       r.source_type,
                    "country_or_region": r.country_or_region,
                    "published_date":    r.published_date.isoformat() if r.published_date else "",
                    "ingested_at":       r.ingested_at.isoformat(),
                    "url":               r.url or "",
                    "topic_tags":        "|".join(topic),
                    "entity_tags":       "|".join(entity),
                    "category":          r.category,
                    "summary_short":     r.summary_short or "",
                    "confidence":        r.confidence or "",
                    "review_status":     r.review_status,
                    "language":          r.language,
                })
                n += 1
        log.info("CSV exported: %d rows → %s", n, output_path)
        return n

    @staticmethod
    def _orm_to_pydantic_stub() -> Record:
        """Internal — just to pull the CSV fieldnames from the Pydantic helper."""
        return Record(
            title="_",
            source_name="_",
            source_type="api",        # type: ignore[arg-type]
            country_or_region="Global",
            category="other",         # type: ignore[arg-type]
        )
