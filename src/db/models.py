"""
SQLAlchemy ORM models — mirror `src/db/schema.sql`.

These are the persistence-layer objects. The wire-level schema is
`src/models/record.py:Record` (Pydantic); use RecordRepository to convert.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, Date, DateTime, ForeignKey, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ─── records ──────────────────────────────────────────────────────────────

class RecordORM(Base):
    __tablename__ = "records"

    record_id:          Mapped[str]              = mapped_column(String, primary_key=True)
    title:              Mapped[str]              = mapped_column(Text, nullable=False)
    source_name:        Mapped[str]              = mapped_column(String, nullable=False, index=True)
    source_type:        Mapped[str]              = mapped_column(String, nullable=False)
    layer:              Mapped[int]              = mapped_column(Integer, nullable=False, default=1)
    country_or_region:  Mapped[str]              = mapped_column(String, nullable=False, index=True)
    published_date:     Mapped[datetime | None]  = mapped_column(Date, nullable=True, index=True)
    ingested_at:        Mapped[datetime]         = mapped_column(DateTime, nullable=False, index=True)
    url:                Mapped[str | None]       = mapped_column(Text, nullable=True)
    category:           Mapped[str]              = mapped_column(String, nullable=False, index=True)
    summary_short:      Mapped[str | None]       = mapped_column(Text, nullable=True)
    confidence:         Mapped[str | None]       = mapped_column(String, nullable=True)
    review_status:      Mapped[str]              = mapped_column(String, nullable=False, default="auto", index=True)
    exportable_to_csv:  Mapped[bool]             = mapped_column(Boolean, nullable=False, default=True)
    external_id:        Mapped[str | None]       = mapped_column(String, nullable=True)
    external_version:   Mapped[str | None]       = mapped_column(String, nullable=True)
    language:           Mapped[str]              = mapped_column(String, default="en")
    created_at:         Mapped[datetime]         = mapped_column(DateTime, default=datetime.utcnow)
    updated_at:         Mapped[datetime]         = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("source_name", "external_id", "external_version", name="uq_records_source_ext"),
    )

    payloads = relationship("RawPayloadORM", back_populates="record", cascade="all, delete-orphan")
    tags     = relationship("RecordTagORM", back_populates="record", cascade="all, delete-orphan")


# ─── raw_payloads ─────────────────────────────────────────────────────────

class RawPayloadORM(Base):
    __tablename__ = "raw_payloads"

    payload_id:   Mapped[str]      = mapped_column(String, primary_key=True)
    record_id:    Mapped[str]      = mapped_column(String, ForeignKey("records.record_id", ondelete="CASCADE"), index=True)
    content_type: Mapped[str]      = mapped_column(String, nullable=False)
    raw_text:     Mapped[str | None] = mapped_column(Text, nullable=True)
    blob_path:    Mapped[str | None] = mapped_column(String, nullable=True)
    fetched_at:   Mapped[datetime] = mapped_column(DateTime, nullable=False)
    checksum:     Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (UniqueConstraint("record_id", "checksum", name="uq_payload_record_checksum"),)

    record = relationship("RecordORM", back_populates="payloads")


# ─── tags + record_tags ───────────────────────────────────────────────────

class TagORM(Base):
    __tablename__ = "tags"

    tag_id:     Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    tag_type:   Mapped[str]  = mapped_column(String, nullable=False)
    tag_value:  Mapped[str]  = mapped_column(String, nullable=False)
    normalized: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    __table_args__ = (UniqueConstraint("tag_type", "tag_value", name="uq_tag_type_value"),)


class RecordTagORM(Base):
    __tablename__ = "record_tags"

    record_id: Mapped[str] = mapped_column(String, ForeignKey("records.record_id", ondelete="CASCADE"), primary_key=True)
    tag_id:    Mapped[int] = mapped_column(Integer, ForeignKey("tags.tag_id", ondelete="CASCADE"), primary_key=True, index=True)

    record = relationship("RecordORM", back_populates="tags")
    tag    = relationship("TagORM")


# ─── ingestion_runs ───────────────────────────────────────────────────────

class IngestionRunORM(Base):
    __tablename__ = "ingestion_runs"

    run_id:          Mapped[str]              = mapped_column(String, primary_key=True)
    source_name:     Mapped[str]              = mapped_column(String, nullable=False, index=True)
    started_at:      Mapped[datetime]         = mapped_column(DateTime, nullable=False, index=True)
    finished_at:     Mapped[datetime | None]  = mapped_column(DateTime, nullable=True)
    status:          Mapped[str]              = mapped_column(String, nullable=False)
    records_fetched: Mapped[int]              = mapped_column(Integer, default=0)
    records_new:     Mapped[int]              = mapped_column(Integer, default=0)
    records_updated: Mapped[int]              = mapped_column(Integer, default=0)
    error_message:   Mapped[str | None]       = mapped_column(Text, nullable=True)
    notes:           Mapped[str | None]       = mapped_column(Text, nullable=True)


# ─── wiki_refs ────────────────────────────────────────────────────────────

class WikiRefORM(Base):
    __tablename__ = "wiki_refs"

    ref_id:     Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id:  Mapped[str | None] = mapped_column(String, ForeignKey("records.record_id", ondelete="SET NULL"), nullable=True, index=True)
    wiki_path:  Mapped[str] = mapped_column(String, nullable=False, index=True)
    wiki_layer: Mapped[int] = mapped_column(Integer, nullable=False)
    relation:   Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ─── review_queue ─────────────────────────────────────────────────────────

class ReviewQueueORM(Base):
    __tablename__ = "review_queue"

    queue_id:       Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id:      Mapped[str | None] = mapped_column(String, ForeignKey("records.record_id", ondelete="CASCADE"), nullable=True)
    reason:         Mapped[str] = mapped_column(Text, nullable=False)
    priority:       Mapped[int] = mapped_column(Integer, default=5)
    assigned_to:    Mapped[str | None] = mapped_column(String, nullable=True)
    status:         Mapped[str] = mapped_column(String, nullable=False, default="pending", index=True)
    created_at:     Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    resolved_at:    Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
