"""
Pydantic models for Layer 1 structured records.

These mirror the `records` table in db/schema.sql and enforce the unified
schema from the master plan:  Executive Summary → Unified Data Schema.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


# ─── Enumerations (match SQL CHECK constraints) ───────────────────────────

class SourceType(str, Enum):
    API           = "api"
    BULK_DATASET  = "bulk_dataset"
    RSS           = "rss"
    WEB_SCRAPE    = "web_scrape"


class Category(str, Enum):
    DRUG_APPROVAL   = "drug_approval"
    SAFETY          = "safety"
    GUIDELINE       = "guideline"
    PATENT          = "patent"
    RESEARCH        = "research"
    CLINICAL_TRIAL  = "clinical_trial"
    REGULATOR       = "regulator"
    OTHER           = "other"


class Confidence(str, Enum):
    HIGH    = "high"
    MEDIUM  = "medium"
    LOW     = "low"


class ReviewStatus(str, Enum):
    AUTO      = "auto"
    REVIEWED  = "reviewed"
    FLAGGED   = "flagged"


# ─── Core Layer 1 record ──────────────────────────────────────────────────

class Record(BaseModel):
    """
    Unified Layer 1 record — one row in the `records` table, plus tags.

    Tags are carried on the model for convenience; the persistence layer
    splits them into `tags` + `record_tags` junction rows.
    """

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)

    record_id:          str = Field(default_factory=lambda: str(uuid.uuid4()))
    title:              str = Field(..., min_length=1, max_length=2000)
    source_name:        str = Field(..., min_length=1, max_length=200)
    source_type:        SourceType
    layer:              Literal[1] = 1
    country_or_region:  str = Field(..., min_length=1, max_length=100)

    published_date:     Optional[date] = None
    ingested_at:        datetime = Field(default_factory=datetime.utcnow)
    url:                Optional[str] = None

    topic_tags:         List[str] = Field(default_factory=list)
    entity_tags:        List[str] = Field(default_factory=list)

    category:           Category
    summary_short:      Optional[str] = Field(default=None, max_length=2000)

    # we deliberately keep raw_text in the ORM layer (raw_payloads table),
    # not on the wire-level Pydantic model, but expose it here for ingest
    raw_text:           Optional[str] = None

    confidence:         Optional[Confidence] = None
    review_status:      ReviewStatus = ReviewStatus.AUTO
    exportable_to_csv:  bool = True

    # Source-specific identifiers (keeps dedup robust across connectors)
    external_id:        Optional[str] = None          # PMID, NCTID, openFDA safety_report_id, ...
    external_version:   Optional[str] = None          # revision / version where applicable

    language:           str = "en"

    # ── validators ──────────────────────────────────────────────────────

    @field_validator("topic_tags", "entity_tags", mode="before")
    @classmethod
    def _clean_tags(cls, v):
        """Deduplicate + strip empty entries; keep order stable."""
        if v is None:
            return []
        seen, out = set(), []
        for t in v:
            t = (t or "").strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out

    @field_validator("country_or_region")
    @classmethod
    def _normalise_region(cls, v: str) -> str:
        v = v.strip()
        aliases = {
            "TH": "Thailand", "thailand": "Thailand", "ไทย": "Thailand",
            "US": "US", "USA": "US", "United States": "US",
            "EU": "EU", "Europe": "EU",
            "global": "Global", "world": "Global", "WHO": "Global",
        }
        return aliases.get(v, aliases.get(v.lower(), v))


# ─── Ingestion run audit record ───────────────────────────────────────────

class IngestionRun(BaseModel):
    """One row in the `ingestion_runs` table — audit of a connector execution."""

    model_config = ConfigDict(use_enum_values=True)

    run_id:          str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_name:     str
    started_at:      datetime = Field(default_factory=datetime.utcnow)
    finished_at:     Optional[datetime] = None
    status:          Literal["running", "success", "partial", "failed"] = "running"
    records_fetched: int = 0
    records_new:     int = 0
    records_updated: int = 0
    error_message:   Optional[str] = None
    notes:           Optional[str] = None


# ─── Helper: CSV-safe dict ────────────────────────────────────────────────

def record_to_csv_row(r: Record) -> dict:
    """Flatten a Record into a CSV-compatible dict (list fields → pipe-delimited)."""
    return {
        "record_id":         r.record_id,
        "title":             r.title,
        "source_name":       r.source_name,
        "source_type":       r.source_type.value if isinstance(r.source_type, SourceType) else r.source_type,
        "country_or_region": r.country_or_region,
        "published_date":    r.published_date.isoformat() if r.published_date else "",
        "ingested_at":       r.ingested_at.isoformat(),
        "url":               r.url or "",
        "topic_tags":        "|".join(r.topic_tags),
        "entity_tags":       "|".join(r.entity_tags),
        "category":          r.category.value if isinstance(r.category, Category) else r.category,
        "summary_short":     r.summary_short or "",
        "confidence":        (r.confidence.value if isinstance(r.confidence, Confidence)
                              else r.confidence) or "",
        "review_status":     r.review_status.value if isinstance(r.review_status, ReviewStatus) else r.review_status,
        "language":          r.language,
    }
