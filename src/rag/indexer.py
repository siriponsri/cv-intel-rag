"""
Indexer — Record (from connectors) → chunks → embeddings → vector store.

This is the bridge between Layer 1 (SQL records) and the RAG vector store.
Call `index_records(records)` after ingestion to keep the vector store
up-to-date with newly fetched documents.

Metadata written with each chunk (for filtering + citation):
  • record_id:        UUID of the parent Record
  • source_name:      'PubMed' / 'openFDA' / etc.
  • source_type:      'api' / 'rss' / etc.
  • country_or_region: 'US' / 'Thailand' / ...
  • published_date:    ISO date string
  • url:               source URL
  • category:          Category enum value
  • title:             record title (up to 200 chars — for compact display)
  • external_id:       PMID / NCTID / ...
  • chunk_index:       position within the record
"""
from __future__ import annotations

import logging
from typing import Iterable, List

from ..db.models import RecordORM
from ..models.record import Record
from .chunker import Chunk, build_chunks
from .vectorstore import VectorStore, get_vectorstore

log = logging.getLogger(__name__)


class Indexer:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or get_vectorstore()

    # ─── from Pydantic Record (post-ingestion) ──────────────────────

    def index_records(self, records: Iterable[Record]) -> int:
        """Index a batch of Pydantic Records. Returns chunks added."""
        all_chunks: List[Chunk] = []
        for r in records:
            text = self._record_text(r)
            if not text:
                continue
            metadata = self._record_metadata(r)
            all_chunks.extend(build_chunks(r.record_id, text, metadata))

        if not all_chunks:
            return 0

        # Re-indexing a record should replace its old chunks, not dupe them
        record_ids = {r.record_id for r in records}
        for rid in record_ids:
            try:
                self.store.delete_by_record(rid)
            except Exception:                                                    # noqa: BLE001
                pass

        return self.store.upsert_chunks(all_chunks)

    # ─── from ORM (full rebuild / backfill) ─────────────────────────

    def index_orm_records(self, orm_records: Iterable[RecordORM], raw_texts: dict[str, str]) -> int:
        """
        Index records from the DB layer. Because RecordORM doesn't carry
        raw_text, caller must supply a {record_id: raw_text} dict (typically
        pulled from `raw_payloads`).
        """
        all_chunks: List[Chunk] = []
        record_ids_seen: set[str] = set()

        for orm in orm_records:
            text = raw_texts.get(orm.record_id)
            if not text:
                continue
            metadata = self._orm_metadata(orm)
            all_chunks.extend(build_chunks(orm.record_id, text, metadata))
            record_ids_seen.add(orm.record_id)

        if not all_chunks:
            return 0

        for rid in record_ids_seen:
            try:
                self.store.delete_by_record(rid)
            except Exception:                                                    # noqa: BLE001
                pass
        return self.store.upsert_chunks(all_chunks)

    # ─── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _record_text(r: Record) -> str:
        """Prefer raw_text (full abstract/description). Fall back to title+summary."""
        if r.raw_text and len(r.raw_text) > 100:
            return r.raw_text
        parts = [r.title or ""]
        if r.summary_short:
            parts.append(r.summary_short)
        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _record_metadata(r: Record) -> dict:
        return {
            "record_id":      r.record_id,
            "source_name":    r.source_name,
            "source_type":    r.source_type.value if hasattr(r.source_type, "value") else r.source_type,
            "country_or_region": r.country_or_region,
            "published_date": r.published_date.isoformat() if r.published_date else "",
            "url":            r.url or "",
            "category":       r.category.value if hasattr(r.category, "value") else r.category,
            "title":          (r.title or "")[:200],
            "external_id":    r.external_id or "",
        }

    @staticmethod
    def _orm_metadata(orm: RecordORM) -> dict:
        return {
            "record_id":      orm.record_id,
            "source_name":    orm.source_name,
            "source_type":    orm.source_type,
            "country_or_region": orm.country_or_region,
            "published_date": orm.published_date.isoformat() if orm.published_date else "",
            "url":            orm.url or "",
            "category":       orm.category,
            "title":          (orm.title or "")[:200],
            "external_id":    orm.external_id or "",
        }
