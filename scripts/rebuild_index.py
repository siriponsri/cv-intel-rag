"""
Rebuild ChromaDB index from existing SQL records.

Use after:
  • Changing the embedding model (CHUNK_SIZE, EMBED_MODEL_NAME, etc.)
  • ChromaDB gets corrupted / deleted
  • You want to reindex everything with updated chunking

Usage:
    python scripts/rebuild_index.py                  # last 365 days
    python scripts/rebuild_index.py --days 90
    python scripts/rebuild_index.py --reset          # wipe and rebuild
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select                                                # noqa: E402

from src.db.models import RawPayloadORM, RecordORM                           # noqa: E402
from src.db.session import SessionLocal                                      # noqa: E402
from src.rag.indexer import Indexer                                          # noqa: E402
from src.rag.vectorstore import get_vectorstore                              # noqa: E402
from src.utils.logger import setup_logging                                   # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--reset", action="store_true",
                        help="Wipe the vector store before rebuilding")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    if args.reset:
        log.warning("--reset: wiping ChromaDB collection")
        get_vectorstore().reset()

    since = datetime.utcnow() - timedelta(days=args.days)

    with SessionLocal() as session:
        records = list(session.execute(
            select(RecordORM).where(RecordORM.ingested_at >= since)
        ).scalars().all())

        raw_texts: dict[str, str] = {}
        for r in records:
            payload = session.execute(
                select(RawPayloadORM).where(RawPayloadORM.record_id == r.record_id).limit(1)
            ).scalar_one_or_none()
            if payload and payload.raw_text:
                raw_texts[r.record_id] = payload.raw_text
            elif r.summary_short:
                raw_texts[r.record_id] = f"{r.title}\n\n{r.summary_short}"

    log.info("Found %d records to reindex (%d with text)", len(records), len(raw_texts))

    indexer = Indexer()
    n = indexer.index_orm_records(records, raw_texts)
    log.info("✓ Indexed %d chunks into vector store", n)


if __name__ == "__main__":
    main()
