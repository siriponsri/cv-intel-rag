"""
End-to-end ingestion + auto-indexing.

connector → records → SQL DB → chunks → embeddings → ChromaDB

Usage:
    python scripts/run_ingestion.py                         # all connectors
    python scripts/run_ingestion.py --connector pubmed
    python scripts/run_ingestion.py --limit 10
    python scripts/run_ingestion.py --skip-index            # skip vector indexing
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.connectors.registry import CONNECTORS, get_connector                # noqa: E402
from src.db.repository import RecordRepository                               # noqa: E402
from src.db.session import SessionLocal                                      # noqa: E402
from src.rag.indexer import Indexer                                          # noqa: E402
from src.utils.logger import setup_logging                                   # noqa: E402


def run_one(connector_key: str, *, limit: int, skip_index: bool) -> dict:
    log = logging.getLogger(__name__)
    log.info("▶  Running connector: %s (limit=%d)", connector_key, limit)

    instance = get_connector(connector_key)
    run, result = instance.run(limit=limit)

    if result.errors:
        log.warning("connector errors: %s", result.errors)

    # 1. Persist Layer 1 records
    new = updated = 0
    with SessionLocal() as session:
        repo = RecordRepository(session)
        for record in result.records:
            checksum = instance.checksum(record.raw_text) if record.raw_text else None
            _, is_new = repo.upsert(record, raw_text=record.raw_text, checksum=checksum)
            if is_new: new += 1
            else:      updated += 1
        run.records_new     = new
        run.records_updated = updated
        repo.record_ingestion_run(run)
        session.commit()

    # 2. Index into ChromaDB
    chunks_indexed = 0
    if not skip_index and result.records:
        try:
            indexer = Indexer()
            chunks_indexed = indexer.index_records(result.records)
        except Exception as exc:                                                 # noqa: BLE001
            log.warning("indexing failed: %s", exc)

    log.info(
        "✓  %s: fetched=%d new=%d updated=%d chunks=%d status=%s",
        connector_key, run.records_fetched, new, updated, chunks_indexed, run.status,
    )
    return {
        "connector":      connector_key,
        "fetched":        run.records_fetched,
        "new":            new,
        "updated":        updated,
        "chunks_indexed": chunks_indexed,
        "status":         run.status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connector", help="Specific connector key", default=None)
    parser.add_argument("--limit", type=int, default=20, help="Max records per connector")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--skip-index", action="store_true", help="Skip vector indexing")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    keys = [args.connector] if args.connector else list(CONNECTORS.keys())
    log.info("Connectors to run: %s", keys)

    results = []
    for key in keys:
        try:
            results.append(run_one(key, limit=args.limit, skip_index=args.skip_index))
        except Exception as exc:                                                  # noqa: BLE001
            log.exception("connector %s crashed", key)
            if not args.skip_errors:
                raise
            results.append({"connector": key, "status": "failed", "error": str(exc)})
        time.sleep(1)

    log.info("=" * 70)
    log.info("Summary:")
    t_fetch = t_new = t_chunks = 0
    for r in results:
        log.info("  %-20s fetched=%-4d new=%-4d chunks=%-4d status=%s",
                 r["connector"], r.get("fetched", 0), r.get("new", 0),
                 r.get("chunks_indexed", 0), r.get("status"))
        t_fetch  += r.get("fetched", 0)
        t_new    += r.get("new", 0)
        t_chunks += r.get("chunks_indexed", 0)
    log.info("  TOTAL               fetched=%-4d new=%-4d chunks=%-4d", t_fetch, t_new, t_chunks)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
