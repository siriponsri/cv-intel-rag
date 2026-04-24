"""
Initialize the SQLite database by executing `src/db/schema.sql`.

Usage:
    python scripts/init_db.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text                                                  # noqa: E402
from src.config.settings import settings                                     # noqa: E402
from src.db.session import engine                                            # noqa: E402

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


SCHEMA_PATH = PROJECT_ROOT / "src" / "db" / "schema.sql"


def split_statements(sql_text: str) -> list[str]:
    out: list[str] = []
    for raw in sql_text.split(";"):
        lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("--")]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            out.append(cleaned)
    return out


def init_db() -> None:
    if not SCHEMA_PATH.exists():
        log.error("Schema file not found: %s", SCHEMA_PATH)
        sys.exit(1)
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = split_statements(sql)

    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "chroma").mkdir(parents=True, exist_ok=True)

    log.info("Initialising DB at %s", engine.url)
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    log.info("✓ DB initialised with %d statements", len(statements))


if __name__ == "__main__":
    init_db()
