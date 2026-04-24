"""
Shared pytest fixtures. Ensures the DB schema exists before any test runs
and provides a temp wiki directory so tests don't pollute the real one.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "pharma_intel.db"


@pytest.fixture(autouse=True, scope="session")
def _ensure_schema() -> None:
    """Fresh DB before the test session."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)

    # always start clean
    if DB_PATH.exists():
        DB_PATH.unlink()

    # Force reload settings so DATABASE_URL env matches test expectations
    from src.config.settings import get_settings
    get_settings.cache_clear()                              # type: ignore[attr-defined]

    # Rebuild engine against the fresh sqlite file
    import importlib
    import src.db.session as session_mod
    importlib.reload(session_mod)

    # Execute schema.sql directly
    from sqlalchemy import text
    engine = session_mod.engine
    schema_sql = (PROJECT_ROOT / "src" / "db" / "schema.sql").read_text(encoding="utf-8")

    statements = []
    for raw in schema_sql.split(";"):
        lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("--")]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            statements.append(cleaned)

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


@pytest.fixture()
def tmp_wiki(tmp_path: Path):
    """Give each test its own wiki root, restored after the test."""
    yield tmp_path / "wiki"
