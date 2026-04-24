"""Centralized logger setup."""
from __future__ import annotations

import logging
import sys

from ..config.settings import settings


def setup_logging(level: str | None = None) -> None:
    """Configure root logger once. Idempotent."""
    lvl = (level or settings.log_level or "INFO").upper()
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=False,
    )


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
