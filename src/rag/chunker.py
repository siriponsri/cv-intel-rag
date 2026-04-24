"""
Document chunker.

Splits long raw_text into overlapping chunks so each chunk fits a sensible
window for embedding AND leaves enough room for retrieval context.

Strategy:
  • Token-based boundaries (tiktoken cl100k_base ≈ what most LLMs use)
  • Paragraph-aware where possible (split on \\n\\n, fall back to sentences)
  • Each chunk carries its parent record_id + chunk_index for traceability
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

import tiktoken

from ..config.settings import settings

log = logging.getLogger(__name__)


# ─── Token counter (lazy-loaded, with char-based fallback) ──────────────

_ENCODING = None


def _get_encoding():
    """
    Lazy-init tiktoken. If tiktoken can't download its BPE file (e.g. offline
    or corporate proxy), fall back to a simple character-based estimator
    so the pipeline still runs.
    """
    global _ENCODING
    if _ENCODING is not None:
        return _ENCODING
    try:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:                                                     # noqa: BLE001
        log.warning("tiktoken unavailable (%s) — using char-based token estimator", exc)
        _ENCODING = _CharEstimator()
    return _ENCODING


class _CharEstimator:
    """Fallback 'encoding': 1 token ≈ 4 chars (English) / 2 chars (Thai)."""
    @staticmethod
    def _char_weight(ch: str) -> float:
        return 0.5 if "\u0e00" <= ch <= "\u0e7f" else 0.25      # Thai ~2 chars/tok, English ~4

    def encode(self, text: str) -> list[int]:
        # Produce a deterministic pseudo-token list whose length matches the estimate
        return [ord(c) for c in text[:: max(1, int(1.0 / max(self._char_weight(" "), 0.01)))]] \
            or list(range(max(1, int(sum(self._char_weight(c) for c in text)))))

    def decode(self, tokens: list[int]) -> str:
        try:
            return "".join(chr(t) for t in tokens if 0 < t < 0x110000)
        except Exception:                                                        # noqa: BLE001
            return ""


@dataclass
class Chunk:
    """A single chunk of text plus everything needed to retrieve it."""
    chunk_id:     str
    record_id:    str
    chunk_index:  int
    text:         str
    token_count:  int
    # Metadata mirrored from the parent Record so ChromaDB can filter on it
    metadata:     dict = field(default_factory=dict)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _get_encoding()
    try:
        return len(enc.encode(text))
    except Exception:                                                            # noqa: BLE001
        return max(1, len(text) // 4)


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[str]:
    """
    Split `text` into chunks of ~chunk_size tokens with `overlap` token overlap.

    Paragraph-first: prefer splitting on \\n\\n boundaries so we keep
    coherent units. If a paragraph is too long, fall back to sentences.
    """
    chunk_size = chunk_size or settings.chunk_size_tokens
    overlap    = overlap    or settings.chunk_overlap_tokens

    if not text:
        return []

    # Short-circuit: whole text fits in one chunk
    if count_tokens(text) <= chunk_size:
        return [text]

    # Split into paragraphs, then into sentences if still too big
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # Break oversized paragraphs into sentence-level pieces
    units: List[str] = []
    for p in paragraphs:
        if count_tokens(p) <= chunk_size:
            units.append(p)
        else:
            sentences = re.split(r"(?<=[.!?。ฯ])\s+", p)
            units.extend(s for s in sentences if s.strip())

    # Greedy packing: pack units into chunks up to chunk_size tokens
    encoding = _get_encoding()
    chunks: List[str] = []
    current_tokens: List[int] = []
    current_text_parts: List[str] = []
    for unit in units:
        unit_tokens = encoding.encode(unit)
        if len(current_tokens) + len(unit_tokens) > chunk_size and current_tokens:
            # flush
            chunks.append(" ".join(current_text_parts).strip())
            # start new chunk with overlap from the tail of previous
            if overlap > 0 and len(current_tokens) > overlap:
                tail = current_tokens[-overlap:]
                current_tokens = list(tail)
                tail_text = encoding.decode(tail)
                current_text_parts = [tail_text] if tail_text else []
            else:
                current_tokens = []
                current_text_parts = []
        current_tokens.extend(unit_tokens)
        current_text_parts.append(unit)

    if current_text_parts:
        chunks.append(" ".join(current_text_parts).strip())

    return [c for c in chunks if c]


def build_chunks(
    record_id: str,
    text: str,
    metadata: dict | None = None,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[Chunk]:
    """Build Chunk objects from a single Record's text."""
    text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    metadata = metadata or {}
    result: List[Chunk] = []
    for idx, chunk_text_ in enumerate(text_chunks):
        chunk_id = f"{record_id}::chunk{idx:03d}"
        result.append(Chunk(
            chunk_id=chunk_id,
            record_id=record_id,
            chunk_index=idx,
            text=chunk_text_,
            token_count=count_tokens(chunk_text_),
            metadata={**metadata, "chunk_index": idx, "record_id": record_id},
        ))
    return result
