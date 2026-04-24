"""
Hybrid retriever.

Combines:
  • Dense search (ChromaDB + BGE-M3 embeddings) — good for semantic match
  • Sparse search (BM25)                         — good for exact-token match
                                                    (drug names, NCT IDs,
                                                    MeSH terms, Thai words)

Mixing: final_score = alpha * dense + (1 - alpha) * bm25
Default alpha = 0.6 (dense-heavy but BM25 contributes)

BM25 index rebuilds are cheap on <100k chunks, so we rebuild in-memory
at retrieve time. If the corpus grows large, cache the BM25 tokens per
chunk in ChromaDB metadata.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from ..config.settings import settings
from .vectorstore import VectorStore, get_vectorstore

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with all the info the agent needs."""
    chunk_id:    str
    text:        str
    score:       float                  # 0-1, higher = more relevant
    record_id:   str
    metadata:    dict


# ─── Tokenisation for BM25 ──────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Simple tokeniser that works for English + Thai characters."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ─── HybridRetriever ────────────────────────────────────────────────────

class HybridRetriever:
    """
    Retriever that runs dense and sparse searches and merges scores.

    For sparse search we fetch a larger pool from ChromaDB (top_k * 3)
    then rescore with BM25 to avoid missing BM25-only hits that dense
    retrieval ranked outside the window.
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ):
        self.store = store or get_vectorstore()
        self.top_k = top_k or settings.retrieve_top_k
        self.alpha = alpha if alpha is not None else settings.retrieve_alpha

    def retrieve(
        self,
        query: str,
        *,
        where: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        k = top_k or self.top_k
        pool_size = max(k * 3, 20)

        # 1. Dense search — pull a larger pool for re-ranking
        dense_hits = self.store.query(query, top_k=pool_size, where=where)
        if not dense_hits:
            return []

        # 2. BM25 over the dense pool (small N, so rebuilt per query is fine)
        bm25_scores = self._bm25_scores(query, dense_hits)

        # 3. Merge — normalise both to [0, 1] then combine
        merged: List[RetrievedChunk] = []
        max_dense = max((h["score"] for h in dense_hits), default=1.0) or 1.0
        max_bm25  = max(bm25_scores.values(), default=1.0) or 1.0

        for hit in dense_hits:
            dense_norm = hit["score"] / max_dense if max_dense > 0 else 0.0
            bm25_norm  = bm25_scores.get(hit["chunk_id"], 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            merged_score = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm
            merged.append(RetrievedChunk(
                chunk_id=hit["chunk_id"],
                text=hit["text"],
                score=merged_score,
                record_id=hit["metadata"].get("record_id", ""),
                metadata=hit["metadata"],
            ))

        merged.sort(key=lambda c: c.score, reverse=True)
        return merged[:k]

    # ── BM25 ────────────────────────────────────────────────────────

    def _bm25_scores(self, query: str, hits: List[dict]) -> dict[str, float]:
        """Return {chunk_id: bm25_score} for the given pool."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            log.warning("rank_bm25 not installed — falling back to dense-only")
            return {}

        corpus_tokens = [_tokenize(h["text"]) for h in hits]
        if not any(corpus_tokens):
            return {}
        query_tokens = _tokenize(query)
        if not query_tokens:
            return {}

        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
        return {hits[i]["chunk_id"]: float(scores[i]) for i in range(len(hits))}


# ─── module singleton ───────────────────────────────────────────────────

_default_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = HybridRetriever()
    return _default_retriever
