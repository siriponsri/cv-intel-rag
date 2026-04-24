"""
Vector store — ChromaDB (embedded, file-backed).

ChromaDB is picked because:
  • Zero-setup (no Docker, runs in-process)
  • Local persistence to disk under settings.chroma_path
  • Metadata filtering for domain + source filtering
  • Good enough for <1M vectors (our scale for years)

Collection name = settings.domain_slug ('cardio-metabolic-renal')
to keep domains separate if we expand later.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional

from ..config.settings import settings
from .chunker import Chunk
from .embedder import get_embedder

log = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB persistent client wrapper."""

    def __init__(
        self,
        path: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ):
        self.path = Path(path or settings.chroma_path)
        self.collection_name = collection_name or settings.domain_slug
        self._client = None
        self._collection = None

    # ── lazy init ──────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
            except ImportError as exc:
                raise RuntimeError(
                    "chromadb not installed. Run: pip install chromadb"
                ) from exc
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            # inner-product metric is cheapest since we pre-normalise embeddings
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── CRUD ────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """
        Embed chunks and upsert into the collection.

        Idempotent on chunk_id — re-running with the same chunk_id updates
        the existing vector.
        """
        if not chunks:
            return 0

        embedder = get_embedder()
        texts = [c.text for c in chunks]
        vectors = embedder.encode(texts)

        collection = self._get_collection()
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[self._coerce_metadata(c.metadata) for c in chunks],
        )
        log.info("[VectorStore] upserted %d chunks into '%s'", len(chunks), self.collection_name)
        return len(chunks)

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 8,
        where: Optional[dict] = None,
    ) -> List[dict]:
        """
        Dense similarity search. Returns list of:
          {chunk_id, text, metadata, score}
        """
        embedder = get_embedder()
        query_vec = embedder.encode_single(query_text)

        collection = self._get_collection()
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns nested lists [query_index][result_index]
        out: List[dict] = []
        ids       = results.get("ids",       [[]])[0]
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            out.append({
                "chunk_id": chunk_id,
                "text":     docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                # cosine distance → similarity (higher = better)
                "score":    float(1.0 - distance),
            })
        return out

    def delete_by_record(self, record_id: str) -> None:
        """Delete all chunks belonging to a record (used when re-ingesting)."""
        collection = self._get_collection()
        collection.delete(where={"record_id": record_id})

    def count(self) -> int:
        return self._get_collection().count()

    def reset(self) -> None:
        """Nuke the collection. Use in tests only."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:                                                        # noqa: BLE001
            pass
        self._collection = None

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _coerce_metadata(meta: dict) -> dict:
        """
        ChromaDB only accepts str/int/float/bool metadata values.
        Coerce lists → pipe-separated strings; drop None.
        """
        out: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif isinstance(v, (list, tuple, set)):
                out[k] = "|".join(str(x) for x in v if x is not None)
            else:
                out[k] = str(v)
        return out


# ─── module-level singleton ─────────────────────────────────────────────

_default_store: Optional[VectorStore] = None


def get_vectorstore() -> VectorStore:
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store
