"""
Embedder — BGE-M3 via sentence-transformers.

BGE-M3 is chosen because:
  • Supports 100+ languages incl. Thai natively
  • 8192 token context (vs. 512 for most older models)
  • 1024-dim dense vectors
  • Apache 2.0 license, Hugging Face hosted

First load takes 30-60s to download model weights (~2.3 GB).
Subsequent runs reuse the HF cache at ~/.cache/huggingface/.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config.settings import settings

log = logging.getLogger(__name__)


class Embedder:
    """
    Thin wrapper around sentence-transformers.

    We lazy-load the model so test suites and CLI --help don't pay the
    model-load cost. Calling `encode()` triggers the download/load once.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        self.model_name = model_name or settings.embed_model_name
        self.device     = device or settings.embed_device
        self.batch_size = batch_size or settings.embed_batch_size
        self._model = None          # lazy

    # ── Public API ──────────────────────────────────────────────────

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        """
        Return a dense vector for each input text.

        Normalised to unit length so cosine similarity == dot product —
        this lets ChromaDB use the faster inner-product metric.
        """
        texts = list(texts)
        if not texts:
            return []
        model = self._get_model()
        log.info("[Embedder] encoding %d texts on %s", len(texts), self.device)
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]

    # ── internals ───────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            log.info("[Embedder] loading %s on %s (first run downloads ~2.3 GB)",
                     self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


# ─── module-level singleton (shared across pipeline) ────────────────────

_default_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder
