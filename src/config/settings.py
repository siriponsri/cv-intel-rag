"""
Application settings, loaded from .env via pydantic-settings.

LLM provider strategy:
  default = typhoon_api  — uses https://api.opentyphoon.ai/v1 (free)
  alt     = vllm_local   — uses http://localhost:8000/v1 (self-hosted)
  alt     = openai_compat — any OpenAI-compatible endpoint
  alt     = null         — offline fallback (no LLM, returns canned message)

All three real providers use the OpenAI SDK underneath (Typhoon is
OpenAI-compatible). Switch by changing DEFAULT_LLM_PROVIDER and
the matching base_url.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── App ─────────────────────────────────────────────────────────
    app_env:    Literal["development", "staging", "production"] = "development"
    app_name:   str = "cv-intel-rag"
    log_level:  str = "INFO"

    # ─── Storage ─────────────────────────────────────────────────────
    database_url:     str = f"sqlite:///{PROJECT_ROOT / 'data' / 'cv_intel.db'}"
    chroma_path:      Path = PROJECT_ROOT / "data" / "chroma"
    raw_payload_root: Path = PROJECT_ROOT / "data" / "raw"

    # ─── LLM provider ────────────────────────────────────────────────
    default_llm_provider: Literal["typhoon_api", "vllm_local", "openai_compat", "null"] = "typhoon_api"

    # Typhoon (SCB 10X) — default free tier
    typhoon_api_key:   Optional[str] = None
    typhoon_base_url:  str = "https://api.opentyphoon.ai/v1"
    typhoon_model:     str = "typhoon-v2.5-30b-a3b-instruct"

    # vLLM self-host
    vllm_base_url:     str = "http://localhost:8000/v1"
    vllm_model:        str = "scb10x/typhoon-v2.5-30b-a3b-instruct"
    vllm_api_key:      str = "not-needed"   # vLLM accepts any string

    # Generic OpenAI-compatible (fallback for OpenThai, Pathumma, etc.)
    openai_compat_api_key:  Optional[str] = None
    openai_compat_base_url: str = ""
    openai_compat_model:    str = ""

    # Generation defaults
    llm_temperature: float = 0.3               # low for factual RAG
    llm_max_tokens:  int = 1024
    llm_timeout_sec: int = 60

    # ─── Embedding ───────────────────────────────────────────────────
    # BGE-M3 is best for Thai+English (100+ langs, 8192 ctx, 1024 dim)
    embed_model_name:  str = "BAAI/bge-m3"
    embed_device:      Literal["cpu", "cuda", "mps"] = "cpu"    # override via env on GPU boxes
    embed_batch_size:  int = 16
    embed_dim:         int = 1024

    # ─── RAG retrieval ───────────────────────────────────────────────
    chunk_size_tokens:   int = 500
    chunk_overlap_tokens: int = 80
    retrieve_top_k:      int = 8
    retrieve_alpha:      float = 0.6           # dense vs BM25 mixing weight
    rerank_enabled:      bool = False          # phase-2 feature

    # ─── Domain scope ────────────────────────────────────────────────
    # CV + Diabetes + CKD (cardio-metabolic-renal axis)
    domain_slug:         str = "cardio-metabolic-renal"

    # ─── Source-specific credentials ─────────────────────────────────
    ncbi_api_key:  Optional[str] = None
    ncbi_email:    Optional[str] = None
    openfda_api_key: Optional[str] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
