# ── Dockerfile for Hugging Face Spaces (Docker SDK) ──────────────────────
#
# HF Spaces conventions:
#   • Dockerfile MUST be at repo root
#   • App MUST listen on port 7860 (declared in README.md frontmatter)
#   • Container MUST run as non-root user (UID 1000 is HF default)
#   • Secrets (TYPHOON_API_KEY) are injected as env vars from Space settings
#   • /data is ephemeral on free tier — we bake the pre-built index into the
#     image. For larger corpora, use an HF Dataset and clone at startup.
#
# Build locally:
#   docker build -t cv-intel-rag .
#   docker run -p 7860:7860 -e TYPHOON_API_KEY=sk-... cv-intel-rag
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── System deps ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libxml2-dev libxslt1-dev curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (HF requirement) ──────────────────────────────────────
RUN useradd -m -u 1000 user
WORKDIR /app

# ── Python deps (separate layer for caching) ────────────────────────────
COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download embedder to /app so it's in the image, not downloaded at
# startup (saves ~30s of cold-start time).
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('BAAI/bge-m3', cache_folder='/app/.hf_cache')" \
    || echo "embedder pre-download skipped (will download at startup)"

# ── App code ────────────────────────────────────────────────────────────
COPY --chown=user:user src/     ./src/
COPY --chown=user:user scripts/ ./scripts/

# ── Pre-built index (built in Colab, committed via git-lfs) ─────────────
# If data/ is empty at build time, the Space will start with an empty
# index and answer only from Typhoon's general knowledge. Run
# scripts/run_ingestion.py at startup to populate, or commit the
# pre-built data/ from your Colab notebook.
COPY --chown=user:user data/ ./data/

# ── Runtime env ─────────────────────────────────────────────────────────
USER user
ENV HF_HOME=/app/.hf_cache \
    TRANSFORMERS_CACHE=/app/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache \
    PORT=7860 \
    DEFAULT_LLM_PROVIDER=typhoon_api \
    TYPHOON_BASE_URL=https://api.opentyphoon.ai/v1 \
    TYPHOON_MODEL=typhoon-v2.5-30b-a3b-instruct \
    EMBED_DEVICE=cpu \
    EMBED_BATCH_SIZE=8 \
    DATABASE_URL=sqlite:///./data/cv_intel.db \
    CHROMA_PATH=./data/chroma
# TYPHOON_API_KEY is injected from HF Space Secrets at runtime.

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
