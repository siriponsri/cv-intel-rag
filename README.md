---
title: CV Intel RAG
emoji: 🫀
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Cardiovascular + Diabetes + CKD regulatory intelligence RAG agent (Thai + English)
---

# CV Intel RAG

**A production-grade Retrieval-Augmented Generation (RAG) system for cardiovascular, diabetes, and chronic kidney disease (CV + DM + CKD) regulatory and research intelligence.**

CV Intel RAG continuously ingests biomedical literature, clinical trial registrations, drug safety communications, and clinical guideline updates from authoritative public sources, indexes them with state-of-the-art multilingual embeddings, and exposes an AI-powered conversational interface capable of answering complex clinical and regulatory questions in both Thai and English with traceable, cited references.

[![CI](https://github.com/siriponsri/cv-intel-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/siriponsri/cv-intel-rag/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/siriponsri/cv-intel-rag)

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Data Sources](#data-sources)
5. [Technology Stack](#technology-stack)
6. [Repository Structure](#repository-structure)
7. [API Reference](#api-reference)
8. [Configuration Reference](#configuration-reference)
9. [Installation and Local Development](#installation-and-local-development)
10. [Running on Google Colab](#running-on-google-colab)
11. [Deployment to Hugging Face Spaces](#deployment-to-hugging-face-spaces)
12. [Testing](#testing)
13. [Development Guidelines](#development-guidelines)
14. [Security Considerations](#security-considerations)
15. [Documentation](#documentation)
16. [License](#license)
17. [Credits and Acknowledgements](#credits-and-acknowledgements)

---

## Overview

CV Intel RAG addresses a critical need in clinical and regulatory environments: the volume of relevant information published daily across PubMed, ClinicalTrials.gov, FDA MedWatch, EMA, and cardiology society journals far exceeds the capacity of any individual to manually review. This system automates the entire pipeline — from source ingestion to natural-language question answering — with full source attribution enabling clinicians, researchers, and regulatory affairs professionals to verify every claim.

The system targets the **cardiometabolic-renal axis**: conditions and drug classes that overlap cardiovascular disease, type 2 diabetes, and chronic kidney disease, including SGLT2 inhibitors, GLP-1 receptor agonists, ACE inhibitors, ARBs, statins, anticoagulants, and related pharmacological interventions.

---

## Key Features

- **Automated multi-source ingestion** from 7 connectors (PubMed, ClinicalTrials.gov, openFDA, plus 4 RSS feeds) with scheduled or on-demand execution
- **Domain-scoped filtering** using MeSH term queries, structured condition lists, pharmacological class keywords, and a compiled regular expression (`CV_TERM_REGEX`) that correctly handles Thai characters, English abbreviations, and biomedical prefixes
- **Hybrid dense + sparse retrieval** combining ChromaDB (BGE-M3 dense vectors, 1024 dimensions) with BM25 (rank-bm25), merged via a configurable α parameter (default 0.6)
- **Multilingual support** in Thai and English via the BGE-M3 model (100+ languages, 8192-token context window)
- **Traceable citations**: the LLM is instructed to emit numbered source markers `[S1]`, `[S2]`, etc., which the frontend resolves to clickable links to original records
- **Streaming responses** via Server-Sent Events (SSE) for real-time token delivery
- **Swappable LLM provider**: Typhoon API (default), vLLM self-hosted, any OpenAI-compatible endpoint, or a `null` offline fallback
- **Dashboard** for browsing, filtering, and inspecting ingested records by source and category
- **Zero-dependency frontend**: both chat and dashboard UIs are single-file vanilla JavaScript — no build toolchain required
- **Docker + HF Spaces ready**: single `Dockerfile`, non-root user, port 7860, with a GitHub Actions workflow for automatic synchronisation to Hugging Face

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                        │
│                                                                     │
│  PubMed  ──►  PubMedConnector   ─┐                                  │
│  CT.gov  ──►  CTGovConnector    ─┤                                  │
│  openFDA ──►  openFDAConnector  ─┼──► RecordRepository (SQLite)    │
│  RSS×4   ──►  RSSConnector×4   ─┘          ↓                       │
│                  (CV_TERM_REGEX filter)   Record (Pydantic)          │
└──────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         INDEXING LAYER                               │
│                                                                     │
│  Chunker (tiktoken 500 tok / 80 overlap, _CharEstimator fallback)   │
│       ↓                                                             │
│  BGE-M3 Embedder (BAAI/bge-m3, 1024-dim, batch=16)                 │
│       ↓                                                             │
│  ChromaDB VectorStore (_coerce_metadata: lists → pipe-joined str)   │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL LAYER                               │
│                                                                     │
│  HybridRetriever: final_score = α·dense_score + (1−α)·bm25_score   │
│  top-k=8, α=0.6 (configurable via RETRIEVE_ALPHA env var)           │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM GENERATION LAYER                            │
│                                                                     │
│  RAGAgent.answer() / answer_stream()                                │
│  Typhoon v2.5-30b-a3b-instruct (or vLLM / OpenAI-compat / null)    │
│  Structured prompt with [S#] citation instruction                   │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        API + UI LAYER                                │
│                                                                     │
│  FastAPI v0.2.0                                                     │
│  POST /chat         — JSON response (blocking)                      │
│  POST /chat/stream  — SSE streaming tokens                          │
│  POST /search       — raw retrieval (no LLM)                        │
│  POST /ingestion/run — on-demand connector invocation               │
│  GET  /stats, /records, /health                                     │
│                                                                     │
│  chat.html     — ChatGPT-style SSE interface with citation links    │
│  dashboard.html — records browser with source/category filtering    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

| Source | Connector | Data Type | Query Scope | Rate Limit (no key) |
|--------|-----------|-----------|-------------|---------------------|
| [PubMed / NCBI E-utilities](https://pubmed.ncbi.nlm.nih.gov/) | `pubmed.py` | Peer-reviewed journal articles | 13 MeSH term queries covering CV, DM, CKD, SGLT2, GLP-1, ARBs | 3 req/sec |
| [ClinicalTrials.gov](https://clinicaltrials.gov/) | `clinical_trials.py` | Registered clinical trials (phase 1–4) | 7 condition terms | Unlimited |
| [openFDA](https://open.fda.gov/) | `openfda.py` | Drug adverse events, recalls, enforcement | 10 pharmacological class keywords | 240 req/min |
| [FDA MedWatch RSS](https://www.fda.gov/safety/medwatch-fda-safety-information-and-adverse-event-reporting-program) | `rss.py` | Drug safety alerts, market withdrawals | CV_TERM_REGEX post-filter | Unlimited |
| [EMA What's New RSS](https://www.ema.europa.eu/) | `rss.py` | European Medicines Agency regulatory updates | CV_TERM_REGEX post-filter | Unlimited |
| [ESC Guidelines RSS](https://www.escardio.org/) | `rss.py` | European Society of Cardiology guidelines | CV_TERM_REGEX post-filter | Unlimited |
| [AHA Circulation RSS](https://www.ahajournals.org/) | `rss.py` | American Heart Association journal articles | CV_TERM_REGEX post-filter | Unlimited |

Providing `NCBI_API_KEY` + `NCBI_EMAIL` raises the PubMed rate limit to 10 req/sec. Providing `OPENFDA_API_KEY` removes the openFDA daily cap.

---

## Technology Stack

| Layer | Component | Version | Justification |
|-------|-----------|---------|---------------|
| Language | Python | 3.11+ | `match` statements, `tomllib`, typing improvements |
| LLM | [Typhoon v2.5-30b-a3b-instruct](https://opentyphoon.ai) | latest | Thai-native, OpenAI-compatible, free tier (5 req/s), MoE architecture |
| LLM alternative | vLLM / any OpenAI-compatible endpoint | — | Self-hosted or alternative provider via env switch |
| Embeddings | [BGE-M3](https://huggingface.co/BAAI/bge-m3) (`BAAI/bge-m3`) | — | 1024-dim dense vectors, 8192-token context, 100+ languages including Thai |
| Vector store | [ChromaDB](https://www.trychroma.com/) (embedded) | 1.x | Zero-infrastructure, file-backed, supports metadata `where` filters |
| Sparse retrieval | rank-bm25 | 0.2.x | Lightweight BM25 for exact-term matching of drug names and abbreviations |
| Relational DB | SQLite via [SQLAlchemy](https://www.sqlalchemy.org/) 2.0 | 2.0.x | Single-file, Postgres-compatible ORM schema |
| Web framework | [FastAPI](https://fastapi.tiangolo.com/) | 0.13x | Async, auto-OpenAPI, SSE via `StreamingResponse` |
| Data validation | [Pydantic](https://docs.pydantic.dev/) v2 / pydantic-settings | 2.x | Runtime validation, `.env` loading |
| Tokeniser | [tiktoken](https://github.com/openai/tiktoken) | 0.12.x | Token-accurate splitting; `_CharEstimator` fallback for sandboxed envs |
| Thai NLP | [PyThaiNLP](https://pythainlp.github.io/) | 5.x | Thai word boundary handling in text processing |
| Feed parsing | [feedparser](https://feedparser.readthedocs.io/) | — | RFC-compliant RSS/Atom parsing |
| HTML parsing | BeautifulSoup4 + lxml | — | RSS content cleaning |
| HTTP client | httpx + requests | — | Async and sync HTTP |
| Testing | pytest + pytest-asyncio | 9.x | Async fixture support |
| Containerisation | Docker (non-root) | — | HF Spaces compatibility, port 7860 |
| CI/CD | GitHub Actions | — | Lint + test on push; auto-sync to HF Space |

---

## Repository Structure

```
cv-intel-rag/
├── src/
│   ├── main.py                  # FastAPI application and all HTTP endpoints
│   ├── config/
│   │   ├── settings.py          # Pydantic-settings from .env (lru_cache singleton)
│   │   └── domain.py            # MeSH queries, CT.gov terms, openFDA classes,
│   │                            #   RSS feed catalog, CV_TERM_REGEX
│   ├── connectors/
│   │   ├── base.py              # BaseConnector ABC: fetch(since, limit) → Iterable[Record]
│   │   ├── pubmed.py            # NCBI E-utilities (esearch + efetch XML)
│   │   ├── clinical_trials.py  # ClinicalTrials.gov v2 JSON API
│   │   ├── openfda.py           # openFDA drug events + enforcement API
│   │   ├── rss.py               # Generic RSS connector with CV_TERM_REGEX filter
│   │   └── registry.py          # CONNECTORS dict; get_connector() factory
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM: RecordORM, RawPayloadORM,
│   │   │                        #   IngestionRunORM
│   │   ├── repository.py        # RecordRepository: upsert, query, stats
│   │   │                        #   (UUID via str(uuid4()) — SQLite has no UUID type)
│   │   ├── schema.sql           # Raw DDL (used by init_db.py)
│   │   └── session.py           # SessionLocal + dependency_session (FastAPI Depends)
│   ├── models/
│   │   └── record.py            # Pydantic Record + SourceType / CategoryType enums
│   ├── rag/
│   │   ├── chunker.py           # Token-aware chunker (tiktoken / _CharEstimator fallback)
│   │   ├── embedder.py          # BGE-M3 wrapper; get_embedder() lazy singleton
│   │   ├── vectorstore.py       # ChromaDB wrapper; _coerce_metadata (lists → pipe-str)
│   │   ├── retriever.py         # HybridRetriever: α·dense + (1−α)·BM25
│   │   └── indexer.py           # Indexer: chunks records → embeds → upserts to Chroma
│   ├── llm/
│   │   ├── client.py            # OpenAICompatClient + NullLLMClient; get_llm_client()
│   │   └── prompts.py           # System prompt template with [S#] citation instruction
│   ├── agent/
│   │   └── rag_agent.py         # RAGAgent: retrieve → format context → LLM generate
│   ├── static/
│   │   ├── chat.html            # SSE chat UI (vanilla JS, citation resolution)
│   │   └── dashboard.html       # Records browser (vanilla JS, /stats + /search)
│   └── utils/
│       └── logger.py            # Structured logging setup
├── scripts/
│   ├── init_db.py               # Create SQLite schema from schema.sql
│   ├── run_ingestion.py         # CLI: run one or all connectors with --limit
│   ├── rebuild_index.py         # Re-embed all DB records → fresh ChromaDB
│   └── build_notebooks.py       # Programmatically build Colab notebooks
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_chunker.py          # Unit tests: token counting, chunk splitting (no ML)
│   ├── test_connectors.py       # Integration tests: HTTP mocked, real parser logic
│   └── test_rag_integration.py  # End-to-end: real MiniLM + real ChromaDB
├── notebooks/
│   ├── 01_ingest_and_index.ipynb   # Colab: full ingest + index to Google Drive
│   └── 02_demo_visualization.ipynb # Colab: demo queries + data visualisations
├── docs/
│   ├── GUIDE_TH.md              # Developer guide (Thai)
│   ├── COLAB_GUIDE_TH.md        # Colab walkthrough (Thai)
│   └── DEPLOY_GUIDE_TH.md       # HF Spaces deployment guide (Thai)
├── .github/
│   └── workflows/
│       ├── ci.yml               # Lint (ruff) + test (pytest) on every push
│       └── sync-to-hf.yml       # Push main → HF Space when secrets are configured
├── .env.example                 # Template for all environment variables
├── .gitignore                   # Excludes .env, data/*.db, data/chroma/, .venv/
├── CLAUDE.md                    # Claude Code agent hints (repo map, gotchas)
├── Dockerfile                   # Multi-stage build, non-root, port 7860
├── requirements.txt             # Pinned Python dependencies
└── pyproject.toml               # ruff + black configuration
```

---

## API Reference

All endpoints are documented interactively at `GET /docs` (Swagger UI) when the server is running.

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|--------------|----------|
| `GET` | `/` | Chat UI | — | `chat.html` |
| `GET` | `/dashboard` | Records dashboard | — | `dashboard.html` |
| `GET` | `/health` | Liveness probe | — | `{"status": "ok"}` |
| `GET` | `/records` | List/filter ingested records | Query params: `source`, `category`, `since`, `limit` | `[Record]` |
| `GET` | `/stats` | Record counts by source and category | — | `{source: count, …}` |
| `POST` | `/chat` | Blocking RAG answer | `{"query": str, "history": […]}` | `{"answer": str, "sources": […]}` |
| `POST` | `/chat/stream` | SSE streaming RAG answer | `{"query": str, "history": […]}` | `text/event-stream` |
| `POST` | `/search` | Raw hybrid retrieval (no LLM) | `{"query": str, "top_k": int}` | `[{"record": …, "score": float}]` |
| `POST` | `/ingestion/run` | Run a specific connector | `{"connector": str, "limit": int}` | `{"inserted": int, "updated": int}` |
| `POST` | `/index/rebuild` | Re-index all DB records | — | `{"indexed": int}` |

---

## Configuration Reference

All configuration is loaded from `.env` via pydantic-settings. Copy `.env.example` and set the required values.

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `APP_ENV` | `development` | No | `development` / `staging` / `production` |
| `LOG_LEVEL` | `INFO` | No | Python logging level |
| `DATABASE_URL` | `sqlite:///./data/cv_intel.db` | No | SQLAlchemy connection string |
| `CHROMA_PATH` | `./data/chroma` | No | ChromaDB persistence directory |
| `RAW_PAYLOAD_ROOT` | `./data/raw` | No | Directory for raw API response payloads |
| `DEFAULT_LLM_PROVIDER` | `typhoon_api` | No | `typhoon_api` / `vllm_local` / `openai_compat` / `null` |
| `TYPHOON_API_KEY` | — | **Yes** (if using `typhoon_api`) | [Get free key](https://playground.opentyphoon.ai/settings/api-key) |
| `TYPHOON_BASE_URL` | `https://api.opentyphoon.ai/v1` | No | Typhoon endpoint |
| `TYPHOON_MODEL` | `typhoon-v2.5-30b-a3b-instruct` | No | Model identifier |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | No | vLLM server URL |
| `VLLM_MODEL` | `scb10x/typhoon-v2.5-30b-a3b-instruct` | No | Model for vLLM |
| `LLM_TEMPERATURE` | `0.3` | No | Generation temperature (low = more factual) |
| `LLM_MAX_TOKENS` | `8192` | No | Maximum tokens in LLM response |
| `LLM_TIMEOUT_SEC` | `60` | No | LLM request timeout |
| `EMBED_MODEL_NAME` | `BAAI/bge-m3` | No | HuggingFace embedding model |
| `EMBED_DEVICE` | `cpu` | No | `cpu` / `cuda` / `mps` |
| `EMBED_BATCH_SIZE` | `16` | No | Embedding batch size |
| `CHUNK_SIZE_TOKENS` | `500` | No | Target chunk size in tokens |
| `CHUNK_OVERLAP_TOKENS` | `80` | No | Overlap between consecutive chunks |
| `RETRIEVE_TOP_K` | `8` | No | Number of chunks to retrieve per query |
| `RETRIEVE_ALPHA` | `0.6` | No | Dense/BM25 blend (0 = BM25 only, 1 = dense only) |
| `NCBI_API_KEY` | — | No | Raises PubMed rate limit to 10 req/sec |
| `NCBI_EMAIL` | — | No | Required by NCBI when providing an API key |
| `OPENFDA_API_KEY` | — | No | Removes openFDA daily request cap |

---

## Installation and Local Development

**Prerequisites:** Python 3.11 or later, Git.

```bash
# 1. Clone the repository
git clone https://github.com/siriponsri/cv-intel-rag.git
cd cv-intel-rag

# 2. Create and activate a virtual environment
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env in your editor and set TYPHOON_API_KEY
# All other values have sensible defaults for local development

# 5. Initialise the database
python scripts/init_db.py

# 6. Run initial ingestion (first run downloads ~5–10 MB and takes several minutes)
python scripts/run_ingestion.py --limit 20

# 7. Start the development server
uvicorn src.main:app --reload --port 8000
```

- Chat UI: http://localhost:8000
- Dashboard: http://localhost:8000/dashboard
- Swagger docs: http://localhost:8000/docs

**Useful development commands:**

```bash
# Run all connectors without limit
python scripts/run_ingestion.py

# Rebuild the vector index from existing database records
python scripts/rebuild_index.py

# Lint
ruff check src/ tests/

# Format
black src/ tests/
```

---

## Running on Google Colab

Two notebooks in [`notebooks/`](./notebooks/) enable a complete workflow on a free Colab T4 GPU, with data persisted to Google Drive.

| Notebook | Purpose | Estimated Runtime |
|----------|---------|-------------------|
| `01_ingest_and_index.ipynb` | Full ingestion from all 7 sources + BGE-M3 indexing. Saves `data/` to `Drive/cv-intel-rag/`. Run once. | ~10 minutes on T4 |
| `02_demo_visualization.ipynb` | Load the pre-built index from Drive, run demo queries, generate visualisations. | ~30 seconds |

Full step-by-step walkthrough: [`docs/COLAB_GUIDE_TH.md`](./docs/COLAB_GUIDE_TH.md)

---

## Deployment to Hugging Face Spaces

The repository includes full configuration for a zero-downtime deployment to Hugging Face Spaces using the Docker SDK.

**Manual deployment:**

1. Run `01_ingest_and_index.ipynb` on Colab to build the `data/` directory.
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) — select **Docker** SDK, **Public** visibility.
3. In **Settings → Repository secrets**, add `TYPHOON_API_KEY` (and optionally `NCBI_API_KEY`, `OPENFDA_API_KEY`).
4. Push this repository to the Space remote.

**Automatic deployment via GitHub Actions:**

Set the following secrets in your GitHub repository (**Settings → Secrets and variables → Actions**):

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | Hugging Face token with write access to the Space |
| `HF_USERNAME` | Your Hugging Face username |
| `HF_SPACE` | Space name (e.g. `cv-intel-rag`) |

Every push to `main` will automatically synchronise the repository to the Space via `.github/workflows/sync-to-hf.yml`.

Full deployment guide: [`docs/DEPLOY_GUIDE_TH.md`](./docs/DEPLOY_GUIDE_TH.md)

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Fast unit tests only (no ML dependencies)
pytest tests/test_chunker.py -v

# Connector parsing tests (HTTP mocked, no network required)
pytest tests/test_connectors.py -v

# Full RAG integration test (requires sentence-transformers + chromadb)
pytest tests/test_rag_integration.py -v
```

| Test Module | Scope | External Dependencies |
|-------------|-------|-----------------------|
| `test_chunker.py` | Token counting, chunk splitting, Thai text, metadata attachment | None |
| `test_connectors.py` | PubMed XML parsing, ClinicalTrials JSON parsing, openFDA drug class filtering, RSS CV keyword filtering | None (HTTP mocked via `httpx` transport) |
| `test_rag_integration.py` | End-to-end: embed → index → retrieve → assert recall | sentence-transformers (MiniLM for CI speed), chromadb |

---

## Development Guidelines

- **Language**: English for all code comments and docstrings; Thai is acceptable in user-facing documentation under `docs/`.
- **Logging**: `log = logging.getLogger(__name__)` in every module; no `print()` statements in production code.
- **Singletons**: use lazy initialisation via `lru_cache` — see `get_embedder()`, `get_vectorstore()`, `get_llm_client()`. Never instantiate heavy objects at module import time.
- **Adding a new connector**: subclass `BaseConnector`, implement `fetch(since, limit) -> Iterable[Record]`, register in `registry.py`, add a mocked parsing test in `test_connectors.py`. If the source is RSS, use `CV_TERM_REGEX` filtering (already implemented in `RSSConnector`).
- **Modifying retrieval**: edit `HybridRetriever` in `src/rag/retriever.py`; adjust `RETRIEVE_ALPHA` via env var (no code change required). Validate with `test_rag_integration.py::test_index_and_retrieve_end_to_end`.
- **Known design decisions (do not change)**:
  - `_CharEstimator` fallback in `chunker.py` — tiktoken cannot download BPE files in sandboxed/proxied environments.
  - `_coerce_metadata` in `vectorstore.py` — ChromaDB rejects list-valued metadata; values are pipe-joined (`"a|b|c"`). Changing to JSON breaks `where` filter syntax.
  - `str(uuid4())` in `repository.py` — SQLite has no native UUID type; explicit string conversion is required.
  - `CV_TERM_REGEX` uses `\b` only on the left of prefix terms (`cardio`, `sglt`, …) so they match compounds (`Cardiovascular`, `SGLT2`). Thai characters do not respect ASCII word boundaries and stand alone.

---

## Security Considerations

- `.env` and `data/` are listed in `.gitignore`. **API keys must never be committed to the repository.** Verify with `git grep "sk-"` before every push.
- For production deployments, inject secrets exclusively via platform secret managers (HF Space Secrets, GitHub Actions Secrets, AWS Secrets Manager, etc.).
- All external API calls are made with explicit timeouts to prevent indefinite hangs.
- The FastAPI application does not expose any unauthenticated write endpoints in the default configuration. If operating in a multi-user environment, add authentication middleware before exposing `/ingestion/run` and `/index/rebuild`.
- Consider adding a [gitleaks](https://github.com/gitleaks/gitleaks) pre-commit hook for automated secret scanning on public repositories.

---

## Documentation

| Document | Language | Description |
|----------|----------|-------------|
| [`docs/GUIDE_TH.md`](./docs/GUIDE_TH.md) | Thai | General developer guide; VS Code and Claude Code workflow |
| [`docs/COLAB_GUIDE_TH.md`](./docs/COLAB_GUIDE_TH.md) | Thai | Step-by-step Colab notebook walkthrough |
| [`docs/DEPLOY_GUIDE_TH.md`](./docs/DEPLOY_GUIDE_TH.md) | Thai | Hugging Face Spaces deployment guide |
| [`CLAUDE.md`](./CLAUDE.md) | English | Claude Code agent hints: repo map, commands, conventions, critical gotchas |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Credits and Acknowledgements

| Component | Provider | Reference |
|-----------|----------|-----------|
| Typhoon LLM | SCB 10X | [opentyphoon.ai](https://opentyphoon.ai) |
| BGE-M3 Embeddings | BAAI | [huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| PubMed / NCBI E-utilities | National Library of Medicine | [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/) |
| ClinicalTrials.gov API | U.S. National Library of Medicine | [clinicaltrials.gov](https://clinicaltrials.gov/) |
| openFDA API | U.S. Food and Drug Administration | [open.fda.gov](https://open.fda.gov/) |
| FDA MedWatch | U.S. Food and Drug Administration | [fda.gov/safety/medwatch](https://www.fda.gov/safety/medwatch-fda-safety-information-and-adverse-event-reporting-program) |
| EMA What's New | European Medicines Agency | [ema.europa.eu](https://www.ema.europa.eu/) |
| ESC Guidelines | European Society of Cardiology | [escardio.org](https://www.escardio.org/) |
| AHA Circulation | American Heart Association | [ahajournals.org](https://www.ahajournals.org/) |
