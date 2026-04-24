# CLAUDE.md — Hints for Claude Code

> This file is read automatically by [Claude Code](https://docs.claude.com/en/docs/agents-and-tools/claude-code/overview) when you run it inside this repo. It gives Claude context about the codebase so you don't have to re-explain every session.

## Project

CV + DM + CKD regulatory/research intelligence RAG. Ingests from PubMed, ClinicalTrials.gov, openFDA, and 4 RSS feeds; embeds with BGE-M3; serves answers via FastAPI with hybrid dense+BM25 retrieval and Typhoon LLM.

## Repo map

```
src/config/        — settings (pydantic-settings) + domain.py (MeSH, RSS, CV_TERM_REGEX)
src/connectors/    — BaseConnector + 7 concrete (pubmed, clinicaltrials, openfda, rss_*)
src/db/            — SQLAlchemy ORM + schema.sql + RecordRepository
src/models/        — Pydantic Record + enums
src/rag/           — chunker (tiktoken + char fallback), embedder (BGE-M3), vectorstore (Chroma),
                     retriever (hybrid, α=0.6), indexer
src/llm/           — OpenAICompatClient (Typhoon/vLLM/any) + NullLLMClient + prompts
src/agent/         — RAGAgent.answer() + answer_stream() + [S#] citation extraction
src/main.py        — FastAPI: /, /dashboard, /chat (JSON), /chat/stream (SSE), /search, /ingestion/run, /stats
src/static/        — chat.html (ChatGPT-style SSE UI), dashboard.html (vanilla JS)
scripts/           — init_db, run_ingestion, rebuild_index, build_notebooks
tests/             — chunker (fast), connectors (HTTP mocked), rag_integration (real MiniLM+Chroma)
notebooks/         — 01_ingest_and_index.ipynb (runs once on Colab T4), 02_demo_visualization.ipynb
```

## Conventions

- **Code:** English comments + docstrings, type hints, pydantic for data, `log = logging.getLogger(__name__)`
- **User-facing docs:** Thai (`docs/*_TH.md`), `README.md` mostly English (HF Space uses it)
- **Line length:** ~100, follow ruff/black defaults in `pyproject.toml`
- **Imports:** stdlib → third-party → local; relative within `src/`
- **No global singletons without lazy init** — see `get_embedder()`, `get_vectorstore()`, `get_llm_client()`

## Commands

```bash
# dev
python scripts/init_db.py
python scripts/run_ingestion.py --limit 5
uvicorn src.main:app --reload --port 8000

# tests
pytest tests/ -v
pytest tests/test_chunker.py -v              # fast, no ML deps
pytest tests/test_rag_integration.py -v      # needs sentence-transformers + chromadb

# lint/format
ruff check src/ tests/
black src/ tests/

# rebuild notebooks (if you edit scripts/build_notebooks.py)
python scripts/build_notebooks.py
```

## Critical gotchas (DO NOT fix, these are intentional)

- **`tiktoken` 403 error in chunker** — `_CharEstimator` fallback kicks in (sandboxed/proxied envs can't download BPE). Don't "fix" by hardcoding encoder.
- **`_coerce_metadata` in vectorstore.py** — ChromaDB rejects list-valued metadata, so we pipe-join (`"a|b|c"`). Don't change to JSON string — it breaks the `where` filter syntax.
- **UUID in repository.py** — `str(uuid4())` is explicit, not left to SQLAlchemy default, because SQLite doesn't have a UUID type.
- **RSS CV regex** — `\b` only on the LEFT of prefix terms (cardio, sglt, ...) so they match `Cardiovascular` and `SGLT2`. Short codes (`bp`, `dm`, `ckd`) have `\b` on both sides. Thai chars don't respect ASCII `\b`, so they stand alone.

## Env

All config via `.env` (see `.env.example`). Minimal dev env:

```
DATABASE_URL=sqlite:///./data/cv_intel.db
DEFAULT_LLM_PROVIDER=null         # or typhoon_api with TYPHOON_API_KEY
EMBED_DEVICE=cpu
```

For HF Space: same vars, plus `TYPHOON_API_KEY` injected from Secrets.

## When adding a new connector

1. Subclass `BaseConnector` in `src/connectors/<name>.py`
2. Implement `fetch(since, limit) -> Iterable[Record]`
3. Add to `CONNECTORS` dict in `src/connectors/registry.py`
4. Add parsing test to `tests/test_connectors.py` (mock HTTP, assert Record fields)
5. If RSS → use `CV_TERM_REGEX` filter (already plumbed in `RSSConnector`)

## When changing retrieval behavior

- `src/rag/retriever.py` — `HybridRetriever` merges dense (Chroma) + BM25 (rank_bm25) by `final = α·dense + (1−α)·bm25`
- α configured via `RETRIEVE_ALPHA` env var (default 0.6)
- Test changes with `tests/test_rag_integration.py::test_index_and_retrieve_end_to_end`

## When editing the UI

- `src/static/chat.html` uses Server-Sent Events from `/chat/stream` — don't break the SSE contract
- Citations are clickable: LLM must emit `[S1]`, `[S2]` etc., and frontend resolves via `/search?record_id=...`
- `src/static/dashboard.html` queries `/stats` and `/search` — both return JSON

## Deploy

HF Space (Docker SDK). `Dockerfile` at repo root, port 7860, non-root user. See `docs/DEPLOY_GUIDE_TH.md`.

`.github/workflows/sync-to-hf.yml` auto-pushes `main` → HF Space when `HF_TOKEN`/`HF_USERNAME`/`HF_SPACE` secrets are set in GitHub.
