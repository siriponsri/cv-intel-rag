"""
FastAPI app — CV Intel RAG.

Endpoints:
  GET  /                     — serves chat.html
  GET  /dashboard            — serves dashboard.html
  GET  /health               — liveness
  GET  /records              — list/filter records (Layer 1 view)
  GET  /stats                — counts by source/category
  POST /chat                 — non-streaming RAG answer (JSON)
  POST /chat/stream          — SSE streaming RAG answer
  POST /search               — raw retrieval (no LLM generation)
  POST /ingestion/run        — run a single connector + auto-index
  POST /index/rebuild        — rebuild vector store from existing DB records
  GET  /docs                 — Swagger
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .agent.rag_agent import RAGAgent
from .config.settings import settings
from .connectors.registry import CONNECTORS, get_connector
from .db.models import IngestionRunORM, RawPayloadORM, RecordORM
from .db.repository import RecordRepository
from .db.session import dependency_session, SessionLocal
from .rag.indexer import Indexer
from .rag.retriever import get_retriever
from .rag.vectorstore import get_vectorstore
from .utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="CV Intel RAG — Cardiovascular + Diabetes + CKD Intelligence",
    description="Thai LLM-powered RAG agent for cardiometabolic-renal regulatory & research intelligence.",
    version="0.2.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "src" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Request models ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query:        str
    history:      list[dict] | None = None
    top_k:        int | None = None
    source:       str | None = None
    category:     str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    source:   str | None = None
    category: str | None = None


# ─── Meta ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["meta"])
def root() -> str:
    chat_html = STATIC_DIR / "chat.html"
    if chat_html.exists():
        return chat_html.read_text(encoding="utf-8")
    return "<h1>CV Intel RAG</h1><p><a href='/docs'>API docs</a></p>"


@app.get("/dashboard", response_class=HTMLResponse, tags=["meta"])
def dashboard() -> str:
    dash_html = STATIC_DIR / "dashboard.html"
    if dash_html.exists():
        return dash_html.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="dashboard.html not found")


@app.get("/health", tags=["meta"])
def health() -> dict:
    try:
        store_count = get_vectorstore().count()
    except Exception as exc:                                                     # noqa: BLE001
        store_count = -1
        log.warning("vectorstore not ready: %s", exc)
    return {
        "status":        "ok",
        "time":          datetime.utcnow().isoformat(),
        "llm_provider":  settings.default_llm_provider,
        "llm_model":     settings.typhoon_model if settings.default_llm_provider == "typhoon_api" else settings.vllm_model,
        "embed_model":   settings.embed_model_name,
        "vector_count":  store_count,
    }


# ─── Records (Layer 1 view) ─────────────────────────────────────────────

@app.get("/records", tags=["records"])
def list_records(
    q:        Optional[str] = Query(None, description="full-text search in title + summary"),
    source:   Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    region:   Optional[str] = Query(None),
    days:     int = Query(7, ge=1, le=365),
    limit:    int = Query(50, ge=1, le=500),
    session:  Session = Depends(dependency_session),
) -> dict:
    since = datetime.utcnow() - timedelta(days=days)
    stmt = select(RecordORM).where(RecordORM.ingested_at >= since)
    if source:   stmt = stmt.where(RecordORM.source_name == source)
    if category: stmt = stmt.where(RecordORM.category == category)
    if region:   stmt = stmt.where(RecordORM.country_or_region == region)
    if q:
        like = f"%{q}%"
        stmt = stmt.where((RecordORM.title.ilike(like)) | (RecordORM.summary_short.ilike(like)))
    stmt = stmt.order_by(RecordORM.ingested_at.desc()).limit(limit)
    rows = list(session.execute(stmt).scalars().all())

    repo = RecordRepository(session)
    return {
        "count":   len(rows),
        "records": [_record_to_dict(r, repo) for r in rows],
    }


@app.get("/records/{record_id}", tags=["records"])
def record_detail(record_id: str, session: Session = Depends(dependency_session)) -> dict:
    orm = session.get(RecordORM, record_id)
    if not orm:
        raise HTTPException(status_code=404, detail="Record not found")
    repo = RecordRepository(session)
    return _record_to_dict(orm, repo)


# ─── Stats ──────────────────────────────────────────────────────────────

@app.get("/stats", tags=["stats"])
def stats(
    days: int = Query(7, ge=1, le=365),
    session: Session = Depends(dependency_session),
) -> dict:
    since = datetime.utcnow() - timedelta(days=days)

    by_source = session.execute(
        select(RecordORM.source_name, func.count()).where(RecordORM.ingested_at >= since).group_by(RecordORM.source_name)
    ).all()
    by_category = session.execute(
        select(RecordORM.category, func.count()).where(RecordORM.ingested_at >= since).group_by(RecordORM.category)
    ).all()
    by_region = session.execute(
        select(RecordORM.country_or_region, func.count()).where(RecordORM.ingested_at >= since).group_by(RecordORM.country_or_region)
    ).all()
    total = session.execute(
        select(func.count()).select_from(RecordORM).where(RecordORM.ingested_at >= since)
    ).scalar_one()

    recent_runs = list(session.execute(
        select(IngestionRunORM).order_by(IngestionRunORM.started_at.desc()).limit(10)
    ).scalars().all())

    try:
        vector_count = get_vectorstore().count()
    except Exception:                                                            # noqa: BLE001
        vector_count = 0

    return {
        "window_days":   days,
        "total_records": total,
        "vector_chunks": vector_count,
        "by_source":     dict(by_source),
        "by_category":   dict(by_category),
        "by_region":     dict(by_region),
        "recent_runs": [
            {
                "source":     r.source_name,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "status":     r.status,
                "fetched":    r.records_fetched,
                "new":        r.records_new,
                "updated":    r.records_updated,
            }
            for r in recent_runs
        ],
    }


# ─── Chat (RAG) ─────────────────────────────────────────────────────────

@app.post("/chat", tags=["chat"])
def chat(req: ChatRequest) -> dict:
    """Non-streaming RAG answer."""
    agent = RAGAgent()
    where = _build_where(req.source, req.category)
    resp = agent.answer(req.query, chat_history=req.history, where=where, top_k=req.top_k)

    return {
        "query":     resp.query,
        "answer":    resp.answer,
        "model":     resp.model,
        "citations": [c.__dict__ for c in resp.citations],
        "chunks_retrieved": len(resp.chunks_used),
        "tokens": {
            "prompt":     resp.prompt_tokens,
            "completion": resp.completion_tokens,
        },
    }


@app.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    Streaming RAG answer as Server-Sent Events.
    Client reads `data:` lines; each is one JSON event.
    """
    agent = RAGAgent()
    where = _build_where(req.source, req.category)

    def event_stream():
        try:
            for event in agent.answer_stream(
                req.query,
                chat_history=req.history,
                where=where,
                top_k=req.top_k,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:                                                 # noqa: BLE001
            log.exception("chat_stream failed")
            err = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/search", tags=["chat"])
def search(req: SearchRequest) -> dict:
    """Raw retrieval — no LLM generation, just the top-k chunks."""
    where = _build_where(req.source, req.category)
    retriever = get_retriever()
    chunks = retriever.retrieve(req.query, where=where, top_k=req.top_k)
    return {
        "query": req.query,
        "count": len(chunks),
        "chunks": [
            {
                "chunk_id":  c.chunk_id,
                "record_id": c.record_id,
                "text":      c.text,
                "score":     round(c.score, 4),
                "title":     c.metadata.get("title", ""),
                "source":    c.metadata.get("source_name", ""),
                "url":       c.metadata.get("url", ""),
                "published_date": c.metadata.get("published_date", ""),
            }
            for c in chunks
        ],
    }


# ─── Ingestion + Indexing ───────────────────────────────────────────────

@app.post("/ingestion/run", tags=["ingestion"])
def run_ingestion(
    connector: str = Query(..., description=f"One of: {sorted(CONNECTORS)}"),
    limit:     int = Query(20, ge=1, le=200),
    session:   Session = Depends(dependency_session),
) -> dict:
    """Run a connector, persist records, and auto-index them into the vector store."""
    try:
        instance = get_connector(connector)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    run, result = instance.run(limit=limit)
    repo = RecordRepository(session)

    # persist records (Layer 1)
    new = updated = 0
    for record in result.records:
        _, is_new = repo.upsert(record, raw_text=record.raw_text,
                                 checksum=instance.checksum(record.raw_text) if record.raw_text else None)
        if is_new: new += 1
        else:      updated += 1

    run.records_new = new
    run.records_updated = updated
    repo.record_ingestion_run(run)
    session.commit()

    # index into vector store
    chunks_indexed = 0
    if result.records:
        try:
            indexer = Indexer()
            chunks_indexed = indexer.index_records(result.records)
        except Exception as exc:                                                  # noqa: BLE001
            log.warning("indexing failed: %s", exc)

    return {
        "connector":      connector,
        "status":         run.status,
        "fetched":        run.records_fetched,
        "new":            new,
        "updated":        updated,
        "chunks_indexed": chunks_indexed,
        "errors":         result.errors,
    }


@app.post("/index/rebuild", tags=["ingestion"])
def rebuild_index(
    days: int = Query(365, ge=1, le=3650, description="Rebuild using records from last N days"),
) -> dict:
    """Rebuild the vector store from existing DB records (use after schema/embedding changes)."""
    with SessionLocal() as session:
        since = datetime.utcnow() - timedelta(days=days)
        records = list(session.execute(
            select(RecordORM).where(RecordORM.ingested_at >= since)
        ).scalars().all())

        # fetch raw_texts from raw_payloads
        raw_texts: dict[str, str] = {}
        for r in records:
            payload = session.execute(
                select(RawPayloadORM).where(RawPayloadORM.record_id == r.record_id).limit(1)
            ).scalar_one_or_none()
            if payload and payload.raw_text:
                raw_texts[r.record_id] = payload.raw_text
            elif r.summary_short:
                raw_texts[r.record_id] = f"{r.title}\n\n{r.summary_short}"

    indexer = Indexer()
    n = indexer.index_orm_records(records, raw_texts)
    return {"records_considered": len(records), "chunks_indexed": n}


# ─── helpers ────────────────────────────────────────────────────────────

def _build_where(source: Optional[str], category: Optional[str]) -> dict | None:
    filters: dict = {}
    if source:
        filters["source_name"] = source
    if category:
        filters["category"] = category
    if not filters:
        return None
    # ChromaDB needs $and wrapper for multi-field filters
    if len(filters) == 1:
        return filters
    return {"$and": [{k: v} for k, v in filters.items()]}


def _record_to_dict(orm: RecordORM, repo: RecordRepository) -> dict:
    return {
        "record_id":        orm.record_id,
        "title":            orm.title,
        "source_name":      orm.source_name,
        "source_type":      orm.source_type,
        "country_or_region": orm.country_or_region,
        "published_date":   orm.published_date.isoformat() if orm.published_date else None,
        "ingested_at":      orm.ingested_at.isoformat(),
        "url":              orm.url,
        "category":         orm.category,
        "summary_short":    orm.summary_short,
        "topic_tags":       repo.tags_for(orm.record_id, "topic"),
        "entity_tags":      repo.tags_for(orm.record_id, "entity"),
    }
