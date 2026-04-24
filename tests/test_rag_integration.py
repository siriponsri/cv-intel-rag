"""
Real RAG integration tests.

Uses a TINY real embedder (all-MiniLM-L6-v2, ~22MB) instead of BGE-M3 (2.3GB)
so tests run fast on CI without GPU. The embedder and vector store are both
REAL (not mocked) — only the model size differs from production.

Tests are auto-skipped if sentence-transformers or chromadb are missing.
"""
from __future__ import annotations

import importlib.util
from datetime import date

import pytest

from src.models.record import Category, Record, SourceType

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None
    or importlib.util.find_spec("chromadb") is None,
    reason="sentence-transformers and chromadb required for RAG integration tests",
)


@pytest.fixture()
def tiny_embedder(monkeypatch):
    """Replace BGE-M3 with MiniLM for speed."""
    from src.rag import embedder as embedder_mod

    monkeypatch.setattr(
        "src.config.settings.settings.embed_model_name",
        "sentence-transformers/all-MiniLM-L6-v2",
        raising=False,
    )
    monkeypatch.setattr(
        "src.config.settings.settings.embed_device", "cpu", raising=False,
    )
    # clear singleton so it picks up new settings
    monkeypatch.setattr(embedder_mod, "_default_embedder", None)
    yield


@pytest.fixture()
def isolated_vectorstore(tmp_path, monkeypatch):
    """Force VectorStore to a temp ChromaDB path."""
    from src.rag import vectorstore as vs_mod

    monkeypatch.setattr(
        "src.config.settings.settings.chroma_path", tmp_path / "chroma", raising=False,
    )
    monkeypatch.setattr(vs_mod, "_default_store", None)
    yield vs_mod.get_vectorstore()


def _sample_record(**kwargs) -> Record:
    defaults = dict(
        title="SGLT2 inhibitors in heart failure",
        source_name="PubMed",
        source_type=SourceType.API,
        country_or_region="Global",
        published_date=date(2026, 4, 1),
        url="https://pubmed.gov/fake",
        category=Category.RESEARCH,
        summary_short="RCT shows cardiovascular benefit.",
        raw_text=(
            "Empagliflozin and other SGLT2 inhibitors significantly reduce "
            "cardiovascular mortality in patients with heart failure with "
            "reduced ejection fraction. The EMPEROR-Reduced trial demonstrated "
            "a 25% reduction in composite cardiovascular death or hospitalization. "
            "Dapagliflozin showed similar benefits in DAPA-HF trial."
        ),
        external_id="TEST-001",
    )
    defaults.update(kwargs)
    return Record(**defaults)


def test_index_and_retrieve_end_to_end(tiny_embedder, isolated_vectorstore):
    """Real embed + real ChromaDB + real retrieval."""
    from src.rag.indexer import Indexer
    from src.rag.retriever import HybridRetriever

    indexer = Indexer(store=isolated_vectorstore)
    records = [
        _sample_record(external_id="R1", title="SGLT2 heart failure benefit"),
        _sample_record(
            external_id="R2",
            title="Atenolol beta blocker hypertension",
            raw_text=(
                "Atenolol is a cardioselective beta blocker used for hypertension "
                "and ischemic heart disease. It reduces heart rate and blood pressure."
            ),
        ),
        _sample_record(
            external_id="R3",
            title="Insulin therapy in type 1 diabetes",
            raw_text=(
                "Intensive insulin therapy remains the mainstay of treatment for "
                "type 1 diabetes. Basal-bolus regimens achieve the best glycemic control."
            ),
        ),
    ]

    n = indexer.index_records(records)
    assert n >= 3

    # Semantic query — should find SGLT2 record first
    retriever = HybridRetriever(store=isolated_vectorstore, top_k=3)
    hits = retriever.retrieve("What drugs help with heart failure mortality?")
    assert len(hits) > 0
    top_metadata = hits[0].metadata
    # Either R1 (SGLT2) or R2 (beta blocker) is cardiac-relevant; MiniLM should rank them above R3 (diabetes)
    assert top_metadata.get("external_id") in {"R1", "R2"}, (
        f"cardiac query ranked non-cardiac record first: {top_metadata}"
    )


def test_metadata_filter(tiny_embedder, isolated_vectorstore):
    """Verify source_name filter works via ChromaDB where clause."""
    from src.rag.indexer import Indexer
    from src.rag.retriever import HybridRetriever

    indexer = Indexer(store=isolated_vectorstore)
    indexer.index_records([
        _sample_record(external_id="P1", source_name="PubMed", title="PubMed study"),
        _sample_record(external_id="F1", source_name="openFDA", title="FDA alert", category=Category.SAFETY),
    ])

    retriever = HybridRetriever(store=isolated_vectorstore, top_k=5)
    hits = retriever.retrieve("heart failure", where={"source_name": "PubMed"})
    assert all(h.metadata.get("source_name") == "PubMed" for h in hits), \
        "metadata filter leaked non-PubMed results"


def test_reindex_replaces_old_chunks(tiny_embedder, isolated_vectorstore):
    """Re-indexing the same record_id should not duplicate chunks."""
    from src.rag.indexer import Indexer

    indexer = Indexer(store=isolated_vectorstore)
    r = _sample_record(external_id="REIDX-1")
    indexer.index_records([r])
    first_count = isolated_vectorstore.count()
    indexer.index_records([r])
    second_count = isolated_vectorstore.count()
    assert first_count == second_count, (
        f"re-indexing duplicated chunks: {first_count} → {second_count}"
    )
