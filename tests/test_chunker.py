"""
Chunker tests — pure Python, no heavy ML deps needed.
Runs in <1s on CI, no GPU required.
"""
from __future__ import annotations

from src.rag.chunker import build_chunks, chunk_text, count_tokens


def test_count_tokens_nonzero_for_nonempty():
    assert count_tokens("") == 0
    assert count_tokens("SGLT2 inhibitors are a drug class.") > 0


def test_short_text_returns_single_chunk():
    text = "Short text about heart failure."
    chunks = chunk_text(text, chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_splits_into_multiple_chunks():
    # 200-sentence blob — guaranteed to exceed any small chunk_size
    sentence = "Empagliflozin reduces cardiovascular death in heart failure patients. "
    text = sentence * 200
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 5, f"expected multiple chunks, got {len(chunks)}"
    for c in chunks:
        assert c.strip(), "empty chunk produced"


def test_build_chunks_attaches_metadata():
    text = "Long. " * 500
    metadata = {"source_name": "PubMed", "title": "Trial X"}
    chunks = build_chunks("record-uuid-1", text, metadata, chunk_size=100, overlap=10)

    assert all(c.record_id == "record-uuid-1" for c in chunks)
    assert all(c.metadata["source_name"] == "PubMed" for c in chunks)
    # chunk_index must match position
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.chunk_id.endswith(f"chunk{i:03d}")


def test_thai_text_chunks():
    """Make sure Thai characters don't break the tokenizer fallback."""
    text = "ยา SGLT2 inhibitors ช่วยลดความเสี่ยงหัวใจล้มเหลว " * 50
    chunks = chunk_text(text, chunk_size=80, overlap=15)
    assert len(chunks) >= 1
    assert all("SGLT2" in c or "หัวใจ" in c or "ยา" in c for c in chunks)
