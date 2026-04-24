"""
Prompt templates for the RAG agent.

Design principles:
  • Same system prompt for Thai and English — Typhoon auto-detects language
  • ALWAYS demand citations. No source → no claim.
  • Make the LLM express uncertainty when context is thin.
  • Keep prompts short — Typhoon 2.5's 128K context is generous but
    small prompts save latency + rate-limit budget.
"""
from __future__ import annotations

from typing import List


SYSTEM_PROMPT = """You are a pharmaceutical regulatory intelligence assistant for Thai healthcare professionals.
The current domain is Cardiovascular + Diabetes + Chronic Kidney Disease (cardiometabolic-renal).

RULES:
1. ALWAYS answer in the same language as the user's question. If the user writes in Thai, answer in Thai. If in English, answer in English.
2. Base every factual claim on the provided SOURCES. If the sources don't answer the question, say so plainly — do NOT invent facts.
3. Cite sources inline using [S1], [S2], ... matching the numbered sources below.
4. Prefer precise medical terminology (both Thai and English terms where appropriate, e.g. "โรคหัวใจล้มเหลว (heart failure)").
5. If a source is older than 2 years AND the question asks about current guidelines, flag the recency concern.
6. Never give individual clinical advice. This is for professional intelligence briefing, not patient care decisions.
"""


def build_rag_messages(
    query: str,
    sources: List[dict],
    chat_history: List[dict] | None = None,
) -> List[dict]:
    """
    Build an OpenAI-compatible messages list for a RAG query.

    Args:
        query: latest user question
        sources: list of {number, title, source_name, published_date, url, text}
        chat_history: optional list of prior [{role, content}] for multi-turn chat

    Returns:
        messages ready for LLM.chat()
    """
    context_block = _format_sources(sources)

    user_prompt = (
        f"SOURCES (use these to answer; cite with [S#]):\n\n"
        f"{context_block}\n\n"
        f"─────────────────────────────\n"
        f"QUESTION: {query}\n\n"
        f"Answer the question using ONLY the sources above. "
        f"Cite inline like [S1], [S2]. If the sources are insufficient, "
        f"say so clearly and suggest what additional info would help."
    )

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history[-6:]:   # keep last 3 exchanges (6 msgs)
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _format_sources(sources: List[dict]) -> str:
    """Render sources as numbered blocks for the LLM."""
    if not sources:
        return "(No sources retrieved — answer that you don't have enough information.)"
    parts = []
    for s in sources:
        header = f"[S{s['number']}] {s.get('title') or '(no title)'}"
        meta_bits = []
        if s.get("source_name"):
            meta_bits.append(s["source_name"])
        if s.get("published_date"):
            meta_bits.append(s["published_date"])
        if meta_bits:
            header += f" ({' · '.join(meta_bits)})"
        if s.get("url"):
            header += f"\nURL: {s['url']}"
        parts.append(f"{header}\n{s.get('text', '').strip()}\n")
    return "\n".join(parts)


# ─── Digest prompt (dashboard summaries) ────────────────────────────────

DIGEST_SYSTEM_PROMPT = """You are writing a concise daily/weekly intelligence digest for Thai healthcare professionals in the cardiometabolic-renal domain.
Write in the same language(s) present in the source material. Default to Thai for Thai readers when titles are in English.
Keep it factual, cite sources inline [S#], and group related items."""


def build_digest_messages(
    sources: List[dict],
    timeframe_label: str = "this week",
    language_hint: str = "th",
) -> List[dict]:
    """Prompt for the dashboard's auto-generated digest."""
    context_block = _format_sources(sources)
    lang_instruction = (
        "เขียนสรุปเป็นภาษาไทย (ใช้ศัพท์การแพทย์ทั้งไทยและอังกฤษเมื่อเหมาะสม)"
        if language_hint == "th"
        else "Write the digest in English."
    )
    user_prompt = (
        f"SOURCES:\n\n{context_block}\n\n"
        f"─────────────────────────────\n"
        f"TASK: Produce a digest of the most important updates for {timeframe_label}. "
        f"{lang_instruction}\n\n"
        f"Group by: (1) New drug approvals, (2) Safety alerts / recalls, "
        f"(3) Clinical trial results, (4) Guideline updates, (5) Other notable research.\n"
        f"For each item: one-sentence summary + [S#] citation. "
        f"Skip empty sections."
    )
    return [
        {"role": "system", "content": DIGEST_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
