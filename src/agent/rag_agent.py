"""
RAG agent — the entry point for chat / question-answering.

Flow:
  1. Receive a user query (+ optional chat history, optional filters)
  2. Retrieve top-k relevant chunks from the vector store (hybrid search)
  3. Build a RAG prompt with numbered sources
  4. Call the LLM
  5. Return answer + citations + used chunks

Callers: FastAPI /chat endpoint, digest generator, CLI tools.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from ..llm.client import BaseLLMClient, ChatMessage, get_llm_client
from ..llm.prompts import build_rag_messages
from ..rag.retriever import HybridRetriever, RetrievedChunk, get_retriever

log = logging.getLogger(__name__)


@dataclass
class Citation:
    """A source cited in the LLM's answer."""
    number:         int
    record_id:      str
    title:          str
    source_name:    str
    published_date: str
    url:            str
    chunk_id:       str
    score:          float


@dataclass
class AgentResponse:
    query:         str
    answer:        str
    citations:     List[Citation] = field(default_factory=list)
    chunks_used:   List[RetrievedChunk] = field(default_factory=list)
    model:         str = ""
    prompt_tokens:     int = 0
    completion_tokens: int = 0


class RAGAgent:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm: Optional[BaseLLMClient] = None,
    ):
        self.retriever = retriever or get_retriever()
        self.llm       = llm       or get_llm_client()

    # ─── main entry ─────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        *,
        chat_history: Optional[List[dict]] = None,
        where: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> AgentResponse:
        """Non-streaming answer. Returns the full AgentResponse."""
        chunks = self.retriever.retrieve(query, where=where, top_k=top_k)
        if not chunks:
            log.info("[Agent] no relevant chunks — returning 'insufficient context' answer")

        sources = self._sources_for_prompt(chunks)
        messages_dict = build_rag_messages(query, sources, chat_history=chat_history)
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_dict]

        resp = self.llm.chat(messages)
        citations = self._extract_citations(resp.content, chunks)

        return AgentResponse(
            query=query,
            answer=resp.content,
            citations=citations,
            chunks_used=chunks,
            model=resp.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )

    def answer_stream(
        self,
        query: str,
        *,
        chat_history: Optional[List[dict]] = None,
        where: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Streaming answer. Yields dicts:
          {"type": "sources",  "sources": [...]}    — once at start
          {"type": "token",    "content": "..."}    — repeatedly
          {"type": "citations","citations": [...]}  — once at end
        """
        chunks = self.retriever.retrieve(query, where=where, top_k=top_k)
        sources = self._sources_for_prompt(chunks)
        yield {"type": "sources", "sources": sources}

        messages_dict = build_rag_messages(query, sources, chat_history=chat_history)
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_dict]

        buffer: List[str] = []
        for token in self.llm.chat_stream(messages):
            buffer.append(token)
            yield {"type": "token", "content": token}

        full_answer = "".join(buffer)
        citations = self._extract_citations(full_answer, chunks)
        yield {
            "type": "citations",
            "citations": [c.__dict__ for c in citations],
        }

    # ─── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _sources_for_prompt(chunks: List[RetrievedChunk]) -> List[dict]:
        """Transform retrieved chunks into the 'sources' dicts the prompt expects."""
        out = []
        for i, c in enumerate(chunks, start=1):
            meta = c.metadata
            out.append({
                "number":         i,
                "chunk_id":       c.chunk_id,
                "record_id":      c.record_id,
                "title":          meta.get("title", ""),
                "source_name":    meta.get("source_name", ""),
                "published_date": meta.get("published_date", ""),
                "url":            meta.get("url", ""),
                "text":           c.text,
                "score":          c.score,
            })
        return out

    @staticmethod
    def _extract_citations(answer: str, chunks: List[RetrievedChunk]) -> List[Citation]:
        """
        Find [S#] markers in the answer and return matching Citation objects.
        If the LLM cited [S3], include chunks[2] — only the ones actually cited.
        """
        nums = sorted({int(m.group(1)) for m in re.finditer(r"\[S(\d+)\]", answer)})
        citations: List[Citation] = []
        for n in nums:
            if 1 <= n <= len(chunks):
                c = chunks[n - 1]
                meta = c.metadata
                citations.append(Citation(
                    number=n,
                    record_id=c.record_id,
                    title=meta.get("title", ""),
                    source_name=meta.get("source_name", ""),
                    published_date=meta.get("published_date", ""),
                    url=meta.get("url", ""),
                    chunk_id=c.chunk_id,
                    score=c.score,
                ))
        return citations
