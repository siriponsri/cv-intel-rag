"""
LLM client — OpenAI-compatible (works for Typhoon, vLLM self-host,
and any OpenAI-compatible ThaiLLM like Pathumma/OpenThai).

Switching providers is a config change, not a code change:
    DEFAULT_LLM_PROVIDER=typhoon_api   → https://api.opentyphoon.ai/v1
    DEFAULT_LLM_PROVIDER=vllm_local    → http://localhost:8000/v1
    DEFAULT_LLM_PROVIDER=openai_compat → any custom base_url
    DEFAULT_LLM_PROVIDER=null          → offline fallback (no LLM)

Typhoon-specific notes (from docs.opentyphoon.ai):
  • Free tier: 5 req/s, 200 req/min on typhoon-v2.5-30b-a3b-instruct
  • 128K context (12B variant has 56K)
  • Auto-detects request language — responds in Thai if asked in Thai
  • OpenAI SDK works out-of-the-box by overriding base_url
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional

from ..config.settings import settings

log = logging.getLogger(__name__)


# ─── Message shape ──────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str                               # 'system' | 'user' | 'assistant'
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    content: str
    model:   str
    prompt_tokens:     int = 0
    completion_tokens: int = 0


# ─── Base interface ─────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield content tokens as they arrive (Server-Sent Events)."""
        ...


# ─── OpenAI-compatible implementation ──────────────────────────────────

class OpenAICompatClient(BaseLLMClient):
    """
    Works with any endpoint that speaks the OpenAI /v1/chat/completions API:
      • Typhoon  (https://api.opentyphoon.ai/v1)
      • vLLM     (http://localhost:8000/v1 after `vllm serve`)
      • LiteLLM  proxy
      • etc.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: int = 60,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        self.base_url = base_url
        self.model    = model
        self.timeout  = timeout
        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=timeout)

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        temp = temperature if temperature is not None else settings.llm_temperature
        tokens = max_tokens or settings.llm_max_tokens
        log.info("[LLM %s] chat (%d msgs, t=%.2f, max=%d)", self.model, len(messages), temp, tokens)

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            temperature=temp,
            max_tokens=tokens,
        )
        choice = resp.choices[0]
        usage = resp.usage
        return ChatResponse(
            content=choice.message.content or "",
            model=resp.model or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def chat_stream(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        temp = temperature if temperature is not None else settings.llm_temperature
        tokens = max_tokens or settings.llm_max_tokens

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            temperature=temp,
            max_tokens=tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content


# ─── Null fallback (for offline/dev) ────────────────────────────────────

class NullLLMClient(BaseLLMClient):
    """
    Returns a canned message instead of calling any API.
    Lets the rest of the pipeline (retrieval, citations) run without
    a network or API key — useful for CI and smoke tests.
    """

    MESSAGE_EN = (
        "[NullLLMClient: no LLM configured] I found relevant sources but "
        "cannot generate a free-form answer without an LLM. Set "
        "DEFAULT_LLM_PROVIDER and the appropriate API key in .env."
    )
    MESSAGE_TH = (
        "[NullLLMClient: ยังไม่ได้ตั้งค่า LLM] ระบบหาแหล่งข้อมูลที่เกี่ยวข้องได้ "
        "แต่ไม่สามารถสรุปเป็นภาษาธรรมชาติได้ หากยังไม่ได้ตั้งค่า LLM "
        "กรุณาแก้ไฟล์ .env → DEFAULT_LLM_PROVIDER และใส่ API key"
    )

    def chat(self, messages, *, temperature=None, max_tokens=None) -> ChatResponse:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        msg = self._pick_language(last_user.content if last_user else "")
        return ChatResponse(content=msg, model="null", prompt_tokens=0, completion_tokens=0)

    def chat_stream(self, messages, *, temperature=None, max_tokens=None):
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        msg = self._pick_language(last_user.content if last_user else "")
        for word in msg.split():
            yield word + " "

    @staticmethod
    def _pick_language(query: str) -> str:
        # crude Thai detection: any Thai char → Thai response
        if any("\u0e00" <= ch <= "\u0e7f" for ch in query):
            return NullLLMClient.MESSAGE_TH
        return NullLLMClient.MESSAGE_EN


# ─── Factory ────────────────────────────────────────────────────────────

def get_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    """
    Return an LLM client based on config.

    If the selected provider can't be constructed (missing key, unreachable
    endpoint, etc.), falls back to NullLLMClient with a warning.
    """
    chosen = (provider or settings.default_llm_provider or "typhoon_api").lower()

    try:
        if chosen == "typhoon_api":
            if not settings.typhoon_api_key:
                log.warning("TYPHOON_API_KEY not set — using NullLLMClient fallback")
                return NullLLMClient()
            return OpenAICompatClient(
                base_url=settings.typhoon_base_url,
                api_key=settings.typhoon_api_key,
                model=settings.typhoon_model,
                timeout=settings.llm_timeout_sec,
            )

        if chosen == "vllm_local":
            return OpenAICompatClient(
                base_url=settings.vllm_base_url,
                api_key=settings.vllm_api_key,
                model=settings.vllm_model,
                timeout=settings.llm_timeout_sec,
            )

        if chosen == "openai_compat":
            if not (settings.openai_compat_base_url and settings.openai_compat_model):
                log.warning("OPENAI_COMPAT_* not fully configured — using NullLLMClient")
                return NullLLMClient()
            return OpenAICompatClient(
                base_url=settings.openai_compat_base_url,
                api_key=settings.openai_compat_api_key or "not-needed",
                model=settings.openai_compat_model,
                timeout=settings.llm_timeout_sec,
            )

        if chosen in ("null", "none", "offline"):
            return NullLLMClient()

        log.warning("Unknown LLM provider '%s' — using NullLLMClient", chosen)
        return NullLLMClient()

    except Exception as exc:                                                     # noqa: BLE001
        log.exception("Failed to construct LLM client for '%s': %s", chosen, exc)
        return NullLLMClient()
