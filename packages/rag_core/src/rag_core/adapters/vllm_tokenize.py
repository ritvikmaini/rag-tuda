"""The token counter of the context budget, by the card's answer provider: vLLM's /tokenize, else the card's characters per
token. The window is the card's effective window; the server's max_model_len can only lower it."""

import logging
import time
from typing import Any, Dict, List, Optional

from rag_core.adapters.http import get_http_client
from rag_core.domain.context_budget import (
    CHARS_PER_TOKEN_FALLBACK,
    MESSAGE_OVERHEAD_TOKENS,
    CharsPerTokenCounter,
    TokenCounter,
)
from rag_core.settings.card import Card, effective_window

logger = logging.getLogger(__name__)


class VllmTokenCounter:
    """Counts tokens via vLLM's /tokenize; estimates conservatively if the call fails."""

    def __init__(self, base_url: str, model: str, fallback_window: int):
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self.url = base + "/tokenize"
        self.model = model
        self.max_model_len = fallback_window
        self.exact = True
        self.skip_until = 0.0

    async def _post(self, body: Dict[str, Any]) -> Optional[int]:
        if time.monotonic() < self.skip_until:
            return None  # /tokenize failed recently; estimate instead of waiting again
        try:
            resp = await get_http_client().post(self.url, json={"model": self.model, **body}, timeout=3.0)
            resp.raise_for_status()
            data = resp.json()
            if data.get("max_model_len"):  # the server's own window may lower the configured one, never raise it
                self.max_model_len = min(self.max_model_len, int(data["max_model_len"]))
            self.exact = True
            return int(data["count"])
        except Exception as e:
            if self.exact:
                logger.warning(f"[ContextBudget] /tokenize unavailable, estimating tokens: {e}")
            self.exact = False
            self.skip_until = time.monotonic() + 30.0
            return None

    async def count_text(self, text: str) -> int:
        n = await self._post({"prompt": text})
        return n if n is not None else int(len(text) / CHARS_PER_TOKEN_FALLBACK) + 1

    async def count_chat(self, messages: List[Dict[str, str]]) -> int:
        n = await self._post({"messages": messages, "add_generation_prompt": True})
        if n is not None:
            return n
        return (
            sum(int(len(m["content"]) / CHARS_PER_TOKEN_FALLBACK) + MESSAGE_OVERHEAD_TOKENS for m in messages)
            + MESSAGE_OVERHEAD_TOKENS
        )


def token_counter(card: Card) -> TokenCounter:
    """/tokenize for a vLLM answer model; the card's chars_per_token for every other provider (no prompt leaves for a
    tokenizer endpoint the provider does not have)."""
    window = effective_window(card)
    if card.answer.provider == "vllm":
        return VllmTokenCounter(card.answer.url, card.answer.model, window)
    return CharsPerTokenCounter(card.chars_per_token, window)
