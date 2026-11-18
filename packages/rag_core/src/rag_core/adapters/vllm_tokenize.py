"""vLLM's /tokenize for the context budget."""

import logging
import time
from typing import Any, Dict, List, Optional

from rag_core.adapters.http import get_http_client
from rag_core.domain.context_budget import CHARS_PER_TOKEN_FALLBACK, MESSAGE_OVERHEAD_TOKENS

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
            if data.get("max_model_len"):
                self.max_model_len = int(data["max_model_len"])
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
