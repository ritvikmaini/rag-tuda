"""Ports of the answer step. The answer itself streams through LangChain inside the graph; the analysis call and the token
count go through these."""

from collections.abc import Sequence
from typing import Protocol

from rag_core.domain.context_budget import TokenCounter

__all__ = ["ChatModel", "TokenCounter"]


class ChatModel(Protocol):
    async def complete(self, messages: Sequence[tuple[str, str]]) -> str:
        """The model's reply text to (role, text) messages; roles "system", "user", "assistant"."""
        ...
