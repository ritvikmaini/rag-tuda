"""Ports of the retrieval sequence. The adapters return domain `Hit`s and `Section`s; the service reads them through
rag_core.domain.types.Doc."""

from collections.abc import Sequence
from typing import Protocol

from rag_core.domain.filters import Filter
from rag_core.domain.retrieval import Hit, Section


class RerankError(Exception):
    """The reranker could not score: transport, HTTP status or reply shape. str() is the log text ("<Type> <message>")."""


class VectorStore(Protocol):
    async def search(
        self, vector: Sequence[float], k: int, *, max_distance: float | None = None, filter: Filter | None = None
    ) -> list[Hit]:
        """The k nearest children among the rows the filter keeps (None: no filter at all)."""
        ...


class FullTextIndex(Protocol):
    async def search(self, query: str, k: int, filter: Filter | None = None) -> list[Hit]:
        """The best keyword matches among the rows the filter keeps (None: no filter at all)."""
        ...


class SectionStore(Protocol):
    async def get_many(self, section_ids: Sequence[str], filter: Filter | None = None) -> dict[str, Section]:
        """The sections found among the rows the filter keeps ({} when the lookup fails: the answer goes on without them)."""
        ...

    async def ping(self) -> None:
        """Raises unless the section view answers (readiness)."""
        ...


class Reranker(Protocol):
    async def rerank(self, queries: Sequence[str], texts: Sequence[str], timeout_s: float) -> list[tuple[float, str]]:
        """(score, text) for each distinct text, the maximum over the queries, best first; RerankError when it cannot."""
        ...
