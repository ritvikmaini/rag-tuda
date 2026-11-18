"""The reranker client: calls the server's /v2/rerank and turns its scores into (score, text) pairs, best first."""

from collections.abc import Sequence

import httpx

from rag_core.adapters.http import get_http_client
from rag_core.ports.retrieval import RerankError


V2_PATH = "/v2/rerank"


def ranked_pairs(documents: Sequence[str], scores: Sequence[float]) -> list[tuple[float, str]]:
    """(score, text) pairs from /v2/rerank's scores: the highest score per distinct text, sorted by (score, text)
    descending. A score list of another length raises ValueError, which the caller's fallback treats like a failed call."""
    by_text: dict[str, float] = {}
    for doc, score in zip(documents, scores, strict=True):
        if doc not in by_text or score > by_text[doc]:
            by_text[doc] = score
    return sorted(((score, doc) for doc, score in by_text.items()), reverse=True)


async def rerank_pairs(
    client: httpx.AsyncClient, base_url: str, queries: Sequence[str], documents: Sequence[str], *, timeout_s: float
) -> list[tuple[float, str]]:
    """(score, text) per distinct text, best first; RerankError on any request or response failure (a short score list
    included)."""
    try:
        response = await client.post(
            base_url.rstrip("/") + V2_PATH,
            json={"queries": list(queries), "documents": list(documents)},
            timeout=timeout_s,
        )
        response.raise_for_status()
        return ranked_pairs(documents, [float(score) for score in response.json()["scores"]])
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, TypeError) as e:
        raise RerankError(f"{type(e).__name__} {e}") from e


class RerankerHttp:
    """The Reranker port over rerank_pairs and the shared client."""

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url
        self.client = client

    async def rerank(self, queries: Sequence[str], texts: Sequence[str], timeout_s: float) -> list[tuple[float, str]]:
        return await rerank_pairs(self.client or get_http_client(), self.base_url, queries, texts, timeout_s=timeout_s)
