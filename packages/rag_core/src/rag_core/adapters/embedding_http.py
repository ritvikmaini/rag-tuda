"""The embedding server's HTTP API: POST /embed, GET /info. The indexer path (embed_passages, embed_queries) uses one
AsyncClient per call and a 300 s timeout; the assistant's aembed_query uses the shared client with retries."""

import asyncio
import logging
from typing import Any, cast

import httpx

from rag_core.adapters.http import get_http_client


class EmbeddingClient:
    def __init__(
        self, base_url: str, *, query_prefix: str = "", passage_prefix: str = "", timeout_s: float = 300.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.query_prefix, self.passage_prefix, self.timeout_s = query_prefix, passage_prefix, timeout_s

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient() as client:
            response = await client.post(self.base_url + "/embed", json={"texts": texts}, timeout=self.timeout_s)
            response.raise_for_status()
            return cast("list[list[float]]", response.json()["embeddings"])

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await self._embed([self.passage_prefix + text for text in texts])

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return await self._embed([self.query_prefix + text for text in texts])

    async def aembed_query(
        self,
        text: str,
        *,
        client: httpx.AsyncClient | None = None,
        tries: int = 3,
        retry_delay: float = 0.5,
        timeout_s: float = 30.0,
    ) -> list[float]:
        """The assistant's query vector: the query prefix, the shared async client, three tries 0.5 s and 1 s apart, 30 s
        each; raises RuntimeError when all fail. (embed_queries is the indexer's: no retries, 300 s.)"""
        payload = {"texts": [self.query_prefix + text]}
        last: Exception | None = None
        for attempt in range(tries):
            try:
                response = await (client or get_http_client()).post(
                    self.base_url + "/embed", json=payload, timeout=timeout_s
                )
                response.raise_for_status()
                return cast("list[float]", response.json()["embeddings"][0])
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last = e
                if attempt < tries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logging.warning(
                        f"Embedding service request failed (attempt {attempt + 1}/{tries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"Embedding service request failed after {tries} attempts: {e}")
        raise RuntimeError(f"Embedding service unavailable after {tries} attempts: {last}")

    def info(self, timeout_s: float = 10.0) -> dict[str, Any] | None:
        """The server's identity (model, revision, dimension, …); None when it has no /info (an older image) or does not
        answer."""
        try:
            response = httpx.get(self.base_url + "/info", timeout=timeout_s)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        return cast("dict[str, Any]", response.json())
