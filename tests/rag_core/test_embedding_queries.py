"""The assistant's query embeddings in EmbeddingClient: the query prefix, the URL, three tries 0.5 s and 1 s apart, the error."""

import asyncio
import json

import httpx
import pytest

from rag_core.adapters import embedding_http
from rag_core.adapters.embedding_http import EmbeddingClient
from rag_core.settings.sections import ModelServersSection


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def embedder() -> EmbeddingClient:
    return EmbeddingClient("http://embedding_service:8001", query_prefix="query: ", passage_prefix="")


def test_a_query_gets_the_prefix_and_the_url() -> None:
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    assert asyncio.run(embedder().aembed_query("Wann?", client=client(handle))) == [0.1, 0.2]
    assert seen == [("http://embedding_service:8001/embed", {"texts": ["query: Wann?"]})]


def test_three_tries_then_the_last_error(monkeypatch) -> None:
    waits = []

    async def no_wait(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(embedding_http.asyncio, "sleep", no_wait)  # pyright: ignore[reportPrivateImportUsage]
    with pytest.raises(RuntimeError, match="Embedding service unavailable after 3 attempts"):
        asyncio.run(embedder().aembed_query("x", client=client(lambda r: httpx.Response(503))))
    assert waits == [0.5, 1.0]


def test_the_settings_base_url_gives_the_endpoint() -> None:
    # the default base URL "http://embedding_service:8001" gets /embed appended
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"embeddings": [[0.1]]})

    asyncio.run(EmbeddingClient(ModelServersSection().embedding_url).aembed_query("x", client=client(handle)))
    assert seen == ["http://embedding_service:8001/embed"]
