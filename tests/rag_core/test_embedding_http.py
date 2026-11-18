"""rag_core.adapters.embedding_http: the prefixes, one POST per call, /info as an optional identity."""

import asyncio
import json

import httpx
import pytest

from rag_core.adapters.embedding_http import EmbeddingClient

RealAsyncClient = httpx.AsyncClient


def mock(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: RealAsyncClient(transport=transport, **kw))
    real_get = httpx.get
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _get(transport, real_get, url, **kw))


def _get(transport, _real, url, **kw):
    with httpx.Client(transport=transport) as client:
        return client.get(url, **kw)


def test_passages_get_the_passage_prefix_and_one_post_per_call(monkeypatch):
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url), json.loads(request.content), request.extensions["timeout"]))
        return httpx.Response(200, json={"embeddings": [[1.0], [2.0]]})

    mock(monkeypatch, handler)
    client = EmbeddingClient("http://emb:8001/", query_prefix="query: ", passage_prefix="p: ")
    assert asyncio.run(client.embed_passages(["a", "b"])) == [[1.0], [2.0]]
    assert len(seen) == 1
    method, url, body, timeout = seen[0]
    assert (method, url, body) == ("POST", "http://emb:8001/embed", {"texts": ["p: a", "p: b"]})
    assert timeout["read"] == 300


def test_queries_get_the_query_prefix(monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.5]]})

    mock(monkeypatch, handler)
    client = EmbeddingClient("http://emb:8001", query_prefix="query: ", passage_prefix="p: ")
    assert asyncio.run(client.embed_queries(["frage"])) == [[0.5]]
    assert bodies == [{"texts": ["query: frage"]}]


def test_info_returns_the_servers_identity(monkeypatch):
    mock(monkeypatch, lambda request: httpx.Response(200, json={"model": "m", "dimension": 768}))
    assert EmbeddingClient("http://emb:8001").info() == {"model": "m", "dimension": 768}


@pytest.mark.parametrize("reply", [404, 500, "connect"])
def test_info_is_none_without_an_info_route_or_a_reply(monkeypatch, reply):
    def handler(request):
        if reply == "connect":
            raise httpx.ConnectError("down")
        return httpx.Response(reply)

    mock(monkeypatch, handler)
    assert EmbeddingClient("http://emb:8001").info() is None


def test_a_failed_embed_raises(monkeypatch):
    mock(monkeypatch, lambda request: httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(EmbeddingClient("http://emb:8001").embed_passages(["a"]))
