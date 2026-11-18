"""The reranker adapter on /v2/rerank: the /rerank-multi-query answer rebuilt from the aligned scores, so the retrieval service
sees what that endpoint returned, and every failure caught as one RerankError whose text is the former log text."""

import asyncio
import json

import httpx
import pytest
from hypothesis import given, strategies as st

from rag_core.adapters import reranker_http
from rag_core.adapters.reranker_http import RerankerHttp
from rag_core.ports.retrieval import RerankError
from rag_core.settings.sections import ModelServersSection
from rag_testing import scoring


def reranker(handler, base_url: str = "http://reranker_service:8003") -> RerankerHttp:
    return RerankerHttp(base_url, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_the_request_and_the_reply_have_the_v2_shape() -> None:
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"scores": [0.2, 0.9]})  # aligned to the documents

    assert asyncio.run(reranker(handle).rerank(["q1", "q2"], ["a", "b"], 15)) == [(0.9, "b"), (0.2, "a")]
    assert seen == [("http://reranker_service:8003/v2/rerank", {"queries": ["q1", "q2"], "documents": ["a", "b"]})]


@pytest.mark.parametrize(
    "handler,kind",
    [
        (lambda r: httpx.Response(503, json={"detail": "Reranker busy."}), "HTTPStatusError"),
        (lambda r: httpx.Response(200, json={"other": []}), "KeyError"),
        (lambda r: httpx.Response(200, text="not json"), "JSONDecodeError"),
    ],
)
def test_every_failure_the_backend_caught_is_a_rerank_error(handler, kind: str) -> None:
    with pytest.raises(RerankError) as raised:
        asyncio.run(reranker(handler).rerank(["q"], ["a"], 15))
    assert str(raised.value).startswith(kind + " ")


def test_the_settings_base_url_gives_the_endpoint() -> None:
    # the default base URL "http://reranker_service:8003" gets the v2 path appended
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        body = json.loads(request.content)
        return httpx.Response(200, json={"scores": scoring.scores(body["queries"], body["documents"])})

    asyncio.run(reranker(handle, ModelServersSection().reranker_url).rerank(["q"], ["a"], 15))
    assert seen == ["http://reranker_service:8003/v2/rerank"]


TEXTS = ["a b", "b c", "c", "a", "d e f", "B c"]


def expected_pairs(queries, documents):
    """The expected pairs: the best score over the queries per position, then the best per distinct text, highest first."""
    best = [max(scoring.score(q, d) for q in queries) for d in documents]
    by_text = {}
    for doc, score in zip(documents, best):
        if doc not in by_text or score > by_text[doc]:
            by_text[doc] = score
    return sorted(((score, doc) for doc, score in by_text.items()), reverse=True)


@given(
    st.lists(st.sampled_from(TEXTS), min_size=1, max_size=14),
    st.lists(st.sampled_from(["a", "b c", "z", "a d"]), min_size=1, max_size=4),
)
def test_ranked_pairs_take_each_texts_best_score(documents, queries):
    assert reranker_http.ranked_pairs(documents, scoring.scores(queries, documents)) == expected_pairs(
        queries, documents
    )


async def call(transport, queries, documents):
    async with httpx.AsyncClient(transport=transport) as client:
        return await reranker_http.rerank_pairs(client, "http://reranker", queries, documents, timeout_s=15.0)


def test_the_adapter_posts_to_v2_and_returns_score_text_pairs():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return httpx.Response(200, json={"scores": scoring.scores(seen["body"]["queries"], seen["body"]["documents"])})

    docs, queries = ["a b", "c", "a b"], ["a", "c"]
    pairs = asyncio.run(call(httpx.MockTransport(handler), queries, docs))
    assert seen == {"path": "/v2/rerank", "body": {"queries": queries, "documents": docs}}
    assert pairs == expected_pairs(queries, docs)


def test_a_short_or_failed_v2_answer_raises_a_rerank_error():
    short = httpx.MockTransport(lambda r: httpx.Response(200, json={"scores": [0.5]}))
    failed = httpx.MockTransport(lambda r: httpx.Response(503, json={"detail": "Reranker busy."}))
    for transport, text in ((short, "ValueError"), (failed, "HTTPStatusError")):
        with pytest.raises(RerankError) as raised:
            asyncio.run(call(transport, ["a"], ["a", "b"]))
        assert str(raised.value).startswith(text)
