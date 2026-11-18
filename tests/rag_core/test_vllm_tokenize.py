"""vLLM's /tokenize for the context budget: the URL, the exact count, the window learned from the reply, and the conservative
estimate after a failure."""

import asyncio
import json
import logging

import httpx
import pytest

from rag_core.adapters import vllm_tokenize
from rag_core.adapters.vllm_tokenize import VllmTokenCounter


def counter(monkeypatch, handler, base_url: str = "http://vllm.test/v1") -> VllmTokenCounter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(vllm_tokenize, "get_http_client", lambda: client)
    return VllmTokenCounter(base_url, "Qwen/Qwen3-14B-AWQ", 8192)


def test_the_exact_count_posts_to_tokenize_and_learns_the_window(monkeypatch) -> None:
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"count": 7, "max_model_len": 6000})

    tokens = counter(monkeypatch, handle)
    assert asyncio.run(tokens.count_text("Hallo")) == 7
    assert asyncio.run(tokens.count_chat([{"role": "user", "content": "q"}])) == 7
    assert seen == [
        ("http://vllm.test/tokenize", {"model": "Qwen/Qwen3-14B-AWQ", "prompt": "Hallo"}),
        (
            "http://vllm.test/tokenize",
            {
                "model": "Qwen/Qwen3-14B-AWQ",
                "messages": [{"role": "user", "content": "q"}],
                "add_generation_prompt": True,
            },
        ),
    ]
    assert tokens.max_model_len == 6000 and tokens.exact


def test_a_failure_estimates_two_characters_a_token_warns_once_and_stops_asking(monkeypatch, caplog) -> None:
    caplog.set_level(logging.WARNING)
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503)

    tokens = counter(monkeypatch, handle, base_url="http://vllm.test")
    assert asyncio.run(tokens.count_text("abcd")) == 3  # int(4 / 2.0) + 1
    assert asyncio.run(tokens.count_chat([{"role": "user", "content": "abcd"}])) == (2 + 8) + 8
    assert len(calls) == 1  # the second call did not wait for /tokenize again
    assert not tokens.exact
    assert caplog.text.count("[ContextBudget] /tokenize unavailable") == 1


@pytest.mark.parametrize("path", ["http://vllm.test/v1/", "http://vllm.test"])
def test_the_v1_suffix_is_not_part_of_the_tokenize_url(path: str) -> None:
    assert VllmTokenCounter(path, "m", 1).url == "http://vllm.test/tokenize"
