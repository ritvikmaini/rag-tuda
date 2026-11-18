"""vLLM's /tokenize for the context budget: the URL, the exact count, the window learned from the reply, and the conservative
estimate after a failure."""

import asyncio
import json
import logging

import httpx
import pytest

from rag_core.adapters import vllm_tokenize
from rag_core.adapters.vllm_tokenize import VllmTokenCounter
from rag_core.domain.context_budget import MESSAGE_OVERHEAD_TOKENS, CharsPerTokenCounter
from rag_core.settings.card import Card


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


def test_the_servers_window_can_lower_the_configured_one_but_never_raise_it(monkeypatch) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"count": 1, "max_model_len": 262144})

    tokens = counter(monkeypatch, handle)
    asyncio.run(tokens.count_text("Hallo"))
    assert tokens.max_model_len == 8192


def test_token_counter_from_card(monkeypatch) -> None:
    posted = []

    def handle(request: httpx.Request) -> httpx.Response:
        posted.append(request)
        return httpx.Response(200, json={"count": 7})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(vllm_tokenize, "get_http_client", lambda: client)

    def card(provider: str, url: str):
        return Card.model_validate(
            {
                "name": "t",
                "class": "M",
                "structured_output": "json_schema",
                "chars_per_token": 4.0,
                "prompt_overlay": "large",
                "slots": 1,
                "answer": {"provider": provider, "url": url, "model": "m", "context_window": 8192},
                "analysis": {
                    "provider": "ollama",
                    "url": "http://127.0.0.1:11434/v1",
                    "model": "a",
                    "context_window": 8192,
                },
                "budget": {"window_tokens": 6000, "answer_tokens": 500},
                "gated": {"iterative_retrieval": True, "version_table": True},
            }
        )

    vllm = vllm_tokenize.token_counter(card("vllm", "http://vllm.test/v1"))
    assert isinstance(vllm, VllmTokenCounter) and vllm.max_model_len == 6000
    assert asyncio.run(vllm.count_text("abcdefgh")) == 7 and len(posted) == 1  # /tokenize answered

    for provider, url in (("ollama", "http://127.0.0.1:11434/v1"), ("groq", "https://api.groq.com/openai/v1")):
        estimate = vllm_tokenize.token_counter(card(provider, url))
        assert isinstance(estimate, CharsPerTokenCounter) and estimate.max_model_len == 6000 and not estimate.exact
        assert asyncio.run(estimate.count_text("abcdefghi")) == 3  # ceil(9 / 4.0)
        chat = [{"role": "user", "content": "abcdefgh"}]
        assert asyncio.run(estimate.count_chat(chat)) == 2 + MESSAGE_OVERHEAD_TOKENS + MESSAGE_OVERHEAD_TOKENS
    assert len(posted) == 1  # nothing was posted to the other providers
