"""The chat model's construction: the request bodies carry the options their provider accepts and no others; the reasoning
level reaches each provider as its own switch; the analysis model sends its bound max_tokens."""

import asyncio
import json
import warnings

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_core.adapters.chat_openai import ChatOpenAIModel, build_chat_openai, extra_body, reasoning_options

URL = "http://vllm.test/v1"
MODEL = "Qwen/Qwen3-14B-AWQ"
EXTRAS = {"top_k": 50, "repetition_penalty": 1.05}
MESSAGES = [SystemMessage(content="s"), HumanMessage(content="q")]


def sse(*deltas: dict, usage: dict | None = None) -> str:
    chunks = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    if usage:  # the last chunk of include_usage has no choices
        chunks.append(
            {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": MODEL, "choices": [], "usage": usage}
        )
    return "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"


def transport(
    bodies: list, deltas: tuple = ({"role": "assistant", "content": "Hal"}, {"content": "lo"})
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("stream"):
            usage = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
            text = sse(*deltas, usage=usage if body.get("stream_options", {}).get("include_usage") else None)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return httpx.MockTransport(handle)


def new(
    temperature: float,
    bodies: list,
    *,
    provider: str = "vllm",
    reasoning: str = "off",
    deltas: tuple = ({"role": "assistant", "content": "Hal"}, {"content": "lo"}),
    max_retries: int = 0,
) -> ChatOpenAI:
    return build_chat_openai(
        provider=provider,
        model=MODEL,
        base_url=URL,
        temperature=temperature,
        top_p=0.1,
        max_tokens=900,
        read_timeout_s=60.0,
        reasoning=reasoning,
        extras=EXTRAS,
        max_retries=max_retries,
        http_async_client=httpx.AsyncClient(transport=transport(bodies, deltas)),
    )


async def calls(build) -> list:
    bodies: list = []
    async for _ in build(0.1, bodies).astream(MESSAGES):  # the answer, streamed as in the graph
        pass
    await build(0.1, bodies).ainvoke(MESSAGES)
    await build(0, bodies).bind(max_tokens=400).ainvoke(MESSAGES)  # the analysis call
    return bodies


def test_the_request_bodies_carry_the_model_the_cap_and_the_sampling_options(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    common = {
        "model": MODEL,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 50,
        "top_p": 0.1,
        "repetition_penalty": 1.05,
    }
    assert asyncio.run(calls(new)) == [
        {**common, "temperature": 0.1, "max_completion_tokens": 900, "stream": True},
        {**common, "temperature": 0.1, "max_completion_tokens": 900, "stream": False},
        {**common, "temperature": 0.0, "max_completion_tokens": 400, "stream": False},
    ]


def test_extras_only_for_accepting_providers(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    assert extra_body("vllm", "off", EXTRAS) == {**EXTRAS, "chat_template_kwargs": {"enable_thinking": False}}
    assert extra_body("ollama", "off", EXTRAS) == {"reasoning_effort": "none"}
    assert extra_body("groq", "low", EXTRAS) == {"reasoning_effort": "low"}
    assert extra_body("lmstudio", "high", EXTRAS) == {}
    # the card's own chat_template_kwargs keep their keys next to the reasoning switch, which wins on its own key
    assert extra_body("vllm", "high", {"chat_template_kwargs": {"enable_thinking": False, "x": 1}}) == {
        "chat_template_kwargs": {"enable_thinking": True, "x": 1}
    }

    async def sent(provider: str) -> dict:
        bodies: list = []
        await new(0.1, bodies, provider=provider).ainvoke(MESSAGES)
        return bodies[0]

    for provider in ("ollama", "groq", "lmstudio"):
        body = asyncio.run(sent(provider))
        assert not {"top_k", "repetition_penalty", "chat_template_kwargs"} & body.keys(), provider
        assert body["top_p"] == 0.1  # the standard fields go to everyone
    assert asyncio.run(sent("vllm"))["top_k"] == 50
    assert asyncio.run(sent("ollama"))["reasoning_effort"] == "none"


@pytest.mark.parametrize(
    ("provider", "level", "option"),
    [
        ("vllm", "off", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("vllm", "low", {"chat_template_kwargs": {"enable_thinking": True}}),
        ("ollama", "off", {"reasoning_effort": "none"}),
        ("ollama", "medium", {"reasoning_effort": "medium"}),
        ("groq", "off", {"reasoning_effort": "none"}),  # a Qwen model on Groq; gpt-oss cards say low
        ("groq", "high", {"reasoning_effort": "high"}),
        ("lmstudio", "high", {}),
    ],
)
def test_the_reasoning_level_maps_to_the_providers_switch(provider: str, level: str, option: dict) -> None:
    assert reasoning_options(provider, level) == option


def test_the_construction_no_longer_warns(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new(0.1, [])
    assert [str(w.message) for w in caught if "extra_body" in str(w.message)] == []


def test_the_analysis_model_sends_its_bound_cap_and_returns_the_text(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    bodies: list = []
    reply = asyncio.run(ChatOpenAIModel(new(0, bodies), max_tokens=400).complete([("system", "s"), ("user", "q")]))
    assert reply == "{}"
    assert bodies[0].get("max_completion_tokens", bodies[0].get("max_tokens")) == 400
    assert bodies[0]["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
