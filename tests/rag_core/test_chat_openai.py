"""The chat model's construction: the request bodies carry the sampling options without langchain-openai's warning; the
analysis model sends its bound max_tokens."""

import asyncio
import json
import warnings

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_core.adapters.chat_openai import ChatOpenAIModel, build_chat_openai

URL = "http://vllm.test/v1"
MODEL = "Qwen/Qwen3-14B-AWQ"
SAMPLING = {"top_k": 50, "top_p": 0.1, "repetition_penalty": 1.05, "frequency_penalty": None, "presence_penalty": None}
MESSAGES = [SystemMessage(content="s"), HumanMessage(content="q")]


def transport(bodies: list) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("stream"):
            chunks = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": MODEL,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": part}, "finish_reason": None}],
                }
                for part in ("Hal", "lo")
            ]
            text = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
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


def new(temperature: float, bodies: list) -> ChatOpenAI:
    return build_chat_openai(
        model=MODEL,
        temperature=temperature,
        base_url=URL,
        max_tokens=900,
        read_timeout_s=60.0,
        disable_thinking=True,
        sampling=SAMPLING,
        http_async_client=httpx.AsyncClient(transport=transport(bodies)),
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
