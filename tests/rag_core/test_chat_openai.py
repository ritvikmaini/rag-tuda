"""The chat model's construction: the request bodies carry the options their provider accepts and no others; the reasoning
level reaches each provider as its own switch; a failed call is told apart by class and status; a rate limit is waited out
within the deadline; a streamed answer reports its usage, reasoning-only chunks included; the analysis model sends its bound
max_tokens."""

import asyncio
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from rag_core.adapters.chat_openai import (
    BUSY,
    TIMEOUT,
    TOO_LONG,
    ChatOpenAIModel,
    ainvoke_waiting,
    build_chat_openai,
    extra_body,
    failure_of,
    reasoning_options,
    response_format,
)
from rag_core.domain.query_analysis import analysis_schema
from rag_core.services.analysis import analyse
from rag_core.services.prompts import PromptService, load_profile_prompts
from rag_core.settings import SettingsError, load_profile
from rag_core.settings.profile import Profile

PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "example" / "profile.yaml"
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
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
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
        {
            **common,
            "temperature": 0.1,
            "max_completion_tokens": 900,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        {**common, "temperature": 0.1, "max_completion_tokens": 900, "stream": False},
        {**common, "temperature": 0.0, "max_completion_tokens": 400, "stream": False},
    ]


def test_extras_only_for_accepting_providers(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
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


def test_ollama_and_lm_studio_get_the_cap_as_max_tokens_too(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    assert extra_body("ollama", "off", EXTRAS, 900) == {"reasoning_effort": "none", "max_tokens": 900}
    assert extra_body("lmstudio", "high", EXTRAS, 900) == {"max_tokens": 900}
    assert "max_tokens" not in extra_body("groq", "low", EXTRAS, 900)
    assert "max_tokens" not in extra_body("vllm", "off", EXTRAS, 900)

    async def sent(provider: str) -> list:
        bodies: list = []
        await new(0.1, bodies, provider=provider).ainvoke(MESSAGES)
        async for _ in new(0.1, bodies, provider=provider).astream(MESSAGES):
            pass
        return bodies

    for provider in ("ollama", "lmstudio"):
        assert [(b["max_tokens"], b["max_completion_tokens"]) for b in asyncio.run(sent(provider))] == [(900, 900)] * 2
    for provider in ("vllm", "groq"):
        assert all("max_tokens" not in body for body in asyncio.run(sent(provider)))


def test_the_analysis_binding_replaces_the_cap_in_the_body_the_client_sends(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    bodies: list = []
    llm = new(0, bodies, provider="ollama")  # the client's own cap is 900
    asyncio.run(ChatOpenAIModel(llm, max_tokens=400).complete([("user", "q")]))
    assert (bodies[0]["max_tokens"], bodies[0]["max_completion_tokens"]) == (400, 400)
    assert bodies[0]["reasoning_effort"] == "none"  # the rest of the client's extra_body is kept


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
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new(0.1, [])
    assert [str(w.message) for w in caught if "extra_body" in str(w.message)] == []


def test_the_analysis_model_sends_its_bound_cap_and_returns_the_text(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    bodies: list = []
    reply = asyncio.run(ChatOpenAIModel(new(0, bodies), max_tokens=400).complete([("system", "s"), ("user", "q")]))
    assert reply == "{}"
    assert bodies[0].get("max_completion_tokens", bodies[0].get("max_tokens")) == 400
    assert bodies[0]["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]


def test_a_reasoning_only_chunk_reaches_the_stream_with_no_text_and_the_usage_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    deltas = (
        {"role": "assistant", "content": None, "reasoning_content": "denke nach"},
        {"content": "Antwort"},
    )

    async def run():
        return [chunk async for chunk in new(0.1, [], deltas=deltas).astream(MESSAGES)]

    chunks = asyncio.run(run())
    assert [c.text for c in chunks if c.text] == ["Antwort"]
    assert chunks[0].text == "" and "denke nach" not in "".join(c.text for c in chunks)  # a chunk, no shown text
    usage = sum((c for c in chunks[1:]), chunks[0]).usage_metadata or {}
    assert (usage["input_tokens"], usage["output_tokens"]) == (11, 3)


def status_error(status: int, code: str | None = None, headers: dict | None = None) -> openai.APIStatusError:
    body = {"error": {"message": "text that is never read", "code": code}}

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body, headers=headers)

    client = openai.AsyncOpenAI(
        base_url=URL, api_key="x", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    with pytest.raises(openai.APIStatusError) as raised:
        asyncio.run(client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "q"}]))
    return raised.value


def test_failures_are_told_apart_by_class_and_status_never_by_message() -> None:
    assert failure_of(status_error(429), "groq") == BUSY
    assert failure_of(status_error(413), "ollama") == TOO_LONG
    assert failure_of(status_error(400, "context_length_exceeded"), "groq") == TOO_LONG
    assert failure_of(status_error(400), "vllm") == TOO_LONG  # vLLM's context error has no code of its own
    assert failure_of(status_error(400), "ollama") == TOO_LONG  # so is Ollama's: 400 and no code
    assert failure_of(status_error(400), "groq") is None  # a rejected parameter is no context error
    assert failure_of(status_error(500), "vllm") is None
    request = httpx.Request("POST", URL)
    assert failure_of(openai.APITimeoutError(request=request), "ollama") == TIMEOUT
    assert failure_of(httpx.ReadTimeout("slow"), "ollama") == TIMEOUT
    assert failure_of(ValueError("input tokens context length 429"), "vllm") is None  # the text decides nothing


def rate_limited(first_retry_after: str | None, calls: list) -> ChatOpenAI:
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(time.monotonic())
        if len(calls) == 1:
            headers = {"retry-after": first_retry_after} if first_retry_after is not None else {}
            return httpx.Response(
                429, json={"error": {"message": "slow down", "code": "rate_limit_exceeded"}}, headers=headers
            )
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )

    return build_chat_openai(
        provider="groq",
        model=MODEL,
        base_url=URL,
        temperature=0.1,
        top_p=0.9,
        max_tokens=10,
        read_timeout_s=5.0,
        reasoning="low",
        extras={},
        max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )


def test_429_waits_retry_after_within_deadline(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    def run(retry_after: str | None, *, deadline_in: float = 60.0, max_wait_s: float = 60.0, retries: int = 1) -> Any:
        calls: list = []
        wait = ainvoke_waiting(
            rate_limited(retry_after, calls),
            MESSAGES,
            retries=retries,
            deadline_at=time.monotonic() + deadline_in,
            max_wait_s=max_wait_s,
        )
        try:
            return asyncio.run(wait), calls
        except openai.RateLimitError:
            return None, calls

    reply, calls = run("0.2")  # inside the deadline: waited out, answered
    assert reply is not None and reply.content == "ok" and len(calls) == 2 and calls[1] - calls[0] >= 0.2
    cases: list[dict[str, Any]] = [
        {"deadline_in": 0.1},  # it would end after the deadline
        {"max_wait_s": 0.1},  # longer than the guard's silence limit
        {"deadline_in": 10.1},  # inside the deadline, but without the margin the retried call needs
        {"max_wait_s": 10.1},  # the same against the silence limit
        {"retries": 0},
    ]
    for kwargs in cases:
        reply, calls = run("0.2", **kwargs)
        assert reply is None and len(calls) == 1, kwargs  # the rate limit is raised: the busy message
    assert run(None)[0] is None  # no retry-after: nothing to wait for
    assert run("soon")[0] is None  # not a number of seconds


def analysis_model(
    bodies: list, *, structured_output: str = "json_schema", provider: str = "ollama", reply: str = "{}"
) -> ChatOpenAIModel:
    """The analysis model as the runtime builds it: a non-streamed client, the schema from the card's structured_output."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("stream"):  # what a streaming client would get
            text = sse({"role": "assistant", "content": reply})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)
        message = {"role": "assistant", "content": reply}
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        return httpx.Response(
            200, json={"id": "c", "object": "chat.completion", "created": 1, "model": MODEL, "choices": [choice]}
        )

    llm = build_chat_openai(
        provider=provider,
        model=MODEL,
        base_url=URL,
        temperature=0,
        top_p=0.1,
        max_tokens=800,
        read_timeout_s=60.0,
        reasoning="off",
        extras={},
        max_retries=0,
        stream=False,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    return ChatOpenAIModel(llm, 800, response_format(structured_output, provider, analysis_schema(None)))


def test_analysis_inside_astream_events_is_not_streamed(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    bodies: list = []
    model = analysis_model(bodies)

    async def analysis_node(_: Any) -> str:
        return await model.complete([("system", "s"), ("user", "q")])

    async def run() -> None:  # the graph's way: an event stream around a runnable that awaits the model
        async for _ in RunnableLambda(analysis_node).astream_events(None, version="v2"):
            pass

    asyncio.run(run())
    assert len(bodies) == 1 and not bodies[0].get("stream")
    assert "stream_options" not in bodies[0]
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert bodies[0]["max_completion_tokens"] == 800


def test_the_cards_structured_output_picks_the_response_format_and_strict_is_groq_only(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    schema = analysis_schema(None)
    assert response_format("json_object", "ollama", schema) == {"type": "json_object"}
    assert response_format("prompt", "ollama", schema) is None
    for provider, strict in (("groq", True), ("ollama", False), ("vllm", False), ("lmstudio", False)):
        assert response_format("json_schema", provider, schema) == {
            "type": "json_schema",
            "json_schema": {"name": "query_analysis", "schema": schema, "strict": strict},
        }, provider

    def body_of(structured_output: str) -> dict:
        bodies: list = []
        asyncio.run(analysis_model(bodies, structured_output=structured_output).complete([("user", "q")]))
        return bodies[0]

    named = response_format("json_schema", "groq", schema, "follow_up_queries")
    assert named is not None and named["json_schema"]["name"] == "follow_up_queries"
    assert body_of("json_object")["response_format"] == {"type": "json_object"}
    assert "response_format" not in body_of("prompt")


def test_the_analysis_schema_is_in_the_strict_shape() -> None:
    schema = analysis_schema(None)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert set(schema["properties"]) == {
        "standalone_question",
        "intent",
        "search_queries",
        "language",
        "date_from",
        "date_to",
        "answer_shape",
    }
    assert schema["properties"]["date_from"]["type"] == ["string", "null"]
    assert "out_of_scope" in schema["properties"]["intent"]["enum"]
    assert schema["properties"]["answer_shape"]["enum"] == ["text", "table", "steps"]


def test_invalid_json_falls_back_to_plain_question(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    question = "Wann hat das Bürgerbüro am Samstag geöffnet?"
    truncated = '{"standalone_question": "Wann hat das Bürgerbüro am Sams'  # cut off at the token cap
    profile = load_profile(Profile, PROFILE)
    analysis, queries = asyncio.run(
        analyse(
            question=question,
            history="",
            now=datetime(2026, 11, 18, 9, 30),
            identity=profile.identity,
            prompts=PromptService(load_profile_prompts(PROFILE.parent / "prompts")),
            model=analysis_model([], reply=truncated),
            timeout_s=5.0,
            max_queries=3,
            first_turn=True,
        )
    )
    assert analysis.parsed == "fallback" and analysis.standalone_question == question
    assert analysis.intent == "lookup" and queries == [question]


def test_groq_authenticates_with_groq_api_key_and_a_local_server_with_openai_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
            },
        )

    for provider in ("groq", "ollama"):
        llm = build_chat_openai(
            provider=provider,
            model=MODEL,
            base_url=URL,
            temperature=0.1,
            top_p=0.1,
            max_tokens=10,
            read_timeout_s=5.0,
            reasoning="off",
            extras={},
            max_retries=0,
            stream=False,
            http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )
        asyncio.run(llm.ainvoke(MESSAGES))
    assert seen == ["Bearer gsk-test", "Bearer not-needed"]


def test_groq_without_its_key_is_a_clear_error_that_never_shows_a_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(SettingsError, match="GROQ_API_KEY: not set in the environment"):
        new(0.1, [], provider="groq")
    monkeypatch.setenv("GROQ_API_KEY", "  ")
    with pytest.raises(SettingsError, match="GROQ_API_KEY"):
        new(0.1, [], provider="groq")
