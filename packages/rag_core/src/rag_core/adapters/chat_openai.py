"""The chat model: ChatOpenAI against any OpenAI-compatible server (vLLM, Ollama, LM Studio, Groq), built from the model card,
and the analysis call behind the ChatModel port. The answer is not wrapped: the graph's generate node awaits the ChatOpenAI
client itself, so LangChain's callbacks stream it.

The options only some servers accept (top_k, repetition_penalty, chat_template_kwargs, bad_words ...) are sent to the providers
that accept them and to no other. The reasoning level maps to each provider's own switch in one table. A failed call is told
apart by the exception class and the HTTP status, never by the message: a rate limit is waited out within the request's
deadline, a timeout and an over-long prompt are named for the caller."""

import asyncio
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_core.settings import SettingsError

# the providers whose servers take the options outside the OpenAI chat API
EXTRAS_PROVIDERS = frozenset({"vllm"})
RETRY_PAUSE_S = 1.0  # between two attempts after a dropped connection or a server error

BUSY, TIMEOUT, TOO_LONG = "busy", "timeout", "too_long"


def accepts_extras(provider: str) -> bool:
    return provider in EXTRAS_PROVIDERS


def reasoning_options(provider: str, level: str) -> dict[str, Any]:
    """The request option that sets the reasoning level (off, low, medium, high) at the provider. Groq's gpt-oss models
    cannot switch it off: their cards say low."""
    if provider == "vllm":
        return {"chat_template_kwargs": {"enable_thinking": level != "off"}}
    if provider in ("ollama", "groq"):
        return {"reasoning_effort": "none" if level == "off" else level}
    return {}  # LM Studio has no switch


def extra_body(provider: str, reasoning: str, extras: Mapping[str, Any]) -> dict[str, Any]:
    """The body fields beyond the OpenAI chat API: the card's extras where the provider accepts them, and the reasoning
    switch (merged into chat_template_kwargs when the extras carry some)."""
    body = dict(extras) if accepts_extras(provider) else {}
    option = reasoning_options(provider, reasoning)
    if "chat_template_kwargs" in option:
        option = {"chat_template_kwargs": {**body.get("chat_template_kwargs", {}), **option["chat_template_kwargs"]}}
    return {**body, **option}


def api_key(provider: str) -> str | None:
    """The key for the provider: Groq's comes from GROQ_API_KEY in the environment, never from a file; None leaves the client's
    own OPENAI_API_KEY in place (the local servers ignore it)."""
    if provider != "groq":
        return None
    if not (key := os.environ.get("GROQ_API_KEY", "").strip()):
        raise SettingsError(
            "GROQ_API_KEY: not set in the environment (the card's answer comes from Groq) (environment)"
        )
    return key


def build_chat_openai(
    *,
    provider: str,
    model: str,
    base_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    read_timeout_s: float,
    reasoning: str,
    extras: Mapping[str, Any],
    max_retries: int,
    stream: bool = True,
    http_async_client: httpx.AsyncClient | None = None,
) -> ChatOpenAI:
    """Bounded waits on the model server: up to read_timeout_s between two streamed chunks (the guard bounds the whole
    answer). A streamed answer asks for the usage of the call. The API key of a local server comes from OPENAI_API_KEY (compose: not-needed), Groq's from GROQ_API_KEY.
    The options travel as extra_body (langchain-openai 1.x warns when they come through model_kwargs; the body is the same).
    stream=False fixes the client to one non-streamed request: astream_events around the graph cannot turn its calls into streams."""
    options: dict[str, Any] = {"base_url": base_url, "max_tokens": int(max_tokens)}
    if key := api_key(provider):
        options["api_key"] = key
    if body := extra_body(provider, reasoning, extras):
        options["extra_body"] = body
    if http_async_client is not None:
        options["http_async_client"] = http_async_client
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        stream_usage=True,
        disable_streaming=not stream,
        timeout=httpx.Timeout(connect=5.0, read=read_timeout_s, write=10.0, pool=10.0),
        max_retries=max_retries,
        **options,
    )


def response_format(structured_output: str, provider: str, schema: Mapping[str, Any]) -> dict[str, Any] | None:
    """The request's response_format for the card's structured_output: the schema (json_schema, strict only at Groq, whose
    models enforce it), any JSON object (json_object), or none (prompt: the lenient parser reads the text)."""
    if structured_output == "json_schema":
        json_schema = {"name": "query_analysis", "schema": dict(schema), "strict": provider == "groq"}
        return {"type": "json_schema", "json_schema": json_schema}
    if structured_output == "json_object":
        return {"type": "json_object"}
    return None


def failure_of(exc: BaseException, provider: str) -> str | None:
    """BUSY, TIMEOUT or TOO_LONG for a failed call, None for any other. A prompt is too long when the server says 413, names
    the context length in the error code, or (vLLM and Ollama, which have no code for it) reject the request with 400."""
    if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException, TimeoutError)):
        return TIMEOUT
    if isinstance(exc, openai.RateLimitError):
        return BUSY
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 413 or exc.code == "context_length_exceeded":
            return TOO_LONG
        if exc.status_code == 400 and provider in ("vllm", "ollama"):
            return TOO_LONG
    return None


def retry_wait(exc: BaseException) -> float | None:
    """Seconds to wait before another attempt, None when another attempt cannot help: the retry-after of a rate limit (None
    without one), a pause after a dropped connection or a server error, nothing after a timeout."""
    if isinstance(exc, openai.APITimeoutError):
        return None
    if isinstance(exc, openai.RateLimitError):
        try:
            return max(0.0, float(exc.response.headers["retry-after"]))
        except (KeyError, ValueError):
            return None
    if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError)):
        return RETRY_PAUSE_S
    return None


async def ainvoke_waiting(llm: Any, prompt: Any, *, retries: int, deadline_at: float | None, max_wait_s: float) -> Any:
    """llm.ainvoke(prompt), repeated up to `retries` times after a failure that retry_wait() allows. A wait must end before
    deadline_at (time.monotonic()) and last at most max_wait_s (the guard's silence limit); otherwise the failure is raised."""
    attempt = 0
    while True:
        try:
            return await llm.ainvoke(prompt)
        except openai.APIError as exc:
            wait = retry_wait(exc)
            left = max_wait_s if deadline_at is None else min(max_wait_s, deadline_at - time.monotonic())
            if attempt >= retries or wait is None or wait > left:
                raise
            attempt += 1
            await asyncio.sleep(wait)


class ChatOpenAIModel:
    """The ChatModel port over a ChatOpenAI client: (role, text) messages in, the reply's text out."""

    def __init__(self, llm: Any, max_tokens: int, response_format: Mapping[str, Any] | None = None) -> None:
        options: dict[str, Any] = {"max_tokens": max_tokens}  # the analysis' length is bounded
        if response_format is not None:
            options["response_format"] = dict(response_format)
        self.runnable = llm.bind(**options)

    async def complete(self, messages: Sequence[tuple[str, str]]) -> str:
        classes: dict[str, Any] = {"system": SystemMessage, "assistant": AIMessage}
        prompt = [classes.get(role, HumanMessage)(content=text) for role, text in messages]
        response = await self.runnable.ainvoke(prompt)
        return response.content
