"""The chat model: ChatOpenAI against any OpenAI-compatible server (vLLM, Ollama, LM Studio, Groq), built from the model card,
and the analysis call behind the ChatModel port. The answer is not wrapped: the graph's generate node awaits the ChatOpenAI
client itself, so LangChain's callbacks stream it.

The options only some servers accept (top_k, repetition_penalty, chat_template_kwargs, bad_words ...) are sent to the providers
that accept them and to no other. The reasoning level maps to each provider's own switch in one table."""

from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# the providers whose servers take the options outside the OpenAI chat API
EXTRAS_PROVIDERS = frozenset({"vllm"})


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
    http_async_client: httpx.AsyncClient | None = None,
) -> ChatOpenAI:
    """Bounded waits on the model server: up to read_timeout_s between two streamed chunks (the guard bounds the whole
    answer). The API key comes from OPENAI_API_KEY (compose: not-needed).
    The options travel as extra_body (langchain-openai 1.x warns when they come through model_kwargs; the body is the same)."""
    options: dict[str, Any] = {"base_url": base_url, "max_tokens": int(max_tokens)}
    if body := extra_body(provider, reasoning, extras):
        options["extra_body"] = body
    if http_async_client is not None:
        options["http_async_client"] = http_async_client
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        timeout=httpx.Timeout(connect=5.0, read=read_timeout_s, write=10.0, pool=10.0),
        max_retries=max_retries,
        **options,
    )


class ChatOpenAIModel:
    """The ChatModel port over a ChatOpenAI client: (role, text) messages in, the reply's text out."""

    def __init__(self, llm: Any, max_tokens: int) -> None:
        self.runnable = llm.bind(max_tokens=max_tokens)  # the analysis' length is bounded

    async def complete(self, messages: Sequence[tuple[str, str]]) -> str:
        classes: dict[str, Any] = {"system": SystemMessage, "assistant": AIMessage}
        prompt = [classes.get(role, HumanMessage)(content=text) for role, text in messages]
        response = await self.runnable.ainvoke(prompt)
        return response.content
