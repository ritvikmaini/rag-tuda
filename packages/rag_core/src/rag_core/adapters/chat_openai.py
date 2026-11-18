"""The chat model: ChatOpenAI against vLLM's OpenAI API, built from the profile, and the analysis call behind the ChatModel
port. The answer is not wrapped: the graph's generate node awaits the ChatOpenAI client itself, so LangChain's callbacks
stream it. The sampling options travel as extra_body (langchain-openai 1.x warns when they come through model_kwargs; the
request body is the same)."""

from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SAMPLING_KEYS = ("top_k", "top_p", "repetition_penalty", "frequency_penalty", "presence_penalty")


def build_chat_openai(
    *,
    model: str,
    temperature: float,
    base_url: str,
    max_tokens: int,
    read_timeout_s: float,
    disable_thinking: bool,
    sampling: Mapping[str, Any],
    http_async_client: httpx.AsyncClient | None = None,
) -> ChatOpenAI:
    """Bounded waits on the model server; up to read_timeout_s between two streamed chunks (the guard bounds the whole
    answer). The API key comes from OPENAI_API_KEY (compose: not-needed)."""
    extra_body: dict[str, Any] = {}
    if disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    for key in SAMPLING_KEYS:
        if sampling.get(key) is not None:
            extra_body[key] = sampling[key]
    options: dict[str, Any] = {"base_url": base_url, "max_tokens": int(max_tokens)}
    if extra_body:
        options["extra_body"] = extra_body
    if http_async_client is not None:
        options["http_async_client"] = http_async_client
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=httpx.Timeout(connect=5.0, read=read_timeout_s, write=10.0, pool=10.0),
        max_retries=1,
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
