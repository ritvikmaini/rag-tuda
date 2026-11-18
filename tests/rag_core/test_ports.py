"""The assistant's ports: what a caller can rely on without an adapter."""

from inspect import getmro
from typing import Protocol

from rag_core.domain import context_budget
from rag_core.ports import chat, retrieval


def test_a_rerank_error_prints_the_adapters_log_text():
    error = retrieval.RerankError("ReadTimeout timed out")
    assert isinstance(error, Exception) and str(error) == "ReadTimeout timed out"


def test_the_chat_port_hands_out_the_context_budgets_token_counter():
    assert chat.TokenCounter is context_budget.TokenCounter


def test_every_port_is_a_protocol():
    for port in (
        chat.ChatModel,
        retrieval.SectionStore,
        retrieval.Reranker,
    ):
        assert Protocol in getmro(port), port
