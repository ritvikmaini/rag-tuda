"""fit_context counts the history it is given and never trims it: select_turns chose it."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from rag_core.domain import context_budget as cb  # noqa: E402
from rag_core.domain.source_meta import SourceLabels  # noqa: E402

EXAMPLE = SourceLabels(  # the literal of tests/rag_core/domain/test_source_labels.py
    source_types={"map": "Karte"},
    origins={"archive": "Archiv", "website": "example.org"},
    hosts={"docs.example.org": "Archiv", "example.org": "example.org", "www.example.org": "example.org"},
)


class FakeCounter:
    max_model_len = 8192
    exact = True

    async def count_text(self, text):
        return len(text) // 3 + 1

    async def count_chat(self, messages):
        return sum(len(m["content"]) // 3 + 8 for m in messages)


BUDGET = cb.ContextBudget(reserve=900, margin=128, window_chars=300, full_sources=3)


def test_history_is_counted_whole_and_never_trimmed():
    history = [
        HumanMessage(content="F" * 600),
        AIMessage(content="A" * 900),
        HumanMessage(content="G" * 300),
        AIMessage(content="B" * 300),
    ]
    fixed = [SystemMessage(content="system"), HumanMessage(content="frage")]
    out = asyncio.run(cb.fit_context(BUDGET, FakeCounter(), fixed, history, [], labels=EXAMPLE))
    rep = out["report"]
    assert [m.content for m in out["history"]] == [m.content for m in history]
    assert rep["history_kept"] == rep["history_total"] == 4
    assert rep["history_tokens"] == sum(len(m.content) // 3 + 1 + cb.MESSAGE_OVERHEAD_TOKENS for m in history)
