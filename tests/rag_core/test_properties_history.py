"""Hypothesis invariants of conversation_history.select_turns: the newest whole turns, no gaps, within the token budget,
stopping only at a turn that does not fit, with no citation marker left in the answers it keeps.
1 token = 3 characters, as test_context_budget.py."""

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, HumanMessage
from rag_core.domain import history as h

word = st.text("abcdefgh", min_size=1, max_size=10)
sentence = st.tuples(st.lists(word, min_size=1, max_size=8), st.sampled_from(["", " [1]", " [2, 3]", "[12]"])).map(
    lambda t: " ".join(t[0]) + "." + t[1]
)
question = st.lists(word, min_size=1, max_size=10).map(lambda w: HumanMessage(content=" ".join(w) + "?"))
answer = st.lists(sentence, min_size=1, max_size=8).map(lambda s: AIMessage(content=" ".join(s)))
turns = st.lists(st.tuples(question, answer), max_size=8)
budget = st.integers(0, 400)


async def count(text):
    return len(text) // 3 + 1


def select(chosen_from, max_tokens):
    return asyncio.run(h.select_turns(chosen_from, max_tokens, count))


def cost(q, a):
    return (
        asyncio.run(count(str(q.content)))
        + asyncio.run(count(h.clean_answer(a.content)))
        + 2 * h.MESSAGE_OVERHEAD_TOKENS
    )


@settings(max_examples=200, deadline=None)
@given(turns=turns, max_tokens=budget)
def test_select_turns_keeps_the_newest_whole_turns_without_gaps(turns, max_tokens):
    chosen = select(turns, max_tokens)
    assert [q for q, _ in chosen] == [q for q, _ in turns[len(turns) - len(chosen) :]]


@settings(max_examples=200, deadline=None)
@given(turns=turns, max_tokens=budget)
def test_select_turns_stays_within_max_tokens(turns, max_tokens):
    total = sum(
        asyncio.run(count(str(q.content))) + asyncio.run(count(a.content)) + 2 * h.MESSAGE_OVERHEAD_TOKENS
        for q, a in select(turns, max_tokens)
    )
    assert total <= max_tokens


@settings(max_examples=200, deadline=None)
@given(turns=turns, max_tokens=budget)
def test_select_turns_stops_only_at_a_turn_that_does_not_fit(turns, max_tokens):
    chosen = select(turns, max_tokens)
    k = len(chosen)
    if not 0 < k < len(turns) or chosen[0][1].content.endswith(h.ELLIPSIS):
        return  # nothing stopped it, or the newest turn was kept shortened (the one case that keeps a turn that did not fit)
    used = sum(cost(q, a) for q, a in turns[len(turns) - k :])
    assert used + cost(*turns[len(turns) - k - 1]) > max_tokens


@settings(max_examples=200, deadline=None)
@given(turns=turns, max_tokens=budget)
def test_selected_answers_carry_no_citation_markers(turns, max_tokens):
    assert all(h.CITATION_MARKER.search(a.content) is None for _, a in select(turns, max_tokens))
