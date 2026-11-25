"""Hypothesis invariants of the citation entries and of the sources shown after an answer: the page links [n] to numbered
sources, so the numbers must be unique, positive integers in ascending order, whatever a source looks like; "cited" mode
keeps what the answer cites, else all."""

import re

from hypothesis import example, given, settings
from hypothesis import strategies as st
from rag_core.domain import citations as ct
from rag_core.domain import source_meta as sm

number = st.one_of(st.integers(-2, 30), st.booleans(), st.text(max_size=3), st.none())
source = st.one_of(
    st.fixed_dictionaries({"n": number}),
    st.dictionaries(st.sampled_from(["n", "url", "page"]), number),
    st.none(),
    st.integers(),
    st.text(max_size=3),
)
sources = st.lists(source, max_size=12)
dict_sources = st.lists(st.fixed_dictionaries({"n": number}, optional={"title": st.text(max_size=3)}), max_size=12)


def valid(n):
    return isinstance(n, int) and not isinstance(n, bool) and n >= 1


@settings(max_examples=200, deadline=None)
@given(sources=sources)
def test_entries_are_unique_ascending_positive_numbers(sources):
    numbers = [e["n"] for e in ct.entries(sources, [])]
    assert numbers == sorted(set(numbers))
    assert all(valid(n) for n in numbers)


@settings(max_examples=200, deadline=None)
@given(sources=sources)
def test_every_valid_number_gets_one_entry(sources):
    wanted = {s["n"] for s in sources if isinstance(s, dict) and valid(s.get("n"))}
    assert {e["n"] for e in ct.entries(sources, [])} == wanted


@settings(max_examples=200, deadline=None)
@example(sources=[{"n": 1}, {"n": 3}], answer="[1,2]", mode="cited")  # a grouped citation cites each of its numbers
@given(
    sources=dict_sources,
    answer=st.text("ab [],0123456789", max_size=40),
    mode=st.sampled_from(["cited", "seen", "other"]),
)
def test_cited_display_keeps_the_cited_sources_or_all(sources, answer, mode):
    shown = sm.select_sources_for_display(sources, answer, mode)
    groups = re.findall(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]", answer)
    cited = {int(n) for group in groups for n in group.split(",")}
    only_cited = [s for s in sources if s.get("n") in cited]
    assert shown == (only_cited or sources if mode == "cited" else sources)
