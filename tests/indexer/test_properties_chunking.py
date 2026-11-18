"""Hypothesis invariants of the structure-aware chunking: the parents lose no word, hold the
target, and every child is a piece of its parent's own text. Pages are paragraphs of lowercase sentences (no heading rule
matches them), as one document or as several numbered pages. Imports as test_chunking_structure.py."""

import os
import re

from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

from indexer.chunking import structure as sc  # noqa: E402

word = st.text("abcdefghij", min_size=1, max_size=12)
sentence = st.lists(word, min_size=1, max_size=10).map(lambda w: " ".join(w) + ".")
paragraph = st.lists(sentence, min_size=1, max_size=8).map(" ".join)
text = st.lists(paragraph, min_size=1, max_size=6).map("\n\n".join)
documents = st.one_of(
    st.lists(st.tuples(st.none(), text), min_size=1, max_size=1),  # a web page
    st.lists(text, min_size=1, max_size=4).map(lambda ts: [(i, t) for i, t in enumerate(ts, 1)]),  # numbered pages
)
targets = st.integers(200, 3000)
PAGE_MARK = re.compile(r"\[Page \d+\]\n")


@settings(max_examples=200, deadline=None)
@given(pages=documents, target=targets)
def test_structure_parents_lose_no_word(pages, target):
    parents = " ".join(t for t, _, _ in sc.structure_parents(pages, target))
    assert {w for _, t in pages for w in t.split()} <= set(parents.split())


@settings(max_examples=200, deadline=None)
@given(pages=documents, target=targets)
def test_plain_text_parents_stay_within_target(pages, target):
    assert all(len(PAGE_MARK.sub("", t)) <= target for t, _, _ in sc.structure_parents(pages, target))


@settings(max_examples=200, deadline=None)
@given(pages=documents, target=targets, mode=st.sampled_from(["recursive", "sentences"]))
def test_children_are_substrings_of_their_parent_without_page_marks(pages, target, mode):
    for parent, _, _ in sc.structure_parents(pages, target):
        segments = sc.page_segments(parent)
        assert all(any(c in s for s in segments) for c in sc.child_texts(parent, mode=mode))


@settings(max_examples=200, deadline=None)
@given(pages=documents, target=targets, mode=st.sampled_from(["recursive", "sentences"]))
def test_children_cover_every_word_of_their_parent(pages, target, mode):
    for parent, _, _ in sc.structure_parents(pages, target):
        wanted = {w for s in sc.page_segments(parent) for w in s.split()}
        assert wanted <= set(" ".join(sc.child_texts(parent, mode=mode)).split())
