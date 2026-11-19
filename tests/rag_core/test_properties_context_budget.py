"""Hypothesis invariants of fit_context and the citation numbering it hands out: whatever the window, the reserve and the
sources, the formatted context fits the budget, a source that was cut to fit never grows back, and the sources are numbered
1..n in candidate order. Texts are one German sentence repeated k times: a random free text did not find an overflow in 400
examples, this does within 300. 1 token = 3 characters, as test_context_budget.py."""

import asyncio

from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from rag_core.domain import context_budget as cb
from rag_core.domain.source_meta import SourceLabels

LABELS = SourceLabels(
    source_types={},
    origins={"archive": "Archiv"},
    hosts={},
    sources_lead="Es liegen {n} Quellen vor:",
    nothing_found="Es wurden keine relevanten Informationen gefunden.",
    date="Stand",
    page="Seite",
    link="Link",
)
SENTENCE = "Die Steuer beträgt 90 Euro im Jahr. "
FIXED = [SystemMessage(content="s")]


class Counter:
    exact = True

    def __init__(self, max_model_len):
        self.max_model_len = max_model_len

    async def count_text(self, text):
        return len(text) // 3 + 1

    async def count_chat(self, messages):
        return sum(len(m["content"]) // 3 + 8 for m in messages)


def build(docs):
    out = []
    for k, title, score, start, length in docs:
        text = SENTENCE * k
        meta = {"title": title, "cross_encoder_score": score, "matched_child_text": text[start : start + length]}
        out.append(Document(page_content=text, metadata=meta))
    return out


def fit(window, reserve, margin, window_chars, full, docs):
    counter = Counter(window)
    budget = cb.ContextBudget(reserve=reserve, margin=margin, window_chars=window_chars, full_sources=full)
    out = asyncio.run(cb.fit_context(budget, counter, FIXED, [], docs, labels=LABELS))
    fixed = asyncio.run(counter.count_chat(cb.to_chat_dicts(FIXED)))
    available = window - reserve - margin - fixed - cb.CONTEXT_LEAD_TOKENS
    return counter, out, fixed, available


document = st.tuples(
    st.integers(1, 200),  # k: the sentence repeated
    st.text("abcdefgh ", min_size=1, max_size=40),
    st.floats(0, 1),
    st.integers(0, 4000),  # the matched child: a slice of the text
    st.integers(0, 300),
)
documents = st.lists(document, min_size=1, max_size=6)
budgets = {
    "window": st.integers(1000, 9000),
    "reserve": st.integers(100, 900),
    "margin": st.integers(0, 200),
    "window_chars": st.sampled_from([50, 300, 2000, 5000]),
    "full": st.integers(0, 3),
}


@settings(max_examples=200, deadline=None)
@given(docs=documents, **budgets)
@example(docs=[(120, "T", 0.9, 0, 200)], window=2000, reserve=900, margin=100, window_chars=5000, full=1)  # the repro
@example(docs=[(74, "T", 0.5, 0, 0)], window=1000, reserve=100, margin=0, window_chars=2000, full=0)  # its shrunk form
def test_fit_context_never_exceeds_the_budget(docs, window, reserve, margin, window_chars, full):
    counter, out, fixed, available = fit(window, reserve, margin, window_chars, full, build(docs))
    # below 300 tokens the deliberate 200-character floor of the truncation branch may overshoot
    assume(available >= 300)
    used = fixed + asyncio.run(counter.count_text(out["context"]))
    assert used <= window - reserve - margin, (used, window - reserve - margin)
    assert out["report"]["left"] >= 0


@settings(max_examples=200, deadline=None)
@given(
    k=st.integers(40, 200),
    window=st.integers(1000, 3000),
    reserve=st.integers(100, 900),
    margin=st.integers(0, 200),
    full=st.integers(0, 3),
)
@example(k=120, window=2000, reserve=900, margin=100, full=1)
def test_a_truncated_source_never_grows_back(k, window, reserve, margin, full):
    doc = build([(k, "T", 0.9, 0, 200)])[0]
    counter, out, _, available = fit(window, reserve, margin, 5000, full, [doc])
    win_text, page = cb._window_and_page(doc, 5000)
    whole = cb.format_source(doc, text=win_text, page=page, labels=LABELS)
    assume(asyncio.run(counter.count_text(whole)) + cb.SOURCE_OVERHEAD_TOKENS > available >= 300)
    assert out["report"]["sources_full"] == 0
    assert len(out["context"]) < len(whole)


@settings(max_examples=200, deadline=None)
@given(docs=documents, **budgets)
def test_sources_are_numbered_one_to_n_in_candidate_order(docs, window, reserve, margin, window_chars, full):
    candidates = build(docs)
    _, out, _, _ = fit(window, reserve, margin, window_chars, full, candidates)
    sources, report = out["sources"], out["report"]
    assert [s["n"] for s in sources] == list(range(1, len(sources) + 1))
    titles = iter(d.metadata["title"] for d in candidates)  # a subsequence of the candidates' titles, in input order
    assert all(any(s["title"] == t for t in titles) for s in sources)
    positions = [out["context"].find("### [%d] " % s["n"]) for s in sources]
    assert positions == sorted(positions) and all(p >= 0 for p in positions)
    assert all(out["context"].count("### [%d] " % s["n"]) == 1 for s in sources)
    assert report["sources_full"] + report["sources_windowed"] == len(sources)
