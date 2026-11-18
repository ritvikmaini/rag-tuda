"""fit_context with typed headers, numbering and returned sources.
Uses a fake token counter (1 token per 3 characters) instead of vLLM."""

import asyncio

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
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


def _doc(title, score, text, **extra):
    meta = {
        "title": title,
        "doc_type": "Satzung/Verordnung",
        "doc_year": 2021,
        "page_number": 4,
        "url": "https://example.org/%s.pdf" % title,
        "cross_encoder_score": score,
        "section_id": "section-" + title,
        "matched_child_text": title + ": " + text[100:300],
    }
    meta.update(extra)
    return Document(page_content=text, metadata=meta)


def run(cands):
    fixed = [SystemMessage(content="system"), HumanMessage(content="frage")]
    return asyncio.run(
        cb.fit_context(BUDGET, FakeCounter(), fixed, [HumanMessage(content="Frage?")], cands, labels=EXAMPLE)
    )


def test_headers_numbers_and_sources():
    text = "[Page 4]\n" + "§ 3 Die Steuer beträgt 90 Euro im Jahr. " * 20 + "<!-- 🖼️❌ Image not available -->"
    out = run([_doc("Hundesteuersatzung", 0.9, text), _doc("Gebühren", 0.5, text)])
    ctx = out["context"]
    assert ctx.startswith("Es liegen 2 Quellen vor:")
    assert (
        "### [1] Hundesteuersatzung\nexample.org | Satzung/Verordnung | Stand: 2021 | Seite 4\nLink: https://example.org/Hundesteuersatzung.pdf"
        in ctx
    )
    assert "[Page" not in ctx and "Image not available" not in ctx
    assert [s["n"] for s in out["sources"]] == [1, 2]
    assert out["sources"][0]["section_id"] == "section-Hundesteuersatzung"


def test_rank_score_wins_over_reranker_score():
    text = "Inhalt " * 3000  # large parents: only the first can go in full
    out = run([_doc("A", 0.9, text, rank_score=0.2), _doc("B", 0.5, text, rank_score=0.8)])
    kinds = {s["title"]: s["kind"] for s in out["sources"]}
    assert kinds["B"] == "full"


def test_window_anchor_ignores_title_prefix():
    text = "Einleitung. " * 60 + "ZIELTEXT " * 30 + "Schluss. " * 60
    doc = _doc("T", 0.9, text, matched_child_text="T: " + "ZIELTEXT " * 20)
    window, page = cb._window_and_page(doc, 50)
    assert window.lstrip().startswith("Einleitung") is False and "ZIELTEXT" in window


def test_numbered_context_fits_the_budget():
    # The "[n] " prefixes, the blank lines between sources and the lead line
    # "Es liegen N Quellen vor:" are part of the prompt, so the guard must count them
    text = "Absatz mit Inhalt. " * 102  # parents sized so a count without the overheads left 4 tokens (overshoot 22)
    cands = [_doc("D%02d" % i, 0.9 - i * 0.01, text) for i in range(14)]
    out = run(cands)
    rep = out["report"]
    used = rep["fixed"] + rep["history_tokens"] + asyncio.run(FakeCounter().count_text(out["context"]))
    assert rep["sources_windowed"] > 0  # the budget binds (without the overheads: left 4, overshoot 22)
    assert used <= rep["window"] - rep["reserve"] - rep["margin"]


def test_the_scope_line_sits_between_the_fields_and_the_link():
    """The document's own scope statement, labelled by the profile; nothing without one or a label."""
    labels = SourceLabels(**{**EXAMPLE.__dict__, "scope": "Geltungsbereich laut Dokument"})
    doc = _doc("Ansparung", 0.9, "Text zur Ansparung. " * 30, scope="An alle Beamtinnen und Beamten der Referate")
    assert cb.format_source(doc, labels=labels).startswith(
        "### Ansparung\nexample.org | Satzung/Verordnung | Stand: 2021 | Seite 4\n"
        "Geltungsbereich laut Dokument: An alle Beamtinnen und Beamten der Referate\n"
        "Link: https://example.org/Ansparung.pdf\n\n"
    )
    assert "Geltungsbereich" not in cb.format_source(doc, labels=EXAMPLE)
    assert "Geltungsbereich" not in cb.format_source(_doc("Ansparung", 0.9, "Text. " * 30), labels=labels)
    fixed = [SystemMessage(content="system"), HumanMessage(content="frage")]
    out = asyncio.run(cb.fit_context(BUDGET, FakeCounter(), fixed, [], [doc], labels=labels))
    assert "Geltungsbereich laut Dokument: An alle Beamtinnen und Beamten der Referate\n" in out["context"]
    assert "scope" not in out["sources"][0]  # the page's sources are unchanged
