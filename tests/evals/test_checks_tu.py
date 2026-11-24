"""The answer checks of the TU profile, run on hand-written answers: a German and an English one for each case, with [n]
markers and the lines code appends. The expected lines come from the profile's messages through the producer
(rag_core.domain.notes), so a change of wording moves producer and check together."""

import dataclasses
import json
import pathlib

import httpx
import pytest

from evals import checks, runner
from evals.checks import (
    answer_language_matches,
    citations_map_to_sources,
    evaluate_turn,
    load_answer_texts,
    po_line_present,
    sie_form,
    visa_disclaimer_present,
)
from rag_core.domain.notes import appended_lines

PROFILE = pathlib.Path(__file__).resolve().parents[2] / "profiles" / "tu-informatik"
TEXTS, PROGRAMMES = load_answer_texts(str(PROFILE))

BSC_2023 = {"programme": "bsc-informatik", "po_version": 2023, "regulation": "po", "lang": "de"}
BSC_2015 = {"programme": "bsc-informatik", "po_version": 2015, "regulation": "po", "lang": "de"}
VISA = {"regulation": "residence", "lang": "de"}
FAQ = {"regulation": "faq", "lang": "de"}

DE = "Sie müssen die Prüfung bis zum Ende des dritten Semesters ablegen [1]. Wenn Sie krank sind, melden Sie sich im Prüfungsbüro [2]."
EN = "You have to take the exam by the end of the third semester [1]. If you are ill, you must contact the examination office [2]."
DE_INFORMAL = (
    "Du musst die Prüfung bis zum Ende des dritten Semesters ablegen [1]. Melde dich bei deinem Prüfungsbüro [2]."
)


def with_lines(answer, cited, language="de"):
    """The answer as the producer would send it: the lines it appends for the cited sources' metadata."""
    lines = appended_lines(cited, PROGRAMMES, TEXTS[language], english=language == "en", term=False)
    return "\n\n".join([answer, *lines])


# ---- citations_map_to_sources


def test_citations_map_to_sources():
    assert citations_map_to_sources(DE, {1, 2, 3})
    assert citations_map_to_sources("Das gilt [1, 3] und [2].", {1, 2, 3})
    assert citations_map_to_sources("Keine Quellenangabe nötig.", set())
    assert not citations_map_to_sources(DE, {1})  # [2] has no source
    assert not citations_map_to_sources("Das gilt [1, 4].", {1, 2, 3})
    assert not citations_map_to_sources(DE, set())  # citations without any source


# ---- answer_language_matches


def test_answer_language_matches():
    assert answer_language_matches(DE, "de")
    assert answer_language_matches(EN, "en")
    assert not answer_language_matches(DE, "en")  # a German answer to an English question
    assert not answer_language_matches(EN, "de")
    assert answer_language_matches(with_lines(EN, [BSC_2023], "en"), "en")


# ---- po_line_present


def test_po_line_present_after_a_version_bound_citation():
    meta = {1: BSC_2023, 2: FAQ}
    line = "Grundlage: Prüfungsordnung 2023 des B.Sc. Informatik"
    assert line in with_lines(DE, [BSC_2023])  # the wording of the spec's example
    assert po_line_present(with_lines(DE, [BSC_2023]), meta, TEXTS["de"], PROGRAMMES, "de")
    assert not po_line_present(DE, meta, TEXTS["de"], PROGRAMMES, "de")


def test_po_line_names_every_cited_version():
    meta = {1: BSC_2023, 2: BSC_2015}
    both = with_lines(DE, [BSC_2023, BSC_2015])
    assert "Grundlage: PO 2023 und PO 2015 des B.Sc. Informatik" in both
    assert po_line_present(both, meta, TEXTS["de"], PROGRAMMES, "de")
    assert not po_line_present(with_lines(DE, [BSC_2023]), meta, TEXTS["de"], PROGRAMMES, "de")  # 2015 is cited too


def test_po_line_is_not_required_for_unbound_or_uncited_sources():
    assert po_line_present(DE, {1: FAQ, 2: FAQ}, TEXTS["de"], PROGRAMMES, "de")
    assert po_line_present("Das gilt [1].", {1: FAQ, 2: BSC_2023}, TEXTS["de"], PROGRAMMES, "de")  # [2] is not cited


def test_po_line_in_english_and_with_a_start_term():
    meta = {1: BSC_2023, 2: FAQ}
    assert po_line_present(with_lines(EN, [BSC_2023], "en"), meta, TEXTS["en"], PROGRAMMES, "en")
    assert not po_line_present(EN, meta, TEXTS["en"], PROGRAMMES, "en")
    term = appended_lines([BSC_2023], PROGRAMMES, TEXTS["de"], english=False, term=True)
    assert po_line_present("\n\n".join([DE, *term]), meta, TEXTS["de"], PROGRAMMES, "de")


# ---- sie_form


def test_sie_form():
    assert sie_form(DE)
    assert not sie_form(DE_INFORMAL)
    assert sie_form("Die Frist endet am 15. Juli [1].")  # no address at all
    assert sie_form(EN)  # English answers have no "Sie"


# ---- visa_disclaimer_present


def test_visa_disclaimer_present_after_a_residence_citation():
    meta = {1: VISA, 2: FAQ}
    answer = "Die Aufenthaltserlaubnis beantragen Sie bei der Ausländerbehörde [1]."
    assert visa_disclaimer_present(with_lines(answer, [VISA]), meta, TEXTS["de"], PROGRAMMES, "de")
    assert not visa_disclaimer_present(answer, meta, TEXTS["de"], PROGRAMMES, "de")
    assert "Keine Rechtsberatung. Verbindlich entscheidet die zuständige Ausländerbehörde" in with_lines(answer, [VISA])


def test_visa_disclaimer_in_english_and_when_not_required():
    meta = {1: {"regulation": "residence", "lang": "en"}, 2: FAQ}
    answer = "You apply for the residence permit at the immigration authority [1]."
    assert visa_disclaimer_present(with_lines(answer, [meta[1]], "en"), meta, TEXTS["en"], PROGRAMMES, "en")
    assert not visa_disclaimer_present(answer, meta, TEXTS["en"], PROGRAMMES, "en")
    assert visa_disclaimer_present("Das Prüfungsbüro hilft [2].", meta, TEXTS["de"], PROGRAMMES, "de")  # FAQ only


# ---- the checks through evaluate_turn


@pytest.fixture
def tu_profile(monkeypatch):
    monkeypatch.setenv("RAG__EVALS__PROFILE_DIR", str(PROFILE))


def run(answer, meta, **expect):
    return evaluate_turn(answer, [], [], expect, meta)["failed"]


def test_evaluate_turn_passes_a_complete_german_answer(tu_profile):
    meta = {"citations": [1, 2], "source_metadata": {1: BSC_2023, 2: VISA}}
    answer = with_lines(DE, [BSC_2023, VISA])
    every = dict.fromkeys(("citations_map_to_sources", "po_line_present", "sie_form", "visa_disclaimer_present"), True)
    assert run(answer, meta, answer_language_matches="de", **every) == []


def test_evaluate_turn_names_each_failed_check(tu_profile):
    meta = {"citations": [1], "source_metadata": {1: BSC_2023, 2: VISA}}
    every = dict.fromkeys(("citations_map_to_sources", "po_line_present", "sie_form", "visa_disclaimer_present"), True)
    assert run(DE_INFORMAL, meta, answer_language_matches="en", **every) == [
        "informal_address",  # the library's own check fires too
        "citations_map_to_sources",
        "answer_language_matches",
        "sie_form",
        "po_line_present",
        "visa_disclaimer_present",
    ]
    assert run(DE, {"citations": [1, 2], "source_metadata": {1: BSC_2023, 2: VISA}}, **every) == [
        "po_line_present",
        "visa_disclaimer_present",
    ]


def test_evaluate_turn_skips_checks_it_is_not_asked_for():
    assert run(DE_INFORMAL, {}) == ["informal_address"]


def test_texts_come_from_the_profile_messages():
    assert set(TEXTS) == {"de", "en"}
    assert TEXTS["de"].visa_disclaimer and TEXTS["en"].po_line
    assert checks.load_answer_texts(str(PROFILE)) is checks.load_answer_texts(str(PROFILE))  # read once


# ---- through the runner: the docs and citations frames of a stream give the checks their sources


def stream(*items):
    return "".join("data: " + json.dumps({"type": t, "content": c}, ensure_ascii=False) for t, c in items).encode()


DOCS = (
    "docs",
    [
        {"page_content": "§ 5", "metadata": {"url": "https://tu.example/po.pdf", **BSC_2023}},
        {"page_content": "Krank", "metadata": {"url": "https://tu.example/x", **FAQ}},
    ],
)
CITATIONS = ("citations", [{"n": 1, "url": "https://tu.example/po.pdf"}, {"n": 2, "url": "https://tu.example/x"}])


def played(monkeypatch, body, expect):
    """One turn through run_conversation and ask against a fake stream; the row it records."""
    monkeypatch.setenv("RAG__EVALS__PROFILE_DIR", str(PROFILE))
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))
    conv = {"id": "T1", "set": "tu", "turns": [{"q": "Wie lange habe ich Zeit?", "expect": expect}]}

    def ask_fn(question, user, opts):
        return runner.ask(client, "http://b", "tok", question, user, 10, **opts)

    return runner.run_conversation(conv, ask_fn, 0, "t", "s", lambda row: None)[0]


LINE_CHECKS = {"citations_map_to_sources": True, "po_line_present": True, "visa_disclaimer_present": True}


def test_runner_fails_a_missing_po_line(monkeypatch):
    row = played(monkeypatch, stream(DOCS, ("chunk", DE), CITATIONS), LINE_CHECKS)
    assert row["checks"]["failed"] == ["po_line_present"]


def test_runner_passes_the_answer_with_its_po_line(monkeypatch):
    row = played(monkeypatch, stream(DOCS, ("chunk", with_lines(DE, [BSC_2023])), CITATIONS), LINE_CHECKS)
    assert row["checks"]["failed"] == []


def test_runner_fails_a_missing_visa_disclaimer(monkeypatch):
    visa_doc = ("docs", [{"page_content": "Visum", "metadata": {"download_url": "https://tu.example/v", **VISA}}])
    cited = ("citations", [{"n": 1, "url": "https://tu.example/v"}])
    answer = "Die Aufenthaltserlaubnis beantragen Sie bei der Ausländerbehörde [1]."
    assert played(monkeypatch, stream(visa_doc, ("chunk", answer), cited), LINE_CHECKS)["checks"]["failed"] == [
        "visa_disclaimer_present"
    ]
    full = with_lines(answer, [VISA])
    assert played(monkeypatch, stream(visa_doc, ("chunk", full), cited), LINE_CHECKS)["checks"]["failed"] == []


def test_runner_fails_a_citation_without_a_source(monkeypatch):
    row = played(monkeypatch, stream(DOCS, ("chunk", DE + " Mehr [3]."), CITATIONS), {"citations_map_to_sources": True})
    assert row["checks"]["failed"] == ["citations_map_to_sources"]


def test_source_metadata_joins_citations_to_documents_by_url():
    docs = [
        {"metadata": {"url": "https://a", "po_version": 2023, "matched_child_text": "Text"}},
        {"metadata": {"download_url": "https://b/x.pdf", "regulation": "residence"}},
        {"page_content": "Keine Quellen", "metadata": {}},
    ]
    cited = [{"n": 1, "url": "https://b/x.pdf"}, {"n": 2, "url": "https://a"}, {"n": 3, "url": "https://zz"}, "x"]
    found = runner.source_metadata([*cited, {"n": 4, "url": ""}], docs)
    # the lines' metadata only; a citation without a url joins nothing, not the document without one
    assert found == {1: {"regulation": "residence"}, 2: {"po_version": 2023}}


def test_runner_fails_the_line_checks_when_the_docs_frame_follows_the_citations(monkeypatch):
    """sources_event seen/cited: the docs frame comes after the citations frame and is joined all the same."""
    late = stream(("chunk", DE), CITATIONS, DOCS)
    assert played(monkeypatch, late, LINE_CHECKS)["checks"]["failed"] == ["po_line_present"]
    complete = stream(("chunk", with_lines(DE, [BSC_2023])), CITATIONS, DOCS)
    assert played(monkeypatch, complete, LINE_CHECKS)["checks"]["failed"] == []


def test_a_cited_number_without_metadata_fails_the_line_checks(monkeypatch):
    row = played(monkeypatch, stream(("chunk", DE), CITATIONS), LINE_CHECKS)  # no docs frame at all
    assert row["checks"]["failed"] == ["po_line_present:unjoined", "visa_disclaimer_present:unjoined"]
    assert not po_line_present(DE, {}, TEXTS["de"], PROGRAMMES, "de")
    assert not visa_disclaimer_present(DE, {1: BSC_2023}, TEXTS["de"], PROGRAMMES, "de")  # [2] has none


def test_the_row_keeps_what_rescoring_needs(monkeypatch):
    row = played(monkeypatch, stream(DOCS, ("chunk", DE), CITATIONS), LINE_CHECKS)
    assert row["citations"] == [1, 2] and row["source_metadata"] == {
        1: {"programme": "bsc-informatik", "po_version": 2023, "regulation": "po", "lang": "de"},
        2: {"regulation": "faq", "lang": "de"},
    }


def test_rescore_gives_the_same_verdict_as_the_run(monkeypatch):
    from evals.rescore import rescore_rows

    failing = played(monkeypatch, stream(DOCS, ("chunk", DE), CITATIONS), LINE_CHECKS)
    passing = played(monkeypatch, stream(DOCS, ("chunk", with_lines(DE, [BSC_2023])), CITATIONS), LINE_CHECKS)
    rows = [json.loads(json.dumps(r)) for r in (failing, passing)]  # as read back from results.jsonl: text keys
    rescored, changed = rescore_rows(rows, set())
    assert [r["checks"]["failed"] for r in rescored] == [["po_line_present"], []] and changed == 0


# ---- quotations, the notes' fields, the declared language


@pytest.mark.parametrize(
    "answer",
    [
        "Im Formular steht: „Du musst dich anmelden“. Sie müssen sich bis Freitag anmelden [1].",
        'Im Formular steht "du musst dich anmelden". Sie müssen sich bis Freitag anmelden [1].',
    ],
)
def test_a_quoted_du_is_not_an_informal_address(answer):
    assert sie_form(answer)
    assert checks.informal_pronouns(answer) == []
    assert evaluate_turn(answer, [], [], {"sie_form": True})["failed"] == []
    assert not sie_form("Du musst dich anmelden, wie im Formular steht: „Sie müssen“.")


def test_answer_texts_has_one_field_per_notes_text():
    from rag_core.domain.notes import Texts

    notes_fields = {name for name, value in vars(Texts).items() if isinstance(value, property)}
    assert {f.name for f in dataclasses.fields(checks.AnswerTexts)} == notes_fields


def test_the_declared_language_picks_the_lines(tu_profile):
    short = "Apply Friday [1]."  # no stopword: the detector would say German
    assert checks.detect(short) == "de"
    english_line = with_lines(short, [BSC_2023], "en")
    meta = {"source_metadata": {1: BSC_2023}}
    assert run(english_line, meta, po_line_present=True, answer_language_matches="en") == []
    german_line = with_lines(short, [BSC_2023], "de")  # the detector alone would read this as German and pass it
    assert "po_line_present" in run(german_line, meta, po_line_present=True, answer_language_matches="en")
    assert run(german_line, meta, po_line_present=True) == []


def test_grouped_citations_are_read_by_the_answer_checks():
    grouped = "Das gilt [1, 2] und [3,4]."
    assert citations_map_to_sources(grouped, {1, 2, 3, 4})
    assert not citations_map_to_sources(grouped, {1, 2, 3})  # [4] has no source
    assert not citations_map_to_sources("Das gilt [1][5].", {1, 2})
    assert evaluate_turn(grouped, [], [], {})["citations"] == [1, 2, 3, 4]


def test_the_line_checks_read_a_grouped_citation():
    meta = {1: BSC_2023, 2: VISA}
    answer = "Frist und Antrag [1, 2]."
    full = with_lines(answer, [BSC_2023, VISA])
    assert po_line_present(full, meta, TEXTS["de"], PROGRAMMES, "de")
    assert not po_line_present(answer, meta, TEXTS["de"], PROGRAMMES, "de")
    assert visa_disclaimer_present(full, meta, TEXTS["de"], PROGRAMMES, "de")
    assert not visa_disclaimer_present(answer, meta, TEXTS["de"], PROGRAMMES, "de")
