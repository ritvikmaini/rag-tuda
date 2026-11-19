"""The answer checks of the TU profile, run on hand-written answers: a German and an English one for each case, with [n]
markers and the lines code appends. The expected lines come from the profile's messages through the producer
(rag_core.domain.notes), so a change of wording moves producer and check together."""

import pathlib

import pytest

from evals import checks
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
