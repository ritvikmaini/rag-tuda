"""The hard filter of a question: the readers of explicit text, the chips, the term-to-PO parser, the analysis enums and the
previous turn's filter, in that order, over roles passed as plain data."""

from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from rag_core.domain.cohort import po_for_term
from rag_core.domain.filters import Chips, Filter, check_chips, resolve_filter
from rag_core.domain.query_analysis import QueryAnalysis, parse_query_analysis
from rag_core.settings import load_profile
from rag_core.settings.profile import Roles

TODAY = date(2026, 11, 19)
REAL_ROLES = ["profiles", "tu-informatik", "roles.yaml"]


def version(year: int, start_from: str | None, start_to: str | None) -> dict:
    return {"year": year, "start_from": start_from, "start_to": start_to}


ROLES = {
    "programmes": {
        "bsc-informatik": {
            "label_de": "B.Sc. Informatik",
            "label_en": "B.Sc. Informatik",
            "degree": "bachelor",
            "po_versions": [version(2023, "WiSe 2023/24", None), version(2015, "WiSe 2015/16", "SoSe 2023")],
        },
        "msc-informatik": {
            "label_de": "M.Sc. Informatik",
            "label_en": "M.Sc. Informatik",
            "degree": "master",
            "po_versions": [version(2023, "WiSe 2023/24", None), version(2015, None, None)],
        },
        "msc-computer-science": {
            "label_de": "M.Sc. Computer Science",
            "label_en": "M.Sc. Computer Science",
            "degree": "master",
            "po_versions": [version(2023, "WiSe 2023/24", None)],
        },
    },
    "groups": {"promotion": {}, "bachelor": {}, "master": {}},
}


def resolve(text: str, chips=None, analysis=None, previous=None) -> Filter:
    return resolve_filter(text, chips, analysis, previous, ROLES, today=TODAY)


def test_explicit_po_in_text() -> None:
    assert resolve("Was gilt in der PO 2015 für die Abmeldung?").po_version == (2015,)
    assert resolve("Was steht in der Prüfungsordnung von 2015?").po_version == (2015,)
    assert resolve("Unterschied zwischen PO 2015 und PO 2023?").po_version == (2015, 2023)
    assert resolve("Differences of PO 2015 vs 2023").po_version == (2015, 2023)
    assert resolve("What do the examination regulations 2023 say?").po_version == (2023,)
    assert resolve("Wie geht der PO-Wechsel?").po_version is None
    assert resolve("Was gilt in der PO 2019?").po_version is None  # a year no programme has a PO for
    assert resolve("Anmeldung für 2023").po_version is None  # a year alone is no version


def test_text_beats_chip_for_this_question() -> None:
    chips = Chips(degree="master", programme="msc-informatik", po=2023)
    both = resolve("Was unterscheidet PO 2015 und PO 2023 beim Master Informatik?", chips)
    assert both.po_version == (2015, 2023) and both.programme == ("msc-informatik",)
    other = resolve("Und im M.Sc. Computer Science nach PO 2023?", chips)
    assert other.programme == ("msc-computer-science",) and other.po_version == (2023,)
    # the chip stays: the next question that names nothing gets the chip's values, not the text's of the last one
    silent = resolve("Wie melde ich mich ab?", chips, previous=both)
    assert (silent.degree, silent.programme, silent.po_version) == (("master",), ("msc-informatik",), (2023,))


def test_chip_used_when_text_silent() -> None:
    chips = Chips(degree="master", programme="msc-informatik", po=2015)
    found = resolve("Wie melde ich mich ab?", chips)
    assert (found.degree, found.programme, found.po_version) == (("master",), ("msc-informatik",), (2015,))
    assert resolve("Wie melde ich mich ab?", Chips(degree="promotion")).degree == ("promotion",)
    # a chip beats the term-to-PO parser and the analysis
    analysis = SimpleNamespace(degree=["bachelor"], po_version=[2023], enrolment_term="WiSe 2024/25")
    assert resolve("Und wie melde ich mich ab?", Chips(po=2015), analysis).po_version == (2015,)


def test_start_term_maps_to_po_from_roles() -> None:
    assert po_for_term("bsc-informatik", "WiSe 2019/20", ROLES) == 2015
    assert po_for_term("bsc-informatik", "SoSe 2023", ROLES) == 2015
    assert po_for_term("bsc-informatik", "WiSe 2023/24", ROLES) == 2023
    assert po_for_term("bsc-informatik", "WiSe 2030", ROLES) == 2023  # open end: the newest version in force
    assert po_for_term("bsc-informatik", "WiSe 2014/15", ROLES) is None  # before any stated start
    assert po_for_term("msc-informatik", "WiSe 2019/20", ROLES) is None  # PO 2015 has no boundaries: never a guess
    assert po_for_term("msc-informatik", "WiSe 2024/25", ROLES) == 2023
    assert (
        po_for_term("unknown", "WiSe 2024/25", ROLES) is None and po_for_term("bsc-informatik", "bald", ROLES) is None
    )

    text = "Ich studiere B.Sc. Informatik seit WiSe 2019/20, was gilt für meine Prüfungsanmeldung?"
    assert resolve(text).po_version == (2015,)
    # the analysis' enrolment term works without the cue words; the programme can come from a chip
    assert resolve(
        "Wie melde ich mich ab?", Chips(programme="bsc-informatik"), SimpleNamespace(enrolment_term="WiSe 2024/25")
    ).po_version == (2023,)
    # a term read as the start of the studies is no date, a PO that cannot be derived is not guessed
    assert resolve(text).date is None
    assert resolve("Ich studiere M.Sc. Informatik seit WiSe 2019/20").po_version is None
    assert resolve("Ich studiere seit WiSe 2019/20, was gilt?").po_version is None  # no programme known
    # a term without an enrolment cue is a date for the rules in force
    asked = resolve("Welche Fristen gelten ab WiSe 2026/27 für den B.Sc. Informatik?")
    assert asked.po_version is None and asked.date == date(2026, 10, 1)


def test_previous_filter_carried_when_nothing_named() -> None:
    previous = Filter(("master",), ("msc-informatik",), (2015,), ("mhb",), date(2020, 4, 1), True, "text")
    kept = replace(previous, po_source="carried")
    assert resolve("Und wann ist die Frist?", previous=previous) == kept
    assert (
        resolve("Und wann ist die Frist?", None, QueryAnalysis("Und wann ist die Frist?", parsed="fallback"), previous)
        == kept
    )
    # a new value replaces its field, the others stay; regulation and date go with the old question
    changed = resolve("Und nach PO 2023?", previous=previous)
    assert (changed.degree, changed.programme, changed.po_version) == (("master",), ("msc-informatik",), (2023,))
    assert changed.regulation is None and changed.date is None
    assert resolve("Und im Modulhandbuch?", previous=Filter(programme=("msc-informatik",))).regulation == ("mhb",)
    assert resolve("Und wann ist die Frist?") == Filter()


def test_archived_lifted_when_po_set() -> None:
    assert resolve("Wie melde ich mich ab?").include_archived is False
    assert resolve("Was gilt in der PO 2015?").include_archived is True
    assert resolve("Wie melde ich mich ab?", Chips(po=2023)).include_archived is True
    assert resolve("Was stand in der alten Prüfungsordnung zur Abmeldung?").include_archived is True
    assert resolve("What did the previous version of the regulations say?").include_archived is True
    assert resolve("Wie alt darf ich sein?").include_archived is False


def test_readers_degree_programme_regulation_date() -> None:
    found = resolve("Wie viele CP brauche ich im Master Computer Science laut Modulhandbuch?")
    assert found.degree == ("master",) and found.programme == ("msc-computer-science",) and found.regulation == ("mhb",)
    assert resolve("Gilt das für Bachelor und Master?").degree == ("bachelor", "master")
    assert resolve("Ich bin Doktorand, was gilt?").degree == ("promotion",)
    assert resolve("Ich schreibe die Bachelorarbeit im B. Sc. Informatik").programme == ("bsc-informatik",)
    assert resolve("Wie viel Zeit bleibt laut APB und laut PO 2015?").regulation == ("apb", "po")
    assert resolve("Wie geht der PO-Wechsel?").regulation is None  # a document type is named with a preposition
    assert resolve("Wo finde ich das Modulhandbuch?").regulation is None
    assert resolve("Was war am 12.03.2019?").date == date(2019, 3, 12)
    assert resolve("Wie viele Master gibt es?").programme is None


def test_analysis_enums_fill_what_nothing_else_names() -> None:
    analysis = SimpleNamespace(
        degree=["master"], programme=["msc-informatik", "nope"], po_version=[2015, 1999], regulation=["spp", "x"]
    )
    found = resolve("Wie melde ich mich ab?", analysis=analysis)
    assert (found.degree, found.programme, found.po_version, found.regulation) == (
        ("master",),
        ("msc-informatik",),
        (2015,),
        ("spp",),
    )
    assert resolve("Was gilt in PO 2023?", analysis=analysis).po_version == (2023,)
    bare = QueryAnalysis("Wie melde ich mich ab?")  # without the arrays: nothing to read
    assert resolve("Wie melde ich mich ab?", analysis=bare) == Filter()


def test_a_value_that_contradicts_a_stronger_one_is_dropped() -> None:
    found = resolve("Wie viele CP im M.Sc. Informatik?", Chips(degree="bachelor", programme="bsc-informatik", po=2023))
    assert found.programme == ("msc-informatik",) and found.degree == (
        "master",
    )  # the chip's bachelor and programme lose
    assert found.po_version is None  # the chips' PO year went with the programme and degree that lost
    chip = resolve("Wie melde ich mich ab?", Chips(po=2015), SimpleNamespace(programme=["msc-computer-science"]))
    assert chip.programme is None and chip.po_version == (2015,)  # the analysis' programme has no PO 2015


def test_chips_are_checked_against_the_roles() -> None:
    assert check_chips(Chips(), ROLES) is None
    assert check_chips(Chips("master", "msc-informatik", 2015), ROLES) is None
    assert check_chips(Chips("promotion"), ROLES) is None
    assert "degree" in (check_chips(Chips("doktor"), ROLES) or "")
    assert "programme" in (check_chips(Chips(programme="nope"), ROLES) or "")
    assert "po" in (check_chips(Chips(po=1999), ROLES) or "")
    assert "po" in (check_chips(Chips(programme="msc-computer-science", po=2015), ROLES) or "")
    assert "not a bachelor" in (check_chips(Chips("bachelor", "msc-informatik"), ROLES) or "")
    assert check_chips(Chips(degree="master"), {"programmes": {}}) is not None


def test_the_shipped_roles_have_no_guessed_boundaries() -> None:
    roles = load_profile(Roles, Path(__file__).resolve().parents[2].joinpath(*REAL_ROLES)).model_dump()
    assert po_for_term("bsc-informatik", "WiSe 2019/20", roles) is None
    assert po_for_term("msc-informatik", "WiSe 2019/20", roles) is None
    assert po_for_term("msc-informatik", "WiSe 2024/25", roles) == 2023
    assert resolve_filter("Seit WiSe 2024/25 im M.Sc. Informatik", None, None, None, roles).po_version == (2023,)


def test_a_po_year_follows_its_degree_and_programme() -> None:
    bachelor = resolve("Wie melde ich mich ab?", Chips("bachelor", "bsc-informatik", 2023))
    assert (bachelor.degree, bachelor.programme, bachelor.po_version) == (("bachelor",), ("bsc-informatik",), (2023,))
    # carried: a new degree drops the programme and the PO year of the old one
    asked = resolve("Und im Master?", previous=bachelor)
    assert (asked.degree, asked.programme, asked.po_version, asked.po_source) == (("master",), None, None, None)
    assert asked.include_archived is False
    # chips: the question names another degree, so the chips' programme and PO year do not apply to it
    chips = Chips("master", "msc-informatik", 2023)
    thesis = resolve("Kann ich meine Bachelorarbeit anrechnen?", chips)
    assert (thesis.degree, thesis.programme, thesis.po_version) == (("bachelor",), None, None)
    # a year that only a programme of another degree has a PO for is no filter
    only_bachelor = {"programmes": {k: v for k, v in ROLES["programmes"].items()}, "groups": {}}
    only_bachelor["programmes"]["bsc-informatik"] = {
        **ROLES["programmes"]["bsc-informatik"],
        "po_versions": [version(2009, None, None)],
    }
    assert (
        resolve_filter("Im Master?", None, None, Filter(po_version=(2009,)), only_bachelor, today=TODAY).po_version
        is None
    )
    assert resolve_filter(
        "Im Bachelor?", None, None, Filter(po_version=(2009,)), only_bachelor, today=TODAY
    ).po_version == (2009,)


def test_a_field_the_text_named_is_not_refilled_by_a_weaker_source() -> None:
    chips = Chips("master", "msc-informatik", 2023)
    found = resolve("Was gilt in PO 2015 für M.Sc. Computer Science?", chips)
    assert found.programme == ("msc-computer-science",) and found.po_version is None  # PO 2015 does not exist there
    assert (
        resolve("Was gilt in der PO 2019?", chips).po_version is None
    )  # an unknown year is not replaced by the chip's
    analysed = resolve("Was gilt in der PO 2019?", analysis=SimpleNamespace(po_version=[2023]))
    assert analysed.po_version is None


def test_po_years_in_more_spellings() -> None:
    for text in (
        "PO15",
        "PO 15",
        "PO (2015)",
        "PO-2015",
        "Prüfungsordnung aus dem Jahr 2015",
        "the exam regulations of 2015",
    ):
        assert resolve(text).po_version == (2015,), text
    assert resolve("PO 2015 und PO15 und Prüfungsordnung (2023)").po_version == (2015, 2023)
    assert resolve("Hat PO 99 eine Frist?").po_version is None


def test_seit_wann_is_a_question_not_the_start_of_the_studies() -> None:
    found = resolve("Seit wann gilt im M.Sc. Informatik die neue PO ab WiSe 2024/25?")
    assert found.po_version is None and found.date == date(2024, 10, 1)
    assert resolve("Since when does it apply to the M.Sc. Informatik in WiSe 2024/25?").po_version is None
    assert resolve("Ich bin im M.Sc. Informatik seit WiSe 2024/25.").po_version == (2023,)


def test_the_po_source_names_where_the_year_came_from() -> None:
    assert resolve("Wie melde ich mich ab?").po_source is None
    assert resolve("Was gilt in PO 2015?").po_source == "text"
    assert resolve("Wie melde ich mich ab?", Chips(po=2015)).po_source == "chip"
    assert resolve("Ich studiere M.Sc. Informatik seit WiSe 2024/25").po_source == "term"
    assert resolve("Wie melde ich mich ab?", analysis=SimpleNamespace(po_version=[2015])).po_source == "analysis"
    term = resolve("Ich studiere M.Sc. Informatik seit WiSe 2024/25")
    assert resolve("Und die Frist?", previous=term).po_source == "term"  # a term-derived PO stays "in der Regel"
    assert resolve("Und die Frist?", previous=resolve("Was gilt in PO 2015?")).po_source == "carried"


def test_a_filter_round_trips_through_json_with_its_tuples_and_its_date() -> None:
    import json

    full = Filter(("master",), ("msc-informatik",), (2015, 2023), ("po",), date(2026, 10, 1), True, "chip")
    assert Filter.from_json(json.loads(json.dumps(full.as_json()))) == full
    assert Filter.from_json(json.loads(json.dumps(Filter().as_json()))) == Filter()
    assert full.as_json()["date"] == "2026-10-01" and full.as_json()["po_version"] == [2015, 2023]
    assert Filter.from_json({}) == Filter()
    for bad in ({"degree": "master"}, {"date": "gestern"}):
        with pytest.raises((TypeError, ValueError)):
            Filter.from_json(bad)


def test_the_parsed_analysis_fills_the_filter() -> None:
    raw = (
        '{"standalone_question": "Was unterscheidet die Prüfungsordnungen?", "intent": "lookup", "queries_de": ["a"], '
        '"language": "de", "degree": ["master"], "programme": ["msc-informatik"], "po_version": [2015, 2023], '
        '"regulation": ["po"], "enrolment_term": null, "needs_more": true}'
    )
    analysis = parse_query_analysis(raw, "x")
    found = resolve("Was unterscheidet die beiden Fassungen?", analysis=analysis)
    assert (found.degree, found.programme, found.po_version, found.regulation) == (
        ("master",),
        ("msc-informatik",),
        (2015, 2023),
        ("po",),
    )
    assert found.po_source == "analysis"
    term = parse_query_analysis(
        '{"standalone_question": "x", "programme": ["bsc-informatik"], "enrolment_term": "WiSe 2023/24"}', "x"
    )
    assert resolve("Was gilt für mich?", analysis=term).po_version == (2023,)
    assert resolve("Was gilt für mich?", analysis=parse_query_analysis("kaputt", "x")) == Filter()
