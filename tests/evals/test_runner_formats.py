"""Probe runner support for response formats v1."""

import httpx
import pytest
from evals import checks
from evals import runner as rp

SAMPLES = [
    "Es gibt zwei Regelungen [1].\n\n| Gruppe | Urlaub |\n| --- | --- |\n| Beamte | 30 Tage [1] |\n| Tarif | 30 Tage |",
    "Vorher [1].\n\n- [ ] Antrag – Personalabteilung [1]\n- [ ] Elterngeld [2]\n- [x] Bescheinigung",
    "So geht es [1].\n\n1. Antrag [1]\n2. Genehmigung [2]\n3. Abrechnen",
    "Überblick [1].\n\n```mindmap\n# A\n## B\n- C [1]\n- D\n```",
    "Die Kernzeit endet um 15 Uhr [1].",
    "",
]


def test_shape_and_citations_are_checked():
    table = SAMPLES[0]
    ok = checks.evaluate_turn(table, [], [], {"shape": "table", "cited_min": 0.5}, {"shape": "table"})
    assert ok["auto_pass"] and ok["format"] == {"sent": "table", "produced": "table", "cited": [1, 2]}
    strict = checks.evaluate_turn(table, [], [], {"shape": "table", "cited_min": 0.8}, {"shape": "table"})
    assert strict["failed"] == ["cited"]
    wrong = checks.evaluate_turn(SAMPLES[4], [], [], {"shape": "table"}, {"shape": "table"})
    assert "shape" in wrong["failed"]
    unsent = checks.evaluate_turn(table, [], [], {"shape": "table", "cited_min": 0.5}, {})
    assert unsent["failed"] == ["shape"]


HEAD = "| Merkmal | Elektronisch | Papier |\n|---|---|---|\n| Antrag | Im Modul | Schriftlich |\n"


@pytest.mark.parametrize(
    ("table", "cited", "failed"),
    [
        (HEAD + "| Quelle | [5] | [2] |", [2, 2], []),
        (HEAD + "| Link | [5], [2] | [2] |", [2, 2], []),  # a source row is read by its content, not its label
        (HEAD + "| Stand | 2024 | 2024 |\n| Link | [5] | Nicht verfügbar |", [1, 3], ["cited"]),  # FV03 t6's defect
        (
            "| Merkmal | A | B |\n|---|---|---|\n| X | y | |\n| Link | [5] | Nicht verfügbar |",
            [1, 2],
            ["cited"],
        ),  # text: no source row
        (
            "| Merkmal | A | B |\n|---|---|---|\n| Dauer | 36 Monate | - |\n| Quelle | [6] | |",
            [2, 2],
            [],
        ),  # "-" needs none
        (HEAD + "| Quelle | [5] | |", [1, 2], ["cited"]),  # the Papier column has text and no source
        (HEAD + "\nQuellen: [5], [2]", [0, 1], ["cited"]),  # a line under the table is not a source row (FT04)
        ("| Wert |\n|---|\n| 1 [1] |\n| 2 |\n| 3 |", [1, 3], ["cited"]),  # a one-cell row is no source row
        (
            "| Tonne | Gebühr | Quelle |\n|---|---|---|\n| 120 l | 213 € | [6] |\n| 80 l | 107 € | – |",
            [1, 2],
            ["cited"],
        ),
    ],
)
def test_a_source_row_cites_its_columns(table, cited, failed):
    got = checks.evaluate_turn(table, [], [], {"shape": "table", "cited_min": 0.8}, {"shape": "table"})
    assert got["format"]["cited"] == cited and got["failed"] == failed


def test_variant_and_switch_turns():
    assert checks.evaluate_turn("x [1]", [], [], {"variant_ok": True}, {"variant_ok": True})["auto_pass"]
    assert checks.evaluate_turn("x [1]", [], [], {"variant_ok": True}, {"variant_ok": False})["failed"] == ["variant"]
    switch = checks.evaluate_turn("", [], [], {"variant_ok": True, "allow_empty": True}, {"variant_ok": True})
    assert switch["auto_pass"]
    assert checks.evaluate_turn("", [], [], {})["failed"] == ["empty_answer"]


def test_variants_refer_to_the_last_question_and_get_no_new_id():
    calls = []

    def ask_fn(question, user, opts):
        calls.append((question, dict(opts, meta=None)))
        opts["meta"]["shape"] = "table"
        opts["meta"]["variant_ok"] = True
        return "| a | b |\n| --- | --- |\n| 1 [1] | 2 [1] |", [], [], 0.1, 0.2, ["chunk"]

    conv = {
        "id": "FV9",
        "set": "variants",
        "turns": [
            {"q": "Frage"},
            {"variant": "table", "expect": {"variant_ok": True, "shape": "table"}},
            {"select": "original", "expect": {"variant_ok": True, "allow_empty": True}},
        ],
    }
    rows = rp.run_conversation(conv, ask_fn, 0.0, "t", "20261003-120000", lambda r: None, formats="v1")
    first = calls[0][1]
    assert first["formats"] == "v1" and "variant" not in first
    assert (
        calls[1][0] == "Frage" and calls[1][1]["variant"] == "table" and calls[1][1]["variant_of"] == first["turn_id"]
    )
    assert calls[2][1]["variant_select"] == "original" and calls[2][1]["variant_of"] == first["turn_id"]
    assert "turn_id" not in calls[1][1]
    assert [r["variant"] for r in rows] == [None, "table", "original"] and rows[1]["checks"]["auto_pass"]
    assert rows[1]["shape"] == "table"


def test_ask_sends_the_format_parameters_and_reads_the_new_events():
    seen = {}

    def handler(request):
        seen.update(dict(request.url.params))
        return httpx.Response(
            200,
            content=b'data: {"type": "shape", "content": "steps"}'
            b'data: {"type": "chunk", "content": "1. a"}'
            b'data: {"type": "citations", "content": [{"n": 1}, {"n": 3}, "x"]}'
            b'data: {"type": "variant", "content": {"name": "steps", "ok": true}}',
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    meta = {}
    answer, *_ = rp.ask(
        client, "http://b", "tok", "Frage?", "u", 10, formats="v1", variant="steps", variant_of="tid", meta=meta
    )
    assert seen["formats"] == "v1" and seen["variant"] == "steps" and seen["variant_of"] == "tid"
    assert meta == {"shape": "steps", "citations": [1, 3], "source_metadata": {}, "variant_ok": True}
    assert answer == "1. a"


# ---- answer language

GERMAN = "Sie stellen den Antrag bei der Personalabteilung [1]. Die Genehmigung erteilt die Abteilungsleitung, und die Reise wird abgerechnet."
ENGLISH = "You submit the application to the Personnel Department (Personalabteilung) [1]. The head of department approves it and the trip is settled."


def test_english_answers_are_checked_for_their_language():
    assert checks.looks_english(ENGLISH) and not checks.looks_english(GERMAN)
    assert not checks.looks_english("")
    ok = checks.evaluate_turn(ENGLISH, [], [], {"variant_ok": True, "language": "en"}, {"variant_ok": True})
    assert ok["auto_pass"]
    german = checks.evaluate_turn(GERMAN, [], [], {"variant_ok": True, "language": "en"}, {"variant_ok": True})
    assert german["failed"] == ["language"]
    # German titles in brackets, English tables and one-sentence answers are English
    english = [
        "Civil servants and employees both receive 30 days of annual leave [1].\n\n| Group | Days | Quelle |\n"
        "| --- | --- | --- |\n| Civil servants | 30 | [1] |\n| Employees | 30 | [1] |\n\nPart-time staff receive leave pro rata [2].",
        "Flexitime applies to part-time staff as well (Betriebsvereinbarung über die gleitende Arbeitszeit an der Universität "
        "Beispielstadt) [1]. Core time runs from 9 a.m. to 3 p.m. [1].",
        "Special leave is regulated in the agreement on working time (Betriebsvereinbarung über die Arbeitszeit und die "
        "Arbeitsbefreiung der Beschäftigten der Universität Beispielstadt) [1]. You apply via the Personnel Department (Personalabteilung) [1].",
        "The university's 2026 budget amounts to 372,914,508 euros [1].",
    ]
    for text in english:
        assert checks.looks_english(text), text
    for text in (
        "Das Gesamtvolumen beträgt 372.914.508 € [2].",
        "Die Kernzeit endet um 15 Uhr (Betriebsvereinbarung) [1].",
    ):
        assert not checks.looks_english(text), text


def test_every_english_variant_turn_gets_the_language_check():
    assert checks.turn_expect({"variant": "english", "expect": {"variant_ok": True}}, None) == {
        "variant_ok": True,
        "language": "en",
    }
    assert checks.turn_expect({"variant": "simple", "expect": {"variant_ok": True}}, None) == {"variant_ok": True}
    assert checks.turn_expect({"q": "Frage?"}, None) == {}


def test_rows_keep_the_checks_that_were_applied():
    conv = {
        "id": "FV9",
        "turns": [
            {"q": "Wie melde ich mich krank?"},
            {"variant": "english", "expect": {"variant_ok": True, "shape": "original"}},
        ],
    }

    def ask_fn(question, user, opts):
        opts["meta"].update({"shape": "steps", "variant_ok": True})
        return "1. Call in [1]\n2. Send the note to the office [1]", [], [], 1.0, 1.0, []

    rows = rp.run_conversation(conv, ask_fn, 0, "t", "s", lambda row: None)
    assert rows[1]["expect"] == {"variant_ok": True, "shape": "steps", "language": "en"}


def test_the_original_form_is_the_shape_of_the_last_question():
    turn = {"variant": "shorter", "expect": {"variant_ok": True, "shape": "original"}}
    assert checks.turn_expect(turn, "steps") == {"variant_ok": True, "shape": "steps"}
    assert checks.turn_expect(turn, None) == {"variant_ok": True, "shape": "text"}
