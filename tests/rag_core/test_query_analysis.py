"""Query analysis."""

from rag_core.domain import query_analysis as qa
import json


def test_json_answer():
    a = qa.parse_query_analysis(
        '{"standalone_question": "Hat das Studienbüro am Samstag geöffnet?", "intent": "lookup", "search_queries": ["Studienbüro Öffnungszeiten Samstag"], "language": "de", "date_from": null, "date_to": null}',
        "Und am Samstag?",
    )
    assert a.parsed == "json" and a.standalone_question.startswith("Hat das Studienbüro")
    assert qa.retrieval_queries(a, "Und am Samstag?", first_turn=False) == [
        "Hat das Studienbüro am Samstag geöffnet?",
        "Studienbüro Öffnungszeiten Samstag",
    ]


def test_fenced_json_with_think_and_bad_values():
    raw = '<think>hmm {"x":1}</think>\n```json\n{"standalone_question": "", "intent": "WEATHER", "search_queries": "x", "date_from": "morgen"}\n```'
    a = qa.parse_query_analysis(raw, "Gibt es heute eine Wetterwarnung?")
    assert a.intent == "lookup" and a.standalone_question == "Gibt es heute eine Wetterwarnung?"
    assert a.search_queries == [] and a.date_from is None


def test_a_bare_list_falls_back_to_the_question():
    q = "Welche Regelungen gibt es zum Homeoffice?"
    a = qa.parse_query_analysis('["Welche Regelungen gibt es zum Homeoffice?", "Homeoffice rechtliche Vorgaben"]', q)
    assert a.parsed == "fallback" and a.intent == "lookup"
    assert qa.retrieval_queries(a, q, True) == [q]


def test_garbage_falls_back():
    a = qa.parse_query_analysis("Entschuldigung, das kann ich nicht.", "Was kostet ein Ausweis?")
    assert a.parsed == "fallback" and qa.retrieval_queries(a, "Was kostet ein Ausweis?", False) == [
        "Was kostet ein Ausweis?"
    ]


def test_non_retrieval_intents_and_situation_cap():
    for intent in ("conversation", "about_bot", "urgent", "out_of_scope"):
        a = qa.parse_query_analysis('{"standalone_question": "x", "intent": "%s", "search_queries": []}' % intent, "x")
        assert a.intent not in qa.RETRIEVAL_INTENTS
    s = qa.parse_query_analysis(
        '{"standalone_question": "Hund", "intent": "situation", "search_queries": ["a","b","c","d","e"]}', "Hund"
    )
    assert s.intent in qa.RETRIEVAL_INTENTS and len(s.search_queries) == 4


def test_raw_followup_not_searched_after_first_turn():
    a = qa.parse_query_analysis(
        '{"standalone_question": "Welche Homeoffice-Regelung der Universität ist die aktuellste?", "intent": "lookup", "search_queries": ["Betriebsvereinbarung mobile Arbeit 2024"]}',
        "Welche davon ist die aktuellste?",
    )
    assert "Welche davon ist die aktuellste?" not in qa.retrieval_queries(
        a, "Welche davon ist die aktuellste?", first_turn=False
    )


def test_fill_template_keeps_json_braces():
    t = 'Heute ist {{current_date}}. Format: {"intent": "..."} Frage: {{ question }} {{unknown}}'
    assert (
        qa.fill_template(t, {"current_date": "24.09.2026", "question": "Q"})
        == 'Heute ist 24.09.2026. Format: {"intent": "..."} Frage: Q {{unknown}}'
    )


def test_answer_shape_from_the_analysis():
    def shape(intent, value):
        raw = json.dumps({"standalone_question": "x", "intent": intent, "search_queries": ["x"], "answer_shape": value})
        return qa.parse_query_analysis(raw, "x").answer_shape

    assert shape("lookup", "table") == "table" and shape("find_document", "steps") == "steps"
    assert shape("lookup", "checklist") == "text"  # only the code chooses a checklist
    assert shape("lookup", "TABLE ") == "table" and shape("lookup", None) == "text" and shape("lookup", 3) == "text"
    assert shape("situation", "table") == "checklist" and shape("situation", None) == "checklist"
    assert shape("conversation", "table") == "text" and shape("out_of_scope", "steps") == "text"
    assert qa.parse_query_analysis("kaputt", "x").answer_shape == "text"
    assert qa.parse_query_analysis('["a", "b"]', "x").answer_shape == "text"


def full(**fields):
    base = {
        "reasoning": "r",
        "language": "de",
        "standalone_question": "Wie melde ich mich ab?",
        "intent": "lookup",
        "queries_de": ["Prüfungsrücktritt Frist", "Abmeldung Prüfung"],
        "queries_en": [],
        "degree": None,
        "programme": None,
        "po_version": None,
        "regulation": None,
        "enrolment_term": None,
        "date_range": None,
        "shape": "text",
        "needs_more": False,
    }
    return json.dumps(base | fields)


def test_the_schema_fields_are_read():
    a = qa.parse_query_analysis(
        full(
            degree=["master"],
            programme=["msc-informatik"],
            po_version=[2015, "2023", 1999.5, True, "x"],
            regulation=["po", " MHB "],
            enrolment_term="WiSe 2019/20",
            date_range={"from": "2026-11-19", "to": "2026-11-23"},
            shape="steps",
            needs_more=True,
        ),
        "x",
    )
    assert a.parsed == "json" and a.search_queries == ["Prüfungsrücktritt Frist", "Abmeldung Prüfung"]
    assert (a.degree, a.programme, a.po_version, a.regulation) == (
        ["master"],
        ["msc-informatik"],
        [2015, 2023],
        ["po", "mhb"],
    )
    assert (a.enrolment_term, a.date_from, a.date_to) == ("WiSe 2019/20", "2026-11-19", "2026-11-23")
    assert a.answer_shape == "steps" and a.needs_more is True and a.language == "de"
    empty = qa.parse_query_analysis(full(), "x")
    assert empty.degree is None and empty.po_version is None and empty.enrolment_term is None and not empty.needs_more
    assert qa.parse_query_analysis(full(degree=[], po_version="2015", needs_more="yes"), "x").degree is None
    assert qa.parse_query_analysis(full(po_version="2015"), "x").po_version is None  # an array or nothing
    assert qa.parse_query_analysis(full(needs_more="yes"), "x").needs_more is False
    bad = qa.parse_query_analysis(full(date_range={"from": "morgen", "to": 3}, enrolment_term=7), "x")
    assert bad.date_from is None and bad.date_to is None and bad.enrolment_term is None
    assert qa.parse_query_analysis(full(date_range="heute"), "x").date_from is None


def test_language_values_are_de_en():
    assert qa.parse_query_analysis(full(language="en"), "x").language == "en"
    assert qa.parse_query_analysis(full(language=" DE "), "x").language == "de"
    # a value that is neither code (a name, a language the page does not offer) is no answer: the detector decides later
    for value in ("Englisch", "English", "klingonisch?", "", None, 7):
        assert qa.parse_query_analysis(full(language=value), "x").language == "", value
    assert qa.parse_query_analysis("kein JSON", "x").language == ""
    assert qa.analysis_schema(None)["properties"]["language"]["enum"] == ["de", "en"]


def test_the_base_shape_still_reads():
    a = qa.parse_query_analysis(
        '{"standalone_question": "x", "intent": "lookup", "search_queries": ["a"], "language": "en", '
        '"date_from": "2026-11-19", "date_to": null, "answer_shape": "table"}',
        "x",
    )
    assert (a.search_queries, a.language, a.date_from, a.answer_shape) == (["a"], "en", "2026-11-19", "table")
    assert a.queries_en == [] and a.degree is None and a.needs_more is False


def test_queries_are_capped_per_intent():
    de = ["a", "b", "c", "d", "e"]
    lookup = qa.parse_query_analysis(full(queries_de=de, queries_en=["f", "g", "h"], language="en"), "x")
    assert lookup.search_queries == ["a", "b", "c"] and lookup.queries_en == ["f", "g"]
    assert len(qa.parse_query_analysis(full(queries_de=de, intent="situation"), "x").search_queries) == 4


def test_german_queries_always_english_ones_for_english_questions():
    q = "How do I withdraw from an exam?"
    english = qa.parse_query_analysis(
        full(
            language="en",
            standalone_question=q,
            queries_de=["Prüfungsrücktritt", "Abmeldung Prüfung"],
            queries_en=["exam withdrawal"],
        ),
        q,
    )
    # German first, English interleaved, so a small budget keeps one of each
    assert qa.retrieval_queries(english, q, first_turn=False) == [
        q,
        "Prüfungsrücktritt",
        "exam withdrawal",
        "Abmeldung Prüfung",
    ]
    assert qa.retrieval_queries(english, q, first_turn=False)[:3] == [q, "Prüfungsrücktritt", "exam withdrawal"]
    german = qa.parse_query_analysis(
        full(queries_en=["exam withdrawal"]), "x"
    )  # English queries of a German question are unused
    assert "exam withdrawal" not in qa.retrieval_queries(german, "x", first_turn=True)


def test_odd_digits_in_po_versions_never_raise():
    a = qa.parse_query_analysis(full(po_version=["²", "２０２３", "2023", " 2015 ", "20233", "١٢٣٤"]), "x")
    assert a.parsed == "json" and a.po_version == [2023, 2015]
    assert qa.parse_query_analysis(full(po_version=["²"]), "x").po_version is None
