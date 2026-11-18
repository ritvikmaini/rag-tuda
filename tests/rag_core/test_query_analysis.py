"""Query analysis."""

from rag_core.domain import query_analysis as qa
import json


def test_json_answer():
    a = qa.parse_query_analysis(
        '{"standalone_question": "Hat das Bürgerbüro am Samstag geöffnet?", "intent": "lookup", "search_queries": ["Bürgerbüro Öffnungszeiten Samstag"], "language": "Deutsch", "date_from": null, "date_to": null}',
        "Und am Samstag?",
    )
    assert a.parsed == "json" and a.standalone_question.startswith("Hat das Bürgerbüro")
    assert qa.retrieval_queries(a, "Und am Samstag?", first_turn=False) == [
        "Hat das Bürgerbüro am Samstag geöffnet?",
        "Bürgerbüro Öffnungszeiten Samstag",
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
        '{"standalone_question": "Welche Homeoffice-Regelung der Stadt ist die aktuellste?", "intent": "lookup", "search_queries": ["Dienstvereinbarung mobile Arbeit 2024"]}',
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
