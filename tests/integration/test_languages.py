"""Languages per request on the TU profile (messages.de.yaml and messages.en.yaml): the texts that fire before the question is
read follow the page's `lang` parameter, whatever the question's language. Black box: HTTP and the fake model's request log."""

import uuid
from typing import Any

import psycopg
from rag_testing import harness
from rag_testing.minisite import pages

EN_BUSY = "Many requests have arrived at once. Please try again in a moment."
EN_TIMEOUT = "The answer is taking unusually long, so I stopped. Please ask your question again in a few minutes."
EN_TOO_LONG = (
    "Your message is too long for me (at most 2.000 characters). Please shorten your question to the essentials or ask "
    "several shorter questions one after another, and I will gladly help."
)
SLOW = "eins zwei drei vier fünf sechs sieben acht [1]."


def test_guard_texts_follow_page_lang(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    # too long: the page's language, German without one or with a language the page does not offer
    for lang, text in (({"lang": "en"}, EN_TOO_LONG), ({"lang": "de"}, harness.TOO_LONG), ({}, harness.TOO_LONG)):
        assert assistant.ask("x" * 2001, user, **lang).frames == [{"type": "chunk", "content": text}]
    assert assistant.ask("x" * 2001, user, lang="fr").text == harness.TOO_LONG
    assert llm.calls() == []
    # busy: the one slot is held by a German question; the English page's request is told so in English, its error frame too
    llm.script({"purpose": "answer", "match": harness.THESIS, "reply": SLOW, "token_delay_s": 0.8})
    thread, out = harness.in_thread(lambda: assistant.ask(harness.THESIS, user))
    harness.wait_until(lambda: bool(llm.calls("answer")))
    busy = assistant.ask(harness.OFFICE_HOURS, user + "-b", lang="en")
    thread.join()
    assert busy.frames == [{"type": "error", "content": EN_BUSY}, {"type": "chunk", "content": EN_BUSY}]
    assert "error" not in out and out["result"].path == harness.LOOKUP
    # timeout: the model stays silent past the card's idle limit; the English page gets the English text after the error
    llm.reset()
    llm.script({"purpose": "answer", "first_delay_s": 8})
    timed_out = assistant.ask(harness.THESIS, user + "-t", lang="en")
    assert timed_out.first("error") == "timeout" and timed_out.text == EN_TIMEOUT
    llm.reset()
    llm.script({"purpose": "answer", "first_delay_s": 8})
    assert assistant.ask(harness.THESIS, user + "-u").text == harness.MSG_TIMEOUT


FACULTY, UNIVERSITY = "https://www.informatik.tu-darmstadt.de", "https://www.intern.tu-darmstadt.de"
APB_ORIGINAL = UNIVERSITY + pages.APB_ORIGINAL_PDF  # the general examination regulation: German only
RETAKE_EN = "How many times can I retake a failed exam?"
ENGLISH_ANALYSIS = {
    "purpose": "analysis",
    "match": RETAKE_EN,
    "reply": {"language": "en", "search_queries": ["Wiederholung nicht bestandene Prüfung"]},
}
ENGLISH_ANSWER = {"purpose": "answer", "match": RETAKE_EN, "reply": "You may retake a failed exam twice [1]."}
EN_REMINDER = "Answer in English only: the whole answer, although the sources are German."


def sent_to_the_model(llm: harness.FakeLLM) -> list[tuple[str, str]]:
    """The messages of the newest call that writes an answer (with sources or without)."""
    return harness.messages([c for c in llm.calls() if c["purpose"] in ("answer", "no_docs")][-1])


def test_english_answer_over_german_sources(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(ENGLISH_ANALYSIS, ENGLISH_ANSWER)
    reply = assistant.ask(RETAKE_EN, user)
    assert reply.path == harness.LOOKUP  # the German-only rule is found: no decline for want of English text
    assert APB_ORIGINAL in {d["metadata"]["url"] for d in reply.first("docs")}
    assert llm.calls("no_docs") == []
    assert reply.text == "You may retake a failed exam twice [1]."
    sent = sent_to_the_model(llm)
    assert "Antworte in: Englisch" in sent[0][1]
    context = sent[-1][1]  # the German sources, framed in English
    assert "There are " in context and "Link: " + APB_ORIGINAL in context and "Es liegen" not in context
    assert context.endswith(EN_REMINDER)


def test_language_values_are_de_en(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str, db: psycopg.Connection[Any]
) -> None:
    def stored(turn: str) -> str:
        row = db.execute("SELECT sections->>'language' FROM chat.turn_context WHERE turn_id = %s", (turn,)).fetchone()
        return str(row and row[0])

    llm.script(ENGLISH_ANALYSIS, ENGLISH_ANSWER)
    german, english, turns = harness.THESIS, RETAKE_EN, [str(uuid.uuid4()) for _ in range(3)]
    assistant.ask(german, user, turn_id=turns[0])
    assistant.ask(english, user + "-e", turn_id=turns[1])
    assert (stored(turns[0]), stored(turns[1])) == ("de", "en")
    # a name instead of the code is no language: the answer's language comes from the question, nothing odd is stored
    llm.reset()
    llm.script({**ENGLISH_ANALYSIS, "reply": {**ENGLISH_ANALYSIS["reply"], "language": "Englisch"}}, ENGLISH_ANSWER)
    reply = assistant.ask(english, user + "-n", turn_id=turns[2])
    assert stored(turns[2]) == "" and sent_to_the_model(llm)[-1][1].endswith(EN_REMINDER)
    assert reply.text == "You may retake a failed exam twice [1]."


def test_detector_ties_to_german(ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str) -> None:
    # the analysis fails, so the detector reads the question: English stopwords outnumber German ones, a tie is German
    tie = "Bachelor thesis duration"  # no stopword of either language
    llm.script(
        {"purpose": "analysis", "match": RETAKE_EN, "fail": 500},
        {"purpose": "analysis", "match": tie, "fail": 500},
    )
    assistant.ask(RETAKE_EN, user)
    english = sent_to_the_model(llm)
    assert "Antworte in: Englisch" in english[0][1] and english[-1][1].endswith(EN_REMINDER)
    assistant.ask(tie, user + "-t")
    german = sent_to_the_model(llm)
    assert "Antworte in: Deutsch" in german[0][1] and EN_REMINDER not in german[-1][1]
