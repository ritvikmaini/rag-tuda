"""Languages per request on the TU profile (messages.de.yaml and messages.en.yaml): the texts that fire before the question is
read follow the page's `lang` parameter, whatever the question's language. Black box: HTTP and the fake model's request log."""

from rag_testing import harness

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
