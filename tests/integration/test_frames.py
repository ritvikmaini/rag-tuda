"""Frame types, keys and order for every path on the real backend: lookup
with and without formats, direct reply, too long, busy, the guard's silence and deadline, a client stop. Every path is
compared whole, so a frame that is no longer sent (prompt_length_*) or a new one shows up."""

import time

from rag_testing import harness

SLOW = "eins zwei drei vier fünf sechs sieben acht [1]."


def test_a_lookup_sends_the_five_steps_its_sources_the_answer_and_citations(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    reply = assistant.ask(harness.THESIS, user)
    assert reply.status == 200 and reply.path == harness.LOOKUP
    assert [f["content"]["step"] for f in reply.frames if f["type"] == "status"] == list(harness.STEPS)
    docs = reply.first("docs")  # one per parent section: the three kept children can share a parent
    assert 1 <= len(docs) <= 14 and all(set(doc) == {"page_content", "metadata"} for doc in docs)


def test_with_formats_the_shape_follows_the_analysis(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script({"purpose": "analysis", "reply": {"answer_shape": "table"}})
    reply = assistant.ask(harness.THESIS, user, formats="v1")
    assert reply.path == harness.LOOKUP_FORMATS and reply.first("shape") == "table"


def test_a_direct_reply_has_no_search_and_no_sources(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(harness.OUT_OF_SCOPE)
    assert assistant.ask(harness.JOKE, user).path == harness.DIRECT
    with_formats = assistant.ask(harness.JOKE, user + "-f", formats="v1")
    assert with_formats.path == harness.DIRECT_FORMATS and with_formats.first("shape") == "text"


def test_a_question_over_the_limit_gets_one_chunk_and_the_limit_itself_passes(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    assert assistant.ask("x" * 2001, user).frames == [{"type": "chunk", "content": harness.TOO_LONG}]
    assert llm.calls() == []
    assert assistant.ask("x" * 2000, user + "-2").path == harness.LOOKUP


def test_a_question_while_the_slot_is_held_gets_busy(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script({"purpose": "answer", "match": harness.THESIS, "reply": SLOW, "token_delay_s": 0.8})
    thread, out = harness.in_thread(lambda: assistant.ask(harness.THESIS, user))
    harness.wait_until(lambda: bool(llm.calls("answer")))  # the first question holds the one slot
    busy = assistant.ask(harness.OFFICE_HOURS, user + "-b")
    thread.join()
    assert busy.frames == [
        {"type": "error", "content": harness.BUSY_ERROR},
        {"type": "chunk", "content": harness.MSG_BUSY},
    ]
    assert "error" not in out and out["result"].path == harness.LOOKUP


def test_silence_ends_the_answer_with_the_timeout_message(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script({"purpose": "answer", "first_delay_s": 8})
    reply = assistant.ask(harness.THESIS, user)
    assert reply.path == harness.LOOKUP[:6] + ["error", "chunk*"]
    assert reply.first("error") == "timeout" and reply.text == harness.MSG_TIMEOUT


def test_the_deadline_ends_a_slow_answer_with_the_timeout_message(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    deadline = assistant.ask(harness.THESIS, user + "-m").stream[0]["content"]["deadline_s"]
    llm.script({"purpose": "answer", "reply": " ".join(["Wort"] * (int(deadline / 1.5) + 10)), "token_delay_s": 1.5})
    reply = assistant.ask(
        harness.THESIS, user
    )  # the answer outlasts the card's deadline; the silence limit never fires
    assert reply.path == harness.LOOKUP[:7] + ["error", "chunk*"]
    assert reply.first("error") == "timeout" and reply.frames[-1]["content"] == "\n\n" + harness.MSG_TIMEOUT


def test_a_client_stop_ends_the_stream_and_frees_the_slot(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script({"purpose": "answer", "reply": SLOW, "token_delay_s": 1.0})
    stopped = assistant.ask(harness.THESIS, user, stop_after="chunk")
    assert stopped.path == harness.LOOKUP[:7]
    llm.script()
    started = time.monotonic()
    assert assistant.ask(harness.OFFICE_HOURS, user + "-next").path == harness.LOOKUP
    assert time.monotonic() - started < 10
