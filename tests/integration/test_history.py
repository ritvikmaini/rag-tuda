"""What the model sees of the conversation: completed
turns only, without citation marks; failed and stopped questions left out; an edit or a retry cuts the stored
conversation, also before an early exit; the page's kept questions place the cut of a question that was never stored;
invalid ids are ignored. Read from the fake LLM's request log."""

import uuid

from rag_testing import harness

G, L, F = harness.THESIS, harness.OFFICE_HOURS, harness.RETAKE
FAIL_OFFICE_HOURS = {"purpose": "answer", "match": L, "fail": 500}


def new_id() -> str:
    return str(uuid.uuid4())


def test_a_follow_up_sees_the_completed_turn_without_its_citation_marks(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(harness.ANSWER_ONE)
    assistant.ask(G, user, turn_id=new_id())
    assistant.ask(L, user, turn_id=new_id())
    assert harness.history(llm, G) == harness.NO_HISTORY
    assert harness.history(llm, L) == harness.turn(G, "Antwort eins.")
    answer = [c for c in llm.calls("answer") if c["question"] == L][-1]
    assert harness.messages(answer)[1:3] == [("user", G), ("assistant", "Antwort eins.")]


def test_a_failed_turn_is_left_out(ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str) -> None:
    llm.script(FAIL_OFFICE_HOURS, harness.ANSWER_ONE)
    assistant.ask(G, user, turn_id=new_id())
    assert "error" in assistant.ask(L, user, turn_id=new_id()).types
    assistant.ask(F, user, turn_id=new_id())
    assert harness.history(llm, F) == harness.turn(G, "Antwort eins.")


def test_a_stopped_turn_is_left_out(ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str) -> None:
    slow = {"purpose": "answer", "match": L, "reply": "eins zwei drei vier fünf [1].", "token_delay_s": 1.0}
    llm.script(slow, harness.ANSWER_ONE)
    assistant.ask(G, user, turn_id=new_id())
    assistant.ask(L, user, turn_id=new_id(), stop_after="chunk")
    assistant.ask(F, user, turn_id=new_id())
    assert harness.history(llm, F) == harness.turn(G, "Antwort eins.")


def test_an_edit_cuts_the_conversation_at_the_edited_question(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(harness.ANSWER_ONE)
    first = new_id()
    assistant.ask(G, user, turn_id=first)
    assistant.ask(L, user, turn_id=new_id())
    assistant.ask(F, user, turn_id=new_id(), replace_from=first)
    assert harness.history(llm, F) == harness.NO_HISTORY


def test_a_retry_replaces_the_failed_turn_without_a_duplicate(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(FAIL_OFFICE_HOURS, harness.ANSWER_ONE)
    assistant.ask(G, user, turn_id=new_id())
    failed = new_id()
    assistant.ask(L, user, turn_id=failed)
    llm.script(harness.ANSWER_ONE, {"purpose": "answer", "match": L, "reply": "Antwort zwei [1]."})
    assistant.ask(L, user, turn_id=new_id(), replace_from=failed)
    assistant.ask(F, user, turn_id=new_id())
    assert harness.history(llm, F) == harness.turn(G, "Antwort eins.") + "\n\n" + harness.turn(L, "Antwort zwei.")


def test_the_kept_questions_place_the_cut_of_a_question_never_stored(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(harness.ANSWER_ONE)
    first, too_long = new_id(), new_id()
    assistant.ask(G, user, turn_id=first)
    assert assistant.ask("x" * 2001, user, turn_id=too_long).text == harness.TOO_LONG  # never stored
    assistant.ask(L, user, turn_id=new_id())
    assistant.ask(F, user, turn_id=new_id(), replace_from=too_long, kept=first)
    assert harness.history(llm, F) == harness.turn(G, "Antwort eins.")


def test_the_cut_comes_before_the_too_long_reply(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(harness.ANSWER_ONE)
    first = new_id()
    assistant.ask(G, user, turn_id=first)
    assert assistant.ask("x" * 2001, user, turn_id=new_id(), replace_from=first).text == harness.TOO_LONG
    assistant.ask(L, user, turn_id=new_id())
    assert harness.history(llm, L) == harness.NO_HISTORY


def test_invalid_ids_are_ignored(ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str) -> None:
    llm.script(harness.ANSWER_ONE)
    assistant.ask(G, user, turn_id=new_id())
    reply = assistant.ask(L, user, turn_id="keine-id", replace_from="auch-keine", kept="a,b")
    assert reply.path == harness.LOOKUP
    assert harness.history(llm, L) == harness.turn(G, "Antwort eins.")
