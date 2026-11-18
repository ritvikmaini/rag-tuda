"""The direct reply: a request about the conversation is
answered without a search, and the reply is told how many earlier turns it sees (conversation_state)."""

from rag_testing import harness

SUMMARY = "Fasse unser bisheriges Gespräch in drei Sätzen zusammen."
CONVERSATION = {"purpose": "analysis", "match": SUMMARY, "reply": {"intent": "conversation"}}


def test_a_first_turn_conversation_request_is_told_there_is_nothing_yet(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(CONVERSATION)
    assert assistant.ask(SUMMARY, user).path == harness.DIRECT
    system = harness.messages(llm.calls("direct")[-1])[0][1]
    assert harness.NO_TURNS in system and 'Die Nachricht wurde als "conversation" eingeordnet.' in system


def test_the_direct_reply_sees_the_earlier_turn_and_is_told_so(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(CONVERSATION, harness.ANSWER_ONE)
    assistant.ask(harness.THESIS, user)
    assert assistant.ask(SUMMARY, user).path == harness.DIRECT
    sent = harness.messages(llm.calls("direct")[-1])
    assert harness.ONE_TURN in sent[0][1]
    assert sent[1:3] == [("user", harness.THESIS), ("assistant", "Antwort eins.")]
