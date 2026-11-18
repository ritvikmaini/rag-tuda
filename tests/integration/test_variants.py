"""Versions of the last answer: a variant enters at the
answer step (no analysis, no search), is always made from the original, keeps every version for a switch without the
model, puts the original back when it fails or is stopped, and is refused without a last answer with sources. A new
question clears the versions; a switch is answered while another question holds the slot."""

import uuid

import pytest
from rag_testing import harness

G = harness.THESIS
ORIGINAL = {"purpose": "answer", "match": G, "reply": "Die Bearbeitungszeit beträgt fünf Monate [1]."}
TABLE = {
    "purpose": "answer",
    "system": "Markdown-Tabelle",
    "reply": "Antwort.\n\n| Ort | Quelle |\n|---|---|\n| Hof | [1] |",
}


def ask_original(assistant: harness.Assistant, user: str) -> str:
    turn = str(uuid.uuid4())
    assert assistant.ask(G, user, turn_id=turn, formats="v1").path == harness.LOOKUP_FORMATS
    return turn


def variant(assistant: harness.Assistant, user: str, turn: str, name: str) -> harness.Reply:
    return assistant.ask(G, user, formats="v1", variant=name, variant_of=turn)


def select(assistant: harness.Assistant, user: str, turn: str, name: str) -> harness.Reply:
    return assistant.ask(G, user, formats="v1", variant_select=name, variant_of=turn)


def switched(name: str, ok: bool) -> list[dict[str, object]]:
    return [{"type": "variant", "content": {"name": name, "ok": ok}}]


def refused(name: str) -> list[dict[str, object]]:
    chunk: list[dict[str, object]] = [{"type": "chunk", "content": harness.REFUSED}]
    return chunk + switched(name, False)


def test_a_variant_enters_at_the_answer_step_and_both_versions_stay(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(TABLE, ORIGINAL)
    turn = ask_original(assistant, user)
    before = len(llm.calls())
    table = variant(assistant, user, turn, "table")
    made = llm.calls()[before:]
    assert [c["purpose"] for c in made] == ["answer"]  # no analysis: no search either
    assert "Markdown-Tabelle" in harness.messages(made[0])[0][1]
    assert table.path == harness.VARIANT and table.first("shape") == "table"
    assert table.first("variant") == {"name": "table", "ok": True}
    assert select(assistant, user, turn, "original").frames == switched("original", True)
    assert select(assistant, user, turn, "table").frames == switched("table", True)


def test_a_language_variant_takes_the_original_form_not_the_one_on_screen(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(ORIGINAL)
    turn = ask_original(assistant, user)
    assert variant(assistant, user, turn, "steps").first("shape") == "steps"
    english = variant(assistant, user, turn, "english")
    assert english.first("shape") == "text"
    sent = harness.messages(llm.calls("answer")[-1])
    assert sent[-1][1].endswith(harness.ENGLISH_REMINDER) and "nummerierte Schritte" not in sent[0][1]


def test_a_shape_variant_after_a_length_variant_takes_its_own_form(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(ORIGINAL)
    turn = ask_original(assistant, user)
    assert variant(assistant, user, turn, "shorter").first("shape") == "text"
    assert variant(assistant, user, turn, "mindmap").first("shape") == "mindmap"
    assert "```mindmap" in harness.messages(llm.calls("answer")[-1])[0][1]


@pytest.mark.parametrize("broken", [{"fail": 500}, {"reply": ""}], ids=["error", "empty"])
def test_a_failed_or_empty_variant_puts_the_original_back(
    broken: dict[str, object], ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(ORIGINAL)
    turn = ask_original(assistant, user)
    llm.script({"purpose": "answer", **broken})
    assert variant(assistant, user, turn, "shorter").first("variant") == {"name": "shorter", "ok": False}
    llm.script(ORIGINAL)
    assistant.ask(harness.OFFICE_HOURS, user)
    assert harness.history(llm, harness.OFFICE_HOURS) == harness.turn(G, "Die Bearbeitungszeit beträgt fünf Monate.")


def test_a_stopped_variant_puts_the_original_back(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(ORIGINAL)
    turn = ask_original(assistant, user)
    llm.script({"purpose": "answer", "reply": "eins zwei drei vier fünf [1].", "token_delay_s": 1.0})
    stopped = assistant.ask(G, user, stop_after="chunk", formats="v1", variant="shorter", variant_of=turn)
    assert stopped.path == harness.VARIANT[:3]
    llm.script(ORIGINAL)
    assistant.ask(
        harness.OFFICE_HOURS, user
    )  # waits for the slot, which the stopped variant frees after putting it back
    assert harness.history(llm, harness.OFFICE_HOURS) == harness.turn(G, "Die Bearbeitungszeit beträgt fünf Monate.")


def test_a_switch_needs_no_model_and_a_missing_version_is_refused(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(ORIGINAL)
    turn = ask_original(assistant, user)
    variant(assistant, user, turn, "shorter")
    before = len(llm.calls())
    assert select(assistant, user, turn, "original").frames == switched("original", True)
    assert select(assistant, user, turn, "table").frames == switched("table", False)
    assert select(assistant, user, str(uuid.uuid4()), "shorter").frames == switched("shorter", False)
    assert len(llm.calls()) == before


def test_variants_are_refused_without_a_last_answer_with_sources(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(harness.OUT_OF_SCOPE, ORIGINAL)
    joke = str(uuid.uuid4())
    assistant.ask(harness.JOKE, user, turn_id=joke, formats="v1")
    assert assistant.ask(harness.JOKE, user, formats="v1", variant="shorter", variant_of=joke).frames == refused(
        "shorter"
    )
    older = ask_original(assistant, user + "-2")
    newer = ask_original(assistant, user + "-2")
    assert variant(assistant, user + "-2", older, "shorter").frames == refused("shorter")  # not the last question
    assert variant(assistant, user + "-2", newer, "fancy").frames == refused("fancy")  # no such variant
    assert assistant.ask(G, user + "-2", formats="v1", variant="shorter").frames == refused("shorter")  # no turn id
    assert assistant.ask(G, user + "-2", variant="shorter", variant_of=newer).frames == refused(
        "shorter"
    )  # formats off


def test_a_new_question_clears_the_versions_and_keeps_the_version_on_screen(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(TABLE, ORIGINAL)
    first = ask_original(assistant, user)
    assert variant(assistant, user, first, "table").first("variant") == {"name": "table", "ok": True}
    second = ask_original(assistant, user)
    assert select(assistant, user, second, "table").frames == switched("table", False)
    assert "| Ort | Quelle |" in harness.history(llm, G)  # the version on screen stayed the answer the history keeps


def test_a_switch_and_a_refusal_are_answered_while_another_question_holds_the_slot(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(ORIGINAL)
    turn = ask_original(assistant, user)
    variant(assistant, user, turn, "shorter")
    slow = {
        "purpose": "answer",
        "match": harness.OFFICE_HOURS,
        "reply": "eins zwei drei vier fünf sechs [1].",
        "token_delay_s": 0.8,
    }
    llm.script(slow, ORIGINAL)
    thread, out = harness.in_thread(lambda: assistant.ask(harness.OFFICE_HOURS, user + "-other"))
    harness.wait_until(lambda: any(c["question"] == harness.OFFICE_HOURS for c in llm.calls("answer")))
    switch = select(assistant, user, turn, "original")
    refusal = assistant.ask(G, user, variant="shorter", variant_of=turn)  # formats off: refused before the slot
    thread.join()
    assert switch.frames == switched("original", True) and switch.seconds < 2
    assert refusal.frames == refused("shorter") and refusal.seconds < 2
    assert "error" not in out and out["result"].path == harness.LOOKUP
