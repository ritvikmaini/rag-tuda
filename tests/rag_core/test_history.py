"""Conversation history policy."""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from rag_core.domain import history as ch  # noqa: E402
from rag_core.domain.retrieval import section_identity

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
U1 = "0b0e7c9a-4f5e-4a1b-9c3d-2e1f0a9b8c7d"
U2 = "1c1f8d0b-5a6f-4b2c-8d4e-3f2a1b0c9d8e"


def q(text, mid=None, hours_ago=1.0):
    return HumanMessage(content=text, id=mid, additional_kwargs={"ts": (NOW - timedelta(hours=hours_ago)).isoformat()})


def a(text, mid=None):
    return AIMessage(content=text, id=mid)


async def count(text):  # 1 token per 3 characters, like the context-budget tests
    return len(text) // 3 + 1


def test_valid_turn_id_accepts_only_canonical_uuids():
    assert ch.valid_turn_id(U1.upper()) == U1
    assert ch.valid_turn_id(" " + U1 + " ") == U1
    for bad in (None, "", "c123456", "p-" + U1, U1[:-1], 42):
        assert ch.valid_turn_id(bad) is None


def test_completed_turns_drop_orphans_empty_answers_and_the_current_question():
    msgs = [q("Q1"), a("A1"), q("Q2 failed"), q("Q3"), a("A3"), q("Q4"), a("   "), q("current")]
    assert [(t[0].content, t[1].content) for t in ch.completed_turns(msgs)] == [("Q1", "A1"), ("Q3", "A3")]
    assert ch.completed_turns([]) == [] and ch.completed_turns([q("only")]) == []


def test_fresh_turns_drop_old_and_timeless_questions_and_zero_switches_off():
    old, new = (q("old", hours_ago=13), a("A")), (q("new", hours_ago=2), a("B"))
    timeless = (HumanMessage(content="before the change"), a("C"))
    assert [t[0].content for t in ch.fresh_turns([timeless, old, new], 12, NOW)] == ["new"]
    assert len(ch.fresh_turns([timeless, old, new], 0, NOW)) == 3
    naive = (HumanMessage(content="naive", additional_kwargs={"ts": "2026-10-03T11:00:00"}), a("D"))
    assert ch.fresh_turns([naive], 12, NOW) == []


def test_context_turns_keep_the_newest_completed_fresh_turns():
    msgs = [q("Q1"), a("A1"), q("Q2"), a("A2"), q("Q3"), a("A3"), q("now")]
    assert [t[0].content for t in ch.context_turns(msgs, 2, 12, NOW)] == ["Q2", "Q3"]
    assert ch.context_turns(msgs, 0, 12, NOW) == []


def test_clean_answer_removes_citation_markers():
    assert ch.clean_answer("Es gilt die DV [2]. Siehe auch [1, 3] und [12].") == "Es gilt die DV. Siehe auch und."
    assert ch.clean_answer("Kein Marker 2024.") == "Kein Marker 2024."


def test_select_turns_takes_whole_turns_newest_first_without_gaps():
    long_answer = "Satz eins. " * 200  # about 740 tokens with the fake counter
    turns = [(q("old short"), a("ok")), (q("middle"), a(long_answer)), (q("newest"), a("kurz [1]"))]
    chosen = asyncio.run(ch.select_turns(turns, 300, count))
    assert [t[0].content for t in chosen] == ["newest"]  # the long middle turn stops the walk: no gap
    assert chosen[0][1].content == "kurz"  # answers come back cleaned


def test_select_turns_shortens_an_overlong_newest_turn_at_a_sentence_end():
    answer = " ".join("Satz Nummer %d ist hier." % i for i in range(200))
    chosen = asyncio.run(ch.select_turns([(q("Frage?"), a(answer))], 200, count))
    text = chosen[0][1].content
    assert text.endswith("ist hier. …") and len(text) < len(answer)
    assert asyncio.run(count(text)) + asyncio.run(count("Frage?")) + 16 <= 200


def test_as_messages_turns_the_turns_into_chat_messages():
    from assistant_api.graph.history import as_messages

    msgs = as_messages([(q("Wie lange?"), a("Zehn Tage [1].")), (q("x" * 600), a("y" * 10))])
    assert [m.type for m in msgs] == ["human", "ai", "human", "ai"] and msgs[1].content == "Zehn Tage."


def test_history_text():
    turns = [(q("Wie lange?"), a("Zehn Tage [1].")), (q("x" * 600), a("y" * 10))]
    text = ch.history_text(turns)
    assert text.startswith("User: Wie lange?\n\nAssistant: Zehn Tage.")
    assert "User: " + "x" * 500 + "..." in text
    assert ch.history_text([]) == "Kein Verlauf"


def test_insert_history_after_the_system_message_or_first():
    from assistant_api.graph.history import insert_history

    base = [SystemMessage(content="sys"), HumanMessage(content="user")]
    hist = [HumanMessage(content="h1"), AIMessage(content="a1")]
    assert [m.content for m in insert_history(base, hist)] == ["sys", "h1", "a1", "user"]
    assert [m.content for m in insert_history(base, hist, "first")] == ["h1", "a1", "sys", "user"]
    assert [m.content for m in insert_history([HumanMessage(content="u")], hist)] == ["h1", "a1", "u"]


def test_question_message_carries_the_turn_id_and_the_time():
    from assistant_api.graph.history import question_message

    m = question_message("Frage?", U1, NOW)
    assert m.id == U1 and m.additional_kwargs["ts"] == NOW.isoformat() and ch.message_time(m) == NOW
    assert ch.valid_turn_id(question_message("Frage?", None, NOW).id)  # a fresh UUID when the page sent none


def test_kept_ids_reads_the_pages_kept_questions():
    assert ch.kept_ids(None) is None
    assert ch.kept_ids("") == [] and ch.kept_ids("  ") == []
    assert ch.kept_ids(U1 + "," + U2.upper()) == [U1, U2]
    assert ch.kept_ids(U1 + ",c123456") is None  # one invalid entry: the whole value is ignored
    six = ["00000000-0000-4000-8000-00000000000%d" % i for i in range(6)]
    assert ch.kept_ids(",".join(six)) == six[1:]  # at most the newest five


def test_conversation_state_tells_the_direct_reply_how_many_earlier_turns_it_sees():
    # the reply model is told, not left to infer, whether there is anything to rework
    assert ch.conversation_state(0) == "Vor der aktuellen Nachricht gibt es keine früheren Fragen und Antworten."
    assert ch.conversation_state(1) == "Vor der aktuellen Nachricht steht eine frühere Frage mit Antwort."
    assert ch.conversation_state(2) == "Vor der aktuellen Nachricht stehen 2 frühere Fragen mit Antworten."


def test_type_dispatch_matches_the_message_classes_the_state_can_hold():
    from langchain_core.messages import AIMessageChunk, ChatMessage, ToolMessage
    from langgraph.graph.message import add_messages

    for message in (
        HumanMessage(content="q"),
        AIMessage(content="a"),
        SystemMessage(content="s"),
        ToolMessage(content="t", tool_call_id="x"),
        ChatMessage(role="user", content="c"),
    ):
        assert (message.type == "human") == isinstance(message, HumanMessage)
        assert (message.type == "ai") == isinstance(message, AIMessage)
    # the reducer turns streamed chunks into messages, so a chunk (whose type differs) never reaches the stored state
    assert [m.type for m in add_messages([], [AIMessageChunk(content="x", id="1")])] == ["ai"]  # pyright: ignore[reportAttributeAccessIssue]


def test_cut_turns_places_the_edit_cut_at_turn_level():
    ids = ["a", "b", "c"]
    assert ch.cut_turns(ids, None) == [] and ch.cut_turns(ids, "b") == ["b", "c"]
    assert ch.cut_turns(ids, "x") == []  # never stored, no kept list: nowhere
    assert ch.cut_turns(ids, "x", []) == ids  # the page kept nothing: before everything
    assert ch.cut_turns(ids, "x", ["a", "b"]) == ["c"]  # after the newest kept turn (found[-1]; found[0] gives b, c)
    assert ch.cut_turns(ids, "x", ["z"]) == []  # no kept turn is stored


def test_section_identity_is_the_evals_runners():
    meta = {"download_url": "https://x/a.pdf", "url": "https://x/b", "page_number": 3}
    digest = hashlib.sha256("Text".encode("utf-8")).hexdigest()
    assert section_identity("Text", meta) == "https://x/a.pdf|3|" + digest
    assert (
        section_identity("Text", {"url": "https://x/b", "part_index": 0, "page_number": 3}) == "https://x/b|0|" + digest
    )
    assert section_identity("Text", {}) == "||" + digest
