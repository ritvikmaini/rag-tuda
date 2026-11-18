"""Stream format 1 on the real backend: meta first on
every response, done last unless the client stopped, every line `data: …` or blank, the codes and outcomes per path, the
stored turn id = meta's. Black box (imports no product code). HARNESS_RECORD_V1=1 writes each response to
tests/contracts/fixtures/transcripts_v1/ (on purpose only, as the index snapshot)."""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
from rag_testing import harness

OUT = Path(__file__).resolve().parents[1] / "contracts" / "fixtures" / "transcripts_v1"
HEX32 = re.compile(r"^[0-9a-f]{32}$")
SLOW = "eins zwei drei vier fünf sechs sieben acht [1]."


def record(name: str, path: str, reply: harness.Reply, **params: str) -> None:
    if os.environ.get("HARNESS_RECORD_V1") == "1":
        OUT.mkdir(exist_ok=True)
        (OUT / (name + ".sse")).write_text(reply.raw, encoding="utf-8")
        side = {"path": path, "status": reply.status, "params": params}
        (OUT / (name + ".json")).write_text(json.dumps(side, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def check(reply: harness.Reply, outcome: str | None, turn: str | None = None) -> list[dict[str, Any]]:
    """The format-1 rules every response keeps; returns the stream."""
    stream = reply.stream
    assert reply.status == 200 and stream[0]["type"] == "meta"
    meta = stream[0]["content"]
    assert meta["format"] == 1
    assert meta["turn_id"] == turn if turn else uuid.UUID(meta["turn_id"]).version == 4
    assert meta["trace_id"] is None or HEX32.match(meta["trace_id"])
    end = harness.frames(reply.raw)[1] if outcome is None else len(reply.raw)  # a client stop may cut inside the marker
    assert all(line == "" or line.startswith("data: ") for line in reply.raw[:end].split("\n"))
    if outcome is None:
        assert "done" not in [f["type"] for f in stream]
    else:
        assert reply.raw.endswith("\n\n") and harness.serialise_v1(stream) == reply.raw
        assert stream[-1]["type"] == "done" and stream[-1]["content"]["outcome"] == outcome
        assert set(stream[-1]["content"]) <= {
            "outcome",
            "generation",
            "prompts",
        }  # the turn's generation and prompt versions
        assert [f["type"] for f in stream].count("meta") == 1
    return stream


def test_a_lookup_and_a_direct_reply_complete(ready: None, assistant, llm, user) -> None:
    turn = str(uuid.uuid4())
    lookup = assistant.ask(harness.THESIS, user, turn_id=turn, formats="v1")
    check(lookup, "completed", turn)
    assert harness.path_of(lookup.stream) == ["meta", *harness.LOOKUP_FORMATS, "done"]
    record("lookup_formats", "lookup_formats", lookup, question=harness.THESIS, turn_id=turn, formats="v1")
    llm.script(harness.OUT_OF_SCOPE)
    direct = assistant.ask(harness.JOKE, user + "-d")
    assert harness.path_of(check(direct, "completed")) == ["meta", *harness.DIRECT, "done"]
    record("direct", "direct", direct, question=harness.JOKE)


def test_meta_turn_id_is_the_stored_question_id(ready: None, assistant, llm, user) -> None:
    first = assistant.ask(harness.THESIS, user, formats="v1")  # no turn_id: the backend names the turn
    turn = check(first, "completed")[0]["content"]["turn_id"]
    shorter = assistant.ask(harness.THESIS, user, formats="v1", variant="shorter", variant_of=turn)
    stream = check(shorter, "completed", turn)
    assert harness.path_of(stream) == ["meta", *harness.VARIANT, "done"] and stream[-2]["content"] == {
        "name": "shorter",
        "ok": True,
    }
    record("variant", "variant", shorter, question=harness.THESIS, formats="v1", variant="shorter", variant_of="<turn>")
    switch = assistant.ask(harness.THESIS, user, formats="v1", variant_select="original", variant_of=turn)
    assert harness.path_of(check(switch, "completed", turn)) == ["meta", "variant", "done"]
    record("version_switch", "version_switch", switch, formats="v1", variant_select="original", variant_of="<turn>")
    missing = assistant.ask(harness.THESIS, user, formats="v1", variant_select="table", variant_of=turn)
    assert check(missing, "refused", turn)[1] == {
        "type": "variant",
        "content": {"name": "table", "ok": False},
        "code": "no_version",
    }
    record(
        "version_switch_missing",
        "version_switch_missing",
        missing,
        formats="v1",
        variant_select="table",
        variant_of="<turn>",
    )


@pytest.mark.parametrize(
    ("params", "code"),
    [
        ({"variant": "shorter"}, "formats_off"),
        ({"variant": "poster", "formats": "v1"}, "unknown_variant"),
        ({"variant": "shorter", "formats": "v1", "variant_of": ""}, "no_turn_id"),
    ],
)
def test_a_refused_variant_names_why(ready: None, assistant, llm, user, params, code) -> None:
    turn = str(uuid.uuid4())
    assistant.ask(harness.THESIS, user, turn_id=turn, formats="v1")
    query = {"variant_of": turn, **params}
    if not query["variant_of"]:
        del query["variant_of"]
    reply = assistant.ask(harness.THESIS, user, **query)
    stream = check(reply, "refused")
    assert [f["type"] for f in stream] == ["meta", "chunk", "variant", "done"]
    assert stream[1]["content"] == harness.REFUSED and stream[2]["code"] == code
    if code == "formats_off":
        record("refused", "refused", reply, question=harness.THESIS, variant="shorter", variant_of="<turn>")


def test_a_variant_without_an_answer_fails(ready: None, assistant, llm, user) -> None:
    turn = str(uuid.uuid4())
    assistant.ask(harness.THESIS, user, turn_id=turn, formats="v1")
    llm.script({"purpose": "answer", "fail": 500})
    stream = check(
        assistant.ask(harness.THESIS, user, formats="v1", variant="shorter", variant_of=turn), "failed", turn
    )
    assert stream[-2] == {"type": "variant", "content": {"name": "shorter", "ok": False}, "code": "failed"}


def test_too_long_busy_and_not_ready(ready: None, first_build, assistant, llm, user) -> None:
    turn = str(uuid.uuid4())
    long = assistant.ask("x" * 2001, user, turn_id=turn)
    assert check(long, "too_long", turn)[1:-1] == [
        {"type": "chunk", "content": harness.TOO_LONG}
    ]  # done: check() read it
    assert long.stream[0]["content"]["trace_id"] is None
    record("too_long", "too_long", long, question="x" * 2001, turn_id=turn)
    llm.script({"purpose": "answer", "match": harness.THESIS, "reply": SLOW, "token_delay_s": 0.8})
    thread, out = harness.in_thread(lambda: assistant.ask(harness.THESIS, user))
    harness.wait_until(lambda: bool(llm.calls("answer")))
    busy = assistant.ask(harness.OFFICE_HOURS, user + "-b")
    thread.join()
    assert check(busy, "busy")[1:-1] == [
        {"type": "error", "content": harness.BUSY_ERROR, "code": "busy"},
        {"type": "chunk", "content": harness.MSG_BUSY},
    ]
    record("busy", "busy", busy, question=harness.OFFICE_HOURS)
    before = first_build["before"]  # asked before the first build: no index yet
    assert check(before, "failed")[1:3] == [
        {"type": "error", "content": harness.NOT_READY_ERROR, "code": "not_ready"},
        {"type": "chunk", "content": harness.MSG_NOT_READY},
    ]
    record("not_ready", "not_ready", before, question=harness.THESIS)


def test_a_guard_timeout_ends_with_done_after_the_german_chunk(ready: None, assistant, llm, user) -> None:
    llm.script({"purpose": "answer", "first_delay_s": 8})
    stream = check(assistant.ask(harness.THESIS, user), "failed")
    assert stream[-3:-1] == [
        {"type": "error", "content": "timeout", "code": "timeout"},
        {"type": "chunk", "content": harness.MSG_TIMEOUT},
    ]


def test_a_client_stop_gets_no_done(ready: None, assistant, llm, user) -> None:
    llm.script({"purpose": "answer", "reply": SLOW, "token_delay_s": 1.0})
    stopped = assistant.ask(harness.THESIS, user, stop_after="chunk")
    assert harness.path_of(check(stopped, None)) == ["meta", *harness.LOOKUP[:7]]
    record("client_stop", "client_stop", stopped, question=harness.THESIS)


def test_done_names_the_turns_index_generation_and_prompt_versions(ready: None, assistant, llm, user, db) -> None:
    turn = str(uuid.uuid4())
    done = check(assistant.ask(harness.THESIS, user, turn_id=turn, formats="v1"), "completed", turn)[-1]["content"]
    row = db.execute("SELECT index_generation, prompts FROM chat.turns WHERE turn_id = %s", (turn,)).fetchone()
    assert isinstance(done["generation"], int) and not isinstance(done["generation"], bool)
    assert (done["generation"], done.get("prompts", {})) == (
        row[0],
        row[1],
    )  # the stored row's: record.run, read at the end
    long_turn = str(uuid.uuid4())
    long = check(assistant.ask("x" * 2001, user, turn_id=long_turn), "too_long", long_turn)[-1]["content"]
    assert long["generation"] == done["generation"] and "prompts" not in long  # an early exit names the generation only
    refused = check(assistant.ask(harness.THESIS, user, variant="shorter", variant_of=turn), "refused")[-1]["content"]
    assert refused == {"outcome": "refused"}  # a variant has no turn row
