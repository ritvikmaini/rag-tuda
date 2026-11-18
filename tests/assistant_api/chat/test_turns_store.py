"""The turn store on Postgres (needs the harness database): history cuts, locking, versions, switches, context rows."""

import asyncio
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.conninfo import make_conninfo
from rag_core.adapters.pg_index import read_pool
from rag_testing import harness

from assistant_api.chat import turns_store as ts

ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "deploy" / "ci" / ".generated"
GOLDEN = ROOT / "tests" / "assistant_api" / "fixtures" / "history_golden.json"
pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_DOCKER_TESTS") and (GEN / "ci.env").exists()), reason="needs the harness"
)


def secret(name: str) -> str:
    return (GEN / "secrets" / (name + ".txt")).read_text(encoding="utf-8").strip()


def conninfo() -> str:
    return make_conninfo(
        host="127.0.0.1",
        port=harness.DB_PORT,
        dbname=secret("pgdatabase"),
        user="rag_app",
        password=secret("rag_app_password"),
    )


def run(work):
    async def main():
        async with read_pool(conninfo(), max_size=3, name="test-turns") as pool:
            return await work(ts.TurnStore(pool))

    return asyncio.run(main())


def doc(text: str, section_id: str, score: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        page_content=text,
        metadata={
            "section_id": section_id,
            "cross_encoder_score": score,
            "url": "https://docs.example.org/x.pdf",
            "page_number": 2,
        },
    )


def result(answer, candidates=(), sources=(), shape="text", scope_fallback=False):
    return SimpleNamespace(
        scope_fallback=scope_fallback,
        answer=answer,
        requested_shape=shape,
        candidates=list(candidates),
        sources=list(sources),
        analysis={"standalone_question": "Frage?", "intent": "lookup", "language": "Deutsch"},
        first_chunk=None,
    )


async def ask(
    store,
    conversation,
    question,
    outcome="completed",
    answer="Antwort [1].",
    candidates=(),
    sources=(),
    replace_from=None,
    kept=None,
    turn_id=None,
    shape="text",
    scope_fallback=False,
):
    t = ts.TurnRecord(turn_id=turn_id or str(uuid.uuid4()), conversation_id=conversation, question=question)
    await store.begin(t, replace_from, kept)
    past = None
    if outcome not in ("too_long", "busy"):
        past = await store.history(t, replace_from, kept)
    t.result = result(answer if outcome == "completed" else None, candidates, sources, shape, scope_fallback)
    t.run = {"collection": "example", "index_generation": 1, "config_hash": "abc", "prompts": {"rag-generation": 2}}
    await store.finish(t, outcome)
    return t, past


def test_the_golden_conversations_give_their_expected_history():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    async def work(store):
        out = []
        for conversation in golden["conversations"]:
            conv, ids = "golden-" + uuid.uuid4().hex, {}
            fresh = lambda i: ids.setdefault(i, str(uuid.uuid4()))  # noqa: E731 - per replay, so the test can run again
            got = []
            for s in conversation["steps"]:
                kept = None if s["kept"] is None else [fresh(k) for k in s["kept"]]
                replace_from = fresh(s["replace_from"]) if s["replace_from"] else None
                _, past = await ask(
                    store,
                    conv,
                    s["q"],
                    s["outcome"],
                    s["answer"],
                    replace_from=replace_from,
                    kept=kept,
                    turn_id=fresh(s["id"]),
                )
                got.append(None if past is None else [[p.question, p.answer] for p in past])
            out.append(got)
        return out

    assert run(work) == [[s["history"] for s in c["steps"]] for c in golden["conversations"]]


def test_the_lock_serialises_writers_of_one_conversation():
    conv = "lock-" + uuid.uuid4().hex

    async def work(store):
        async with await psycopg.AsyncConnection.connect(conninfo()) as holder:
            async with holder.transaction():
                await holder.execute("SELECT pg_advisory_xact_lock(1, hashtext(%s))", (conv,))
                begin = asyncio.ensure_future(store.begin(ts.TurnRecord(str(uuid.uuid4()), conv, "Frage"), None, None))
                await asyncio.sleep(0.5)
                waited = not begin.done()
            await asyncio.wait_for(begin, 5)
        return waited

    assert run(work) is True


def test_the_second_cut_catches_a_turn_stored_after_the_first():
    conv = "race-" + uuid.uuid4().hex

    async def work(store):
        first, _ = await ask(store, conv, "Frage eins")
        late_id = str(uuid.uuid4())
        retry = ts.TurnRecord(str(uuid.uuid4()), conv, "Frage zwei, neu")
        await store.begin(retry, late_id, [first.turn_id])  # the replaced question is not stored yet
        late = ts.TurnRecord(late_id, conv, "Frage zwei")  # a slow request of the same conversation stores it now
        await store.begin(late, None, None)
        late.result = result("Antwort zwei")
        await store.finish(late, "completed")
        past = await store.history(retry, late_id, [first.turn_id])  # the slot is held: the second cut
        retry.result = result("Antwort zwei, neu")
        await store.finish(retry, "completed")
        _, third = await ask(store, conv, "Frage drei")  # without the second cut the late turn would be read here
        return [p.question for p in past], [p.question for p in third]

    # the retry's own history hides the late turn by time anyway (HISTORY's created_at <); the next question is what only the
    # second cut protects: the late turn is superseded, so it never comes back
    assert run(work) == (["Frage eins"], ["Frage eins", "Frage zwei, neu"])


def test_a_variant_keeps_both_versions_and_the_history_reads_the_active_one():
    conv = "variant-" + uuid.uuid4().hex

    async def work(store):
        first, _ = await ask(
            store,
            conv,
            "Wie melde ich mich krank?",
            answer="Original [1].",
            candidates=[doc("Quelle", "p1", float("nan"))],
            sources=[{"n": 1, "section_id": "p1"}],
            shape="steps",
        )
        problem = await store.variant_problem(conv, "mindmap", first.turn_id)
        taken = await store.variant_input(conv, "mindmap", first.turn_id)
        kept = await store.keep_version(conv, first.turn_id, "mindmap", "Karte [1].", taken.shape)
        shorter = await store.variant_input(conv, "shorter", first.turn_id)
        _, past_after_variant = await ask(store, conv, "Danke", answer="Gern.")
        return problem, taken, kept, shorter.shape, past_after_variant

    problem, taken, kept, shorter_shape, past = run(work)
    assert problem == "" and kept is True
    assert (taken.question, taken.shape, taken.standalone_question, taken.intent, taken.language) == (
        "Wie melde ich mich krank?",
        "mindmap",
        "Frage?",
        "lookup",
        "Deutsch",
    )
    assert taken.candidates[0]["metadata"]["cross_encoder_score"] is None  # NaN stored as null
    assert taken.candidates[0]["section"].startswith("https://docs.example.org/x.pdf|2|")
    assert shorter_shape == "steps"  # a length variant is written in the original's shape
    assert [(p.question, p.answer) for p in past] == [("Wie melde ich mich krank?", "Karte [1].")]


def test_a_switch_flips_the_active_version_and_needs_a_stored_version():
    conv = "switch-" + uuid.uuid4().hex

    async def work(store):
        first, _ = await ask(store, conv, "Frage", answer="Original.", candidates=[doc("Q", "p1")])
        before = await store.select(conv, first.turn_id, "original")  # no version stored yet: refused
        await store.keep_version(conv, first.turn_id, "table", "| a |", "table")
        back = await store.select(conv, first.turn_id, "original")
        unknown = await store.select(conv, first.turn_id, "english")
        t2 = ts.TurnRecord(str(uuid.uuid4()), conv, "Weiter")
        await store.begin(t2, None, None)
        past = await store.history(t2, None, None)
        return before, back, unknown, [p.answer for p in past]

    assert run(work) == (False, True, False, ["Original."])


def test_variant_refusals_keep_their_reasons():
    conv = "refuse-" + uuid.uuid4().hex

    async def work(store):
        none_yet = await store.variant_problem(conv, "table", str(uuid.uuid4()))
        direct, _ = await ask(store, conv, "Hallo", answer="Guten Tag.")  # no sources: a direct reply
        no_sources = await store.variant_problem(conv, "table", direct.turn_id)
        lookup, _ = await ask(store, conv, "Frage", answer="Antwort.", candidates=[doc("Q", "p1")])
        await ask(store, conv, "x" * 2100, "too_long")  # too long after it: never stored, so not "the last"
        still = await store.variant_problem(conv, "table", lookup.turn_id)
        earlier = await store.variant_problem(conv, "table", direct.turn_id)
        await ask(store, conv, "Abbruch", "stopped")
        stopped = await store.variant_problem(conv, "table", lookup.turn_id)
        return none_yet, no_sources, still, earlier, stopped

    assert run(work) == ("no answer to change", "no sources", "", "not the last question", "no answer to change")


def test_a_fallback_turn_stores_the_flag_in_its_context_and_its_variant_input_reads_it():
    """Only a turn answered by the out-of-scope fallback has the key; its variant re-fits from the sources."""
    conv = "fallback-" + uuid.uuid4().hex

    async def work(store):
        plain, _ = await ask(store, conv, "Frage eins", candidates=[doc("Q", "p1")], sources=[{"section_id": "p1"}])
        fell_back, _ = await ask(
            store,
            conv,
            "Frage zwei",
            candidates=[doc("R", "p2")],
            sources=[{"section_id": "p2"}],
            scope_fallback=True,
        )
        async with store.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT sections ? 'scope_fallback' FROM chat.turn_context WHERE turn_id = ANY(%s::uuid[]) "
                    "ORDER BY turn_id = %s",
                    ([plain.turn_id, fell_back.turn_id], fell_back.turn_id),
                )
            ).fetchall()
        return rows, await store.variant_input(conv, "shorter", fell_back.turn_id)

    rows, taken = run(work)
    assert rows == [(False,), (True,)] and taken.scope_fallback is True


def test_finish_writes_the_row_and_one_context_for_a_completed_lookup():
    conv = "finish-" + uuid.uuid4().hex

    async def work(store):
        done, _ = await ask(
            store,
            conv,
            "Frage",
            answer="Antwort [1].",
            candidates=[doc("Q", "p1"), doc("R", "p2")],
            sources=[{"n": 1, "section_id": "p1"}, {"n": 2, "section_id": "p2"}],
        )
        fallback, _ = await ask(store, conv, "Frage zwei", answer=None)  # no model message: never completed
        async with store.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT outcome, retrieved_section_ids, cited_section_ids, prompts, collection, index_generation, "
                    "config_hash, stage_ms ? 'total', (SELECT count(*) FROM chat.turn_context c WHERE c.turn_id = t.turn_id) "
                    "FROM chat.turns t WHERE conversation_id = %s ORDER BY created_at",
                    (conv,),
                )
            ).fetchall()
        return rows

    rows = run(work)
    assert rows[0] == ("completed", ["p1", "p2"], ["p1"], {"rag-generation": 2}, "example", 1, "abc", True, 1)
    assert rows[1][0] == "failed" and rows[1][-1] == 0


def test_a_duplicate_turn_id_is_recorded_once():
    conv, tid = "dup-" + uuid.uuid4().hex, str(uuid.uuid4())

    async def work(store):
        a = ts.TurnRecord(tid, conv, "Frage")
        b = ts.TurnRecord(tid, conv, "Frage")
        await store.begin(a, None, None)
        await store.begin(b, None, None)
        return a.stored, b.stored

    assert run(work) == (True, False)


def test_the_live_generation_and_the_ages_are_read():
    async def work(store):
        return await store.live_generation("rag"), await store.ages()

    generation, ages = run(work)
    assert isinstance(generation, int)  # the harness's index build ran before the Docker-gated line
    assert set(ages) == {"<1d", "1-7d", "7-30d", "30-365d", ">365d"} and ages["<1d"] >= 1


def test_finish_writes_the_error_of_a_failed_turn_and_null_otherwise():
    """chat.turns.error holds `<code>:<ExceptionClass>` for a failed turn and stays null for a completed one."""
    conv = "error-" + uuid.uuid4().hex

    async def work(store):
        failed = ts.TurnRecord(turn_id=str(uuid.uuid4()), conversation_id=conv, question="Frage eins")
        await store.begin(failed, None, None)
        failed.result = result(None)
        await ts.end_turn(store, failed, "failed", "exception:KeyError")
        fine, _ = await ask(store, conv, "Frage zwei")
        async with store.pool.connection() as c:
            rows = await (
                await c.execute(
                    "SELECT turn_id::text, outcome, error FROM chat.turns WHERE conversation_id = %s", (conv,)
                )
            ).fetchall()
        return failed.turn_id, fine.turn_id, {r[0]: (r[1], r[2]) for r in rows}

    failed_id, fine_id, rows = run(work)
    assert rows == {failed_id: ("failed", "exception:KeyError"), fine_id: ("completed", None)}
