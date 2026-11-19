"""The chat service on the turn store (fake store, run_stream replaced): cuts, history, variants, switches, the record."""

import asyncio
import logging
from dataclasses import replace
from datetime import date, datetime, timezone
from types import SimpleNamespace

from assistant_api.chat import service
from assistant_api.chat.turns_store import TurnRecord
from assistant_api.sse import frames as fr
from rag_core.domain.filters import Filter
from rag_core.domain.history import PastTurn, VariantInput

T1, T2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
NOW = datetime.now(timezone.utc)


class Store:
    def __init__(self, problem=""):
        self.calls, self.problem, self.inputs = [], problem, None

    async def begin(self, t, replace_from, kept):
        self.calls.append(("begin", replace_from, kept))
        t.stored = True

    async def history(self, t, replace_from, kept):
        self.calls.append(("history", rt.slot.locked()))
        return [PastTurn(T1, "Frage eins", "Antwort eins.", NOW)]

    async def variant_problem(self, conversation_id, variant, variant_of):
        self.calls.append(("variant_problem", variant))
        return self.problem

    async def variant_input(self, conversation_id, variant, variant_of):
        self.calls.append(("variant_input", rt.slot.locked()))
        return VariantInput(
            T1,
            "Frage eins",
            NOW,
            (),
            ({"page_content": "Q", "metadata": {"section_id": "p"}},),
            "S",
            "lookup",
            "Deutsch",
            "steps",
        )

    async def keep_version(self, conversation_id, turn_id, name, text, shape):
        self.calls.append(("keep_version", name, text, shape))
        return True

    async def select(self, conversation_id, variant_of, name):
        self.calls.append(("select", name))
        return True


def fake_stream(answer):
    async def run_stream(chain, inputs, config, *, result, **kwargs):
        store.inputs = inputs
        result.answer = answer
        if answer:
            yield fr.Chunk(content=answer)

    return run_stream


store = Store()
rt = SimpleNamespace()  # filled per test from the attributes the landed service reads


def runtime(problem=""):
    global store, rt
    store = Store(problem)
    rt = SimpleNamespace(
        chain=object(),
        turns=store,
        slot=asyncio.Semaphore(1),
        handler=None,
        langfuse=None,
        collection="example",
        messages={"variant.refused": "abgelehnt"},
        run_info=lambda: {"collection": "example"},
        settings=SimpleNamespace(
            guard=SimpleNamespace(max_question_chars=20),
            card=SimpleNamespace(quirks=SimpleNamespace(strip_cjk=False)),
            profile=SimpleNamespace(
                guard=SimpleNamespace(max_question_chars=20), formats=SimpleNamespace(sources_event="before")
            ),
            overlay=SimpleNamespace(compute=SimpleNamespace(slot_wait_s=0.1)),
            messages=SimpleNamespace(
                question=SimpleNamespace(too_long="zu lang {limit}"),
                docs=SimpleNamespace(none="keine"),
                variant=SimpleNamespace(refused="abgelehnt"),
            ),
        ),
    )
    return rt


def collect(monkeypatch, answer="Antwort zwei.", **kwargs):
    monkeypatch.setattr(service, "run_stream", fake_stream(answer))

    async def main():
        # trace_id has no declared default in the pipeline: passed explicitly
        return [f async for f in service.pipeline(rt, user="c-1", turn=kwargs.pop("turn", T2), trace_id=None, **kwargs)]

    return asyncio.run(main())


def test_a_question_is_stored_first_and_reads_its_history_with_the_slot(monkeypatch):
    runtime()
    record = TurnRecord(T2, "c-1", "Frage zwei")
    collect(monkeypatch, question="Frage zwei", turn_id=T2, replace_from=T1, kept="", record=record)
    assert store.calls == [("begin", T1, []), ("history", True)]
    assert [m.content for m in store.inputs["messages"]] == ["Frage eins", "Antwort eins.", "Frage zwei"]
    assert record.result.answer == "Antwort zwei." and record.run == {"collection": "example"}


def seeded(applied):
    """The filter a question's run starts from when the newest stored turn holds `applied` (its JSON, or None)."""
    runtime()

    async def history(t, replace_from, kept):
        return [PastTurn(T1, "Frage eins", "Antwort eins.", NOW, applied)]

    store.history = history
    return store, TurnRecord(T2, "c-1", "Frage drei")


def test_a_question_starts_from_the_previous_turns_applied_filter(monkeypatch):
    stored = Filter(programme=("bsc-informatik",), po_version=(2015,), date=date(2026, 10, 1), include_archived=True)
    for applied, expected in (
        (stored.as_json(), stored),
        (Filter().as_json(), Filter()),  # an empty filter still leaves archived rows out: it is not "no filter"
        (None, None),  # a turn run without filtering
        ({"degree": "master"}, None),  # a stored value that is no filter costs the carry-over, not the answer
    ):
        _, record = seeded(applied)
        collect(monkeypatch, question="Frage drei", turn_id=T2, record=record)
        assert store.inputs["filter"] == expected, applied
    runtime()  # no earlier turn but the fake's one without a stored filter
    collect(monkeypatch, question="Frage drei", turn_id=T2, record=TurnRecord(T2, "c-1", "Frage drei"))
    assert store.inputs["filter"] is None


def test_a_variant_runs_on_the_filter_of_its_turn(monkeypatch):
    runtime()
    kept = Filter(programme=("bsc-informatik",), po_version=(2015,), include_archived=True, po_source="term")

    async def variant_input(conversation_id, variant, variant_of):
        return replace(await Store.variant_input(store, conversation_id, variant, variant_of), filter=kept)

    store.variant_input = variant_input
    collect(monkeypatch, question="Frage eins", formats="v1", variant="shorter", variant_of=T1, turn=T1)
    assert store.inputs["filter"] == kept


def test_a_too_long_question_is_stored_before_it_is_refused(monkeypatch):
    runtime()
    record = TurnRecord(T2, "c-1", "x" * 30)
    out = collect(monkeypatch, question="x" * 30, turn_id=T2, record=record)
    assert store.calls == [("begin", None, None)] and any(
        isinstance(f, fr.Declared) and f.outcome == "too_long" for f in out
    )
    assert record.run == {"collection": "example", "prompts": {}}


def test_a_variant_runs_from_the_stored_context_and_keeps_its_answer(monkeypatch):
    runtime()
    out = collect(monkeypatch, question="Frage eins", formats="v1", variant="shorter", variant_of=T1, turn=T1)
    assert [c[0] for c in store.calls] == ["variant_problem", "variant_input", "keep_version"]
    assert store.calls[1] == ("variant_input", True) and store.calls[2] == (
        "keep_version",
        "shorter",
        "Antwort zwei.",
        "steps",
    )
    assert store.inputs["answer_shape"] == "steps" and [d.page_content for d in store.inputs["grouped_raw_docs"]] == [
        "Q"
    ]
    assert not any(isinstance(f, fr.Declared) for f in out)


def test_a_variant_without_an_answer_keeps_nothing(monkeypatch):
    runtime()
    out = collect(
        monkeypatch, answer=None, question="Frage eins", formats="v1", variant="shorter", variant_of=T1, turn=T1
    )
    assert "keep_version" not in [c[0] for c in store.calls]
    assert any(isinstance(f, fr.Declared) and f.outcome == "failed" for f in out)


def test_a_refused_variant_and_a_switch_touch_no_turn_row(monkeypatch):
    runtime(problem="no sources")
    out = collect(monkeypatch, question="Frage eins", formats="v1", variant="table", variant_of=T1, turn=T1)
    assert store.calls == [("variant_problem", "table")] and any(isinstance(f, fr.Declared) for f in out)
    runtime()
    collect(monkeypatch, question="Frage eins", formats="v1", variant_select="original", variant_of=T1, turn=T1)
    assert store.calls == [("select", "original")]


def test_the_record_is_given_its_run_after_the_stream_so_it_names_what_the_turn_used(monkeypatch):
    runtime()
    rt.run_info = lambda: {
        "ran": store.inputs is not None
    }  # the prompt versions Langfuse served are known only after the run
    record = TurnRecord(T2, "c-1", "Frage zwei")
    collect(monkeypatch, question="Frage zwei", turn_id=T2, record=record)
    assert record.run == {"ran": True} and not rt.slot.locked()


def test_the_turn_write_counts_as_activity_before_the_slot_wait(monkeypatch):
    runtime()
    rt.slot = asyncio.Semaphore(0)  # the slot stays busy: the request ends with the busy message
    out = collect(monkeypatch, question="Frage", turn_id=T2, record=TurnRecord(T2, "c-1", "Frage"))
    assert out[0] == service.PROGRESS and any(isinstance(f, fr.Declared) and f.outcome == "busy" for f in out)


def test_a_pipeline_exception_names_its_class_on_the_record(monkeypatch, caplog):
    """The record keeps `pipeline_error:<ExceptionClass>` (never the message); the log line names the class only."""
    runtime()

    async def failing(chain, inputs, config, *, result, **kwargs):
        raise ValueError("Personalnummer 4711")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(service, "run_stream", failing)
    record = TurnRecord(T2, "c-1", "Frage zwei")

    async def main():
        return [
            f
            async for f in service.pipeline(
                rt, user="c-1", turn=T2, trace_id=None, question="Frage zwei", turn_id=T2, record=record
            )
        ]

    out = asyncio.run(main())
    assert record.error == "pipeline_error:ValueError"
    assert [f.code for f in out if isinstance(f, fr.Error)] == ["pipeline_error"]
    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert lines == ["Error in RAG pipeline (user: c-1): ValueError"]
