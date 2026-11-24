"""Request guard: every request ends with an answer or a German message."""

import asyncio
import json
from typing import Any

import pytest
from assistant_api.chat import guard as rg  # noqa: E402
from assistant_api.sse import frames as fr

MESSAGES = {  # the four guard texts under the profile's flat keys (Runtime.messages)
    "guard.not_ready": "Der Assistent wird gerade aktualisiert und ist in wenigen Minuten wieder für Sie da.",
    "guard.busy": "Gerade sind sehr viele Anfragen gleichzeitig eingegangen. Bitte versuchen Sie es in einem Moment noch einmal.",
    "guard.error": "Entschuldigung, bei der Beantwortung ist ein technischer Fehler aufgetreten. Bitte versuchen Sie es in einem "
    "Moment noch einmal.",
    "guard.timeout": "Die Antwort dauert gerade ungewöhnlich lange, deshalb habe ich abgebrochen. Bitte stellen Sie Ihre Frage in "
    "ein paar Minuten noch einmal.",
}
META = fr.meta("11111111-1111-4111-8111-111111111111", None, 5.0, 5.0, "m", "vllm")


def frame(kind, content, code: Any = "pipeline_error"):
    if kind == "error":
        return fr.Error(content=content, code=code)
    return {"docs": fr.Docs, "chunk": fr.Chunk, "status": fr.Status}[kind](content=content)


def kinds_and_contents(frames):
    out = []
    for f in frames:
        obj = json.loads(f[len("data: ") :])
        if obj["type"] not in ("meta", "done"):
            out.append((obj["type"], obj["content"]))
    return out


def collect(agen):
    async def run():
        return [f async for f in agen]

    return asyncio.run(run())


def test_frames_pass_through_in_order():
    async def producer():
        yield frame("docs", [])
        yield frame("chunk", "Hallo ")
        yield frame("chunk", "Welt")

    out = collect(rg.guard(producer, deadline_s=5, idle_s=5, meta=META, messages=MESSAGES))
    assert kinds_and_contents(out) == [("docs", []), ("chunk", "Hallo "), ("chunk", "Welt")]


def test_total_deadline_stops_a_trickling_answer_and_cancels_it():
    state = {"cancelled": False}

    async def producer():
        try:
            for i in range(100):
                yield frame("chunk", "x")
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        finally:
            state.setdefault("closed", True)

    out = kinds_and_contents(collect(rg.guard(producer, deadline_s=0.3, idle_s=1, meta=META, messages=MESSAGES)))
    assert out[-2] == ("error", "timeout")
    assert out[-1] == ("chunk", "\n\n" + MESSAGES["guard.timeout"])
    assert 2 <= sum(1 for k, _ in out if k == "chunk") < 20
    assert state.get("closed")


def test_silence_ends_the_request_with_the_timeout_message():
    async def producer():
        await asyncio.sleep(5)
        yield frame("chunk", "late")

    out = kinds_and_contents(collect(rg.guard(producer, deadline_s=10, idle_s=0.1, meta=META, messages=MESSAGES)))
    assert out == [("error", "timeout"), ("chunk", MESSAGES["guard.timeout"])]


def test_english_errors_get_the_matching_german_message():
    cases = [
        ("Service is currently busy. Please try again shortly.", "busy", MESSAGES["guard.busy"]),
        (
            "No index has been built yet. Build it first with `indexer build`.",
            "not_ready",
            MESSAGES["guard.not_ready"],
        ),
        ("An error occurred processing your request.", "pipeline_error", MESSAGES["guard.error"]),
    ]
    for english, code, german in cases:

        async def producer(text=english, code=code):
            yield frame("error", text, code)

        out = kinds_and_contents(collect(rg.guard(producer, deadline_s=5, idle_s=5, meta=META, messages=MESSAGES)))
        assert out == [("error", english), ("chunk", german)], english


def test_an_exception_in_the_pipeline_becomes_the_error_message(caplog):
    async def producer():
        yield frame("chunk", "Teil")
        raise RuntimeError("retrieval unavailable")

    out = kinds_and_contents(collect(rg.guard(producer, deadline_s=5, idle_s=5, meta=META, messages=MESSAGES)))
    assert out == [("chunk", "Teil"), ("error", "RuntimeError"), ("chunk", "\n\n" + MESSAGES["guard.error"])]
    # the log line names the class, never the message (it may hold input values)
    assert [r.getMessage() for r in caplog.records] == ["[RequestGuard] pipeline failed: RuntimeError"]


def test_an_answer_without_any_text_gets_the_error_message():
    async def producer():
        yield frame("docs", [])
        yield frame("status", {"step": "writing"})

    out = kinds_and_contents(collect(rg.guard(producer, deadline_s=5, idle_s=5, meta=META, messages=MESSAGES)))
    assert out[-2:] == [("error", "empty answer"), ("chunk", MESSAGES["guard.error"])]


def test_a_version_switch_is_a_complete_reply():
    async def producer():
        yield fr.variant("original", True)

    out = collect(rg.guard(producer, deadline_s=5, idle_s=5, meta=META, messages=MESSAGES))
    assert [json.loads(f[6:])["type"] for f in out] == ["meta", "variant", "done"]
    assert out[1] == fr.encode(fr.variant("original", True))
    assert json.loads(out[-1][6:])["content"] == {"outcome": "completed"}


def test_a_client_that_disconnects_cancels_the_pipeline():
    state = {"cancelled": False}

    async def producer():
        try:
            yield frame("chunk", "a")
            await asyncio.sleep(5)
            yield frame("chunk", "b")
        except (asyncio.CancelledError, GeneratorExit):
            state["cancelled"] = True
            raise

    async def run():
        agen = rg.guard(producer, deadline_s=10, idle_s=10, meta=META, messages=MESSAGES)
        await agen.__anext__()  # meta
        first = await agen.__anext__()
        await agen.aclose()
        await asyncio.sleep(0.05)
        return first

    first = asyncio.run(run())
    assert json.loads(first[len("data: ") :])["content"] == "a"
    assert state["cancelled"]


def test_progress_signals_keep_a_working_request_alive_and_are_not_forwarded():
    # the slot wait and silent stages (analysis, retrieval,
    # reranking) must not count as silence while the pipeline is working
    async def producer():
        for _ in range(5):
            await asyncio.sleep(0.06)
            yield rg.PROGRESS
        yield frame("chunk", "Antwort")

    out = kinds_and_contents(collect(rg.guard(producer, deadline_s=5, idle_s=0.1, meta=META, messages=MESSAGES)))
    assert out == [("chunk", "Antwort")]


def test_a_reasoning_only_stretch_is_written_as_throttled_comment_lines(monkeypatch):
    # a model that thinks for a long time sends PROGRESS only: the guard's silence limit is met, and the page's watchdog must see
    # the same activity, as SSE comment lines (no data frames), at most one per PROGRESS_COMMENT_EVERY_S
    monkeypatch.setattr(rg, "PROGRESS_COMMENT_EVERY_S", 0.1)

    async def producer():
        for _ in range(30):
            await asyncio.sleep(0.02)
            yield rg.PROGRESS
        yield frame("chunk", "Antwort")

    out = collect(rg.guard(producer, deadline_s=5, idle_s=0.3, meta=META, messages=MESSAGES))
    comments = [f for f in out if f.startswith(":")]
    assert comments == [": progress\n\n"] * len(comments)
    assert 3 <= len(comments) <= 7  # 0.6 s of thinking, one line per 0.1 s at most
    assert kinds_and_contents([f for f in out if not f.startswith(":")]) == [("chunk", "Antwort")]
    assert out[0].startswith("data: ") and out[-1].startswith("data: ")  # comments sit between frames, never inside one


def test_a_fast_request_writes_no_comment_lines():
    async def producer():
        yield rg.PROGRESS
        yield rg.PROGRESS
        yield frame("chunk", "Antwort")

    out = collect(rg.guard(producer, deadline_s=5, idle_s=1, meta=META, messages=MESSAGES))
    assert not [f for f in out if f.startswith(":")]


def test_progress_signals_do_not_outlive_the_total_deadline():
    async def producer():
        while True:
            await asyncio.sleep(0.02)
            yield rg.PROGRESS

    out = kinds_and_contents(collect(rg.guard(producer, deadline_s=0.2, idle_s=0.1, meta=META, messages=MESSAGES)))
    assert out == [("error", "timeout"), ("chunk", MESSAGES["guard.timeout"])]


def run(items, deadline=5.0, idle=1.0):
    async def producer():
        for item in items:
            if isinstance(item, float):
                await asyncio.sleep(item)
            elif isinstance(item, BaseException):
                raise item
            else:
                yield item

    async def collect():
        return [json.loads(f[6:]) async for f in rg.guard(producer, deadline, idle, META, MESSAGES)]

    return asyncio.run(collect())


def kinds(out):
    return [(f["type"], f.get("code") or (f["content"].get("outcome") if f["type"] == "done" else None)) for f in out]


def test_done_reads_the_run_once_the_pipeline_has_ended():
    run_record: dict = {}

    async def producer():
        yield frame("chunk", "Antwort")
        run_record.update(index_generation=5, prompts={"a": 1})  # the service writes record.run in its finally

    async def collect():
        return [json.loads(f[6:]) async for f in rg.guard(producer, 5, 1, META, MESSAGES, run_info=lambda: run_record)]

    assert asyncio.run(collect())[-1]["content"] == {"outcome": "completed", "generation": 5, "prompts": {"a": 1}}


def test_done_on_a_timeout_names_the_generation_without_prompts():
    async def producer():
        await asyncio.sleep(5)
        yield frame("chunk", "late")

    async def collect():
        run_info = lambda: {"index_generation": 2, "prompts": {}}  # noqa: E731
        return [json.loads(f[6:]) async for f in rg.guard(producer, 5, 0.2, META, MESSAGES, run_info=run_info)]

    assert asyncio.run(collect())[-1]["content"] == {"outcome": "failed", "generation": 2}


def test_a_guard_without_run_info_sends_the_plain_done():
    async def producer():
        yield frame("chunk", "Antwort")

    out = collect(rg.guard(producer, deadline_s=5, idle_s=5, meta=META, messages=MESSAGES))
    assert out[-1] == 'data: {"type": "done", "content": {"outcome": "completed"}}\n\n'


def test_done_names_the_scope_fallback_only_on_a_fallback_turn():
    async def producer():
        yield frame("chunk", "Antwort")

    def done_of(**kwargs):
        async def run():
            return [f async for f in rg.guard(producer, 5, 1, META, MESSAGES, **kwargs)]

        return asyncio.run(run())[-1]

    assert done_of(scope_fallback=lambda: True) == (
        'data: {"type": "done", "content": {"outcome": "completed", "scope_fallback": true}}\n\n'
    )
    plain = 'data: {"type": "done", "content": {"outcome": "completed"}}\n\n'
    assert done_of(scope_fallback=lambda: False) == plain == done_of()  # every other turn keeps its bytes


def test_meta_is_first_even_when_the_pipeline_never_yields():
    out = run([5.0], idle=0.2)
    assert kinds(out) == [("meta", None), ("error", "timeout"), ("chunk", None), ("done", "failed")]
    assert out[2]["content"] == MESSAGES["guard.timeout"]


def test_every_path_ends_with_done_after_the_german_chunk():
    assert kinds(run([fr.Chunk(content="a")])) == [("meta", None), ("chunk", None), ("done", "completed")]
    assert kinds(run([])) == [("meta", None), ("error", "empty_answer"), ("chunk", None), ("done", "failed")]
    out = run([fr.Chunk(content="a"), KeyError("k")])
    assert kinds(out)[2:] == [("error", "exception"), ("chunk", None), ("done", "failed")]
    assert out[2]["content"] == "KeyError" and out[3]["content"] == "\n\n" + MESSAGES["guard.error"]


def test_a_declared_outcome_wins_over_the_error_default():
    busy = run(
        [fr.Error(content="Service is currently busy. Please try again shortly.", code="busy"), fr.Declared("busy")]
    )
    assert kinds(busy) == [("meta", None), ("error", "busy"), ("chunk", None), ("done", "busy")]
    assert busy[2]["content"] == MESSAGES["guard.busy"]
    ready = run([fr.Error(content="No index has been built yet.", code="not_ready")])
    assert kinds(ready)[-1] == ("done", "failed") and ready[2]["content"] == MESSAGES["guard.not_ready"]
    refused = run([fr.Chunk(content="r"), fr.variant("table", False, "formats_off"), fr.Declared("refused")])
    assert kinds(refused)[-2:] == [("variant", "formats_off"), ("done", "refused")]


def test_guard_failures_are_failed_whatever_was_declared():
    assert kinds(run([fr.Declared("too_long"), 5.0], idle=0.2))[-1] == ("done", "failed")


def test_progress_and_declarations_are_never_sent():
    out = run([rg.PROGRESS, fr.Declared("too_long"), fr.Chunk(content="lang")])
    assert kinds(out) == [("meta", None), ("chunk", None), ("done", "too_long")]


def test_a_closed_stream_writes_nothing_more():
    async def producer():
        yield fr.Chunk(content="a")
        await asyncio.sleep(10)

    async def main():
        stream = rg.guard(producer, 5.0, 5.0, META, MESSAGES)
        got = [await stream.__anext__(), await stream.__anext__()]
        await stream.aclose()  # the client went away
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        return got

    got = asyncio.run(main())
    assert [json.loads(f[6:])["type"] for f in got] == ["meta", "chunk"]


def run_ending(items, idle=1.0):
    ends = []

    async def on_end(outcome, error=None):
        ends.append((outcome, error))

    async def producer():
        for item in items:
            if isinstance(item, float):
                await asyncio.sleep(item)
            elif isinstance(item, BaseException):
                raise item
            else:
                yield item

    async def collect():
        return [f async for f in rg.guard(producer, 5.0, idle, META, MESSAGES, on_end=on_end)]

    asyncio.run(collect())
    return ends


def test_on_end_gets_the_final_outcome_once_on_every_path():
    def outcomes(items, idle=1.0):
        return [outcome for outcome, _ in run_ending(items, idle)]

    assert outcomes([fr.Chunk(content="a")]) == ["completed"]
    assert outcomes([]) == ["failed"]
    assert outcomes([5.0], idle=0.2) == ["failed"]
    assert outcomes([fr.Chunk(content="a"), KeyError("k")]) == ["failed"]
    assert outcomes([fr.Error(content="busy", code="busy"), fr.Declared("busy")]) == ["busy"]
    assert outcomes([fr.Chunk(content="lang"), fr.Declared("too_long")]) == ["too_long"]


def test_on_end_gets_the_error_of_every_path():
    """The first error the guard sent, as chat.turns.error and the [Turn] line's code read it."""
    assert run_ending([fr.Chunk(content="a")]) == [("completed", None)]
    assert run_ending([]) == [("failed", "empty_answer")]
    assert run_ending([5.0], idle=0.2) == [("failed", "timeout")]
    assert run_ending([fr.Chunk(content="a"), KeyError("k")]) == [("failed", "exception:KeyError")]
    assert run_ending([fr.Error(content="busy", code="busy"), fr.Declared("busy")]) == [("busy", "busy")]
    assert run_ending([fr.Chunk(content="a"), fr.Error(content="x", code="pipeline_error")]) == [
        ("failed", "pipeline_error")
    ]
    assert run_ending([fr.Chunk(content="lang"), fr.Declared("too_long")]) == [("too_long", None)]


def test_a_client_stop_records_stopped_after_the_pipeline_stopped():
    ends, stopped = [], []

    async def producer():
        try:
            yield fr.Chunk(content="a")
            await asyncio.sleep(10)
        finally:
            stopped.append(True)

    async def on_end(outcome, error=None):
        ends.append((outcome, error, bool(stopped)))

    async def main():
        stream = rg.guard(producer, 5.0, 5.0, META, MESSAGES, on_end=on_end)
        await stream.__anext__()
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(main())
    assert ends == [("stopped", None, True)]


def test_a_cancelled_reader_still_records_stopped():
    ends = []

    async def producer():
        yield fr.Chunk(content="a")
        await asyncio.sleep(10)

    async def on_end(outcome, error=None):
        await asyncio.sleep(0.05)
        ends.append(outcome)

    async def main():
        async def read():
            async for _ in rg.guard(producer, 5.0, 5.0, META, MESSAGES, on_end=on_end):
                pass

        reader = asyncio.ensure_future(read())
        await asyncio.sleep(0.1)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        await asyncio.sleep(0.3)  # the shielded end finishes even when the reader's cancellation hits its await

    asyncio.run(main())
    assert ends == ["stopped"]


def test_the_end_waits_for_a_pipeline_that_cleans_up_slowly():
    ends, cleaned = [], []

    async def producer():
        try:
            yield fr.Chunk(content="a")
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0.1)  # the pipeline's own cleanup (a settle, the slot) is not instant
            cleaned.append(True)

    async def on_end(outcome, error=None):
        ends.append((outcome, bool(cleaned)))

    async def main():
        stream = rg.guard(producer, 5.0, 5.0, META, MESSAGES, on_end=on_end)
        await stream.__anext__()
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(main())
    assert ends == [("stopped", True)]


def test_a_second_cancellation_during_the_end_does_not_lose_it():
    # anyio's level cancellation raises again at every await of the finally: the end runs on, shielded
    ends = []

    async def producer():
        yield fr.Chunk(content="a")
        await asyncio.sleep(10)

    async def on_end(outcome, error=None):
        await asyncio.sleep(0.2)
        ends.append(outcome)

    async def main():
        async def read():
            async for _ in rg.guard(producer, 5.0, 5.0, META, MESSAGES, on_end=on_end):
                pass

        reader = asyncio.ensure_future(read())
        await asyncio.sleep(0.1)
        reader.cancel()
        await asyncio.sleep(0.05)  # the reader is now in the finally, awaiting the end
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        await asyncio.sleep(0.4)

    asyncio.run(main())
    assert ends == ["stopped"]


def test_a_failing_end_is_logged_and_never_breaks_the_stream(caplog):
    async def producer():
        yield fr.Chunk(content="a")

    async def on_end(outcome, error=None):
        raise RuntimeError("store down")

    out = collect(rg.guard(producer, 5.0, 5.0, META, MESSAGES, on_end=on_end))
    assert [json.loads(f[6:])["type"] for f in out] == ["meta", "chunk", "done"]
    assert "the turn's end was not recorded: RuntimeError: store down" in caplog.text


def test_the_guard_ends_an_urgent_turn_with_the_contact_line():
    line = "Notrufe: 112 und 110."
    messages = {**MESSAGES, "answer.urgent_contacts": line}

    async def silent():
        await asyncio.sleep(5)
        yield frame("chunk", "late")

    async def failing():
        yield frame("chunk", "Rufen Sie")
        raise RuntimeError("boom")
        yield  # pragma: no cover

    for factory, urgent, expected in (
        (silent, True, [("error", "timeout"), ("chunk", MESSAGES["guard.timeout"] + "\n\n" + line)]),
        (silent, False, [("error", "timeout"), ("chunk", MESSAGES["guard.timeout"])]),
    ):
        out = kinds_and_contents(
            collect(rg.guard(factory, 10, 0.1, META, messages, urgent=lambda urgent=urgent: urgent))
        )
        assert out == expected
    out = kinds_and_contents(collect(rg.guard(failing, 10, 5, META, messages, urgent=lambda: True)))
    assert out[-1] == ("chunk", "\n\n" + MESSAGES["guard.error"] + "\n\n" + line)
