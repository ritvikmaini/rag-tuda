"""end_turn logs the [Turn] line for every turn (code before the colon of the record's error) and hands the record to the store's finish."""

import asyncio
import logging
import re
import time
from types import SimpleNamespace

import pytest

from assistant_api.chat.turns_store import TurnRecord, end_turn

LINE = re.compile(r"^\[Turn\] outcome=\S+ code=\S+ intent=\S+ ttft_ms=(\d+|-) total_ms=(\d+|-)$")
T = "11111111-1111-4111-8111-111111111111"


class Store:
    def __init__(self, fail=False):
        self.finished, self.fail = [], fail

    async def finish(self, record, outcome):
        self.finished.append((record, outcome))
        if self.fail:
            raise RuntimeError("store down")


def turn_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[Turn]")]


def end(store, record, outcome, error):
    asyncio.run(end_turn(store, record, outcome, error))


def test_the_turn_line_per_outcome(caplog):
    caplog.set_level(logging.INFO)
    store = Store()
    ok = TurnRecord(T, "c", "Frage")
    ok.result = SimpleNamespace(analysis={"intent": "lookup"}, first_chunk=ok.started + 0.5)
    end(store, ok, "completed", None)
    kept = TurnRecord(T, "c", "Frage", error="pipeline_error:KeyError")
    end(store, kept, "failed", "pipeline_error")
    late = TurnRecord(T, "c", "Frage")
    end(store, late, "failed", "timeout")
    end(store, None, "completed", None)  # a variant or a switch: no record
    end(None, TurnRecord(T, "c", "Frage"), "failed", "busy")  # no store
    lines = turn_lines(caplog)
    assert all(LINE.match(line) for line in lines), lines
    assert re.fullmatch(r"\[Turn\] outcome=completed code=- intent=lookup ttft_ms=\d+ total_ms=\d+", lines[0])
    assert 500 <= int(re.search(r"ttft_ms=(\d+)", lines[0]).group(1)) < 5000
    assert lines[1].startswith("[Turn] outcome=failed code=pipeline_error intent=- ttft_ms=-")
    assert kept.error == "pipeline_error:KeyError"  # the class the pipeline named stays
    assert late.error == "timeout"
    assert lines[3] == "[Turn] outcome=completed code=- intent=- ttft_ms=- total_ms=-"
    assert lines[4].startswith("[Turn] outcome=failed code=busy ")
    assert [(r, o) for r, o in store.finished] == [
        (ok, "completed"),
        (kept, "failed"),
        (late, "failed"),
    ]  # no record/store: none


def test_the_turn_line_names_a_fallback_turn_only(caplog):
    caplog.set_level(logging.INFO)
    answered = TurnRecord(T, "c", "Frage")
    answered.result = SimpleNamespace(analysis={"intent": "out_of_scope"}, first_chunk=None, scope_fallback=True)
    declined = TurnRecord(T, "c", "Frage")
    declined.result = SimpleNamespace(analysis={"intent": "out_of_scope"}, first_chunk=None, scope_fallback=False)
    end(Store(), answered, "completed", None)
    end(Store(), declined, "completed", None)
    end(Store(), TurnRecord(T, "c", "Frage"), "completed", None)  # no result yet
    lines = turn_lines(caplog)
    assert re.fullmatch(
        r"\[Turn\] outcome=completed code=- intent=out_of_scope ttft_ms=- total_ms=\d+ fallback=yes", lines[0]
    )
    assert all("fallback" not in line and LINE.match(line) for line in lines[1:]), lines


def test_a_fallback_turn_context_says_so_and_no_other_does():
    from assistant_api.chat.turns_store import turn_context

    doc = SimpleNamespace(page_content="Q", metadata={"section_id": "p1"})
    analysis = {"standalone_question": "F?", "intent": "out_of_scope", "language": "de"}
    assert turn_context([doc], analysis, [], scope_fallback=True)["scope_fallback"] is True
    assert "scope_fallback" not in turn_context([doc], analysis, [])
    assert "scope_fallback" not in turn_context([doc], analysis, [], scope_fallback=False)


def test_a_failing_store_still_logs_the_turn_line(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(RuntimeError, match="store down"):
        end(Store(fail=True), TurnRecord(T, "c", "Frage"), "failed", "empty_answer")
    assert len(turn_lines(caplog)) == 1 and "code=empty_answer" in turn_lines(caplog)[0]


def test_the_turn_line_never_holds_the_question(caplog):
    caplog.set_level(logging.INFO)
    record = TurnRecord(T, "c", "Wo steht der Fixturbrunnen?", started=time.monotonic())
    end(Store(), record, "completed", None)
    assert "Fixturbrunnen" not in caplog.text
