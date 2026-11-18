"""The retention job: the periods are pinned, the SQL of both cutoffs, the deletion order, the trace paging and batching
against a fake Langfuse client, a skipped step that does not stop the others, and the report line."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import psycopg
import pytest

from assistant_api import retention

AGES = {"<1d": 12, "1-7d": 3, "7-30d": 0, "30-365d": 0, ">365d": 0}


class Result:
    def __init__(self, rows):
        self.rows = rows

    async def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, rows, fail_on=None):
        self.rows, self.fail_on, self.calls = rows, fail_on, []

    async def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if self.fail_on and self.fail_on in sql:
            raise psycopg.errors.InsufficientPrivilege("permission denied for table checkpoint_writes\nDETAIL: x")
        return Result(self.rows if sql.startswith("SELECT") else [])

    @asynccontextmanager
    async def transaction(self):
        yield


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def connection(self):
        yield self.conn


class FakeApi:
    """observations.get_many pages 3 observations per trace with a cursor; trace.delete_multiple records the batches."""

    def __init__(self, total, per_page=500, per_trace=3):
        self.rows = ["t%03d" % (i // per_trace) for i in range(total * per_trace)]
        self.per_page, self.listed, self.deleted = per_page, [], []
        self.observations = SimpleNamespace(get_many=self.get_many)
        self.trace = SimpleNamespace(delete_multiple=self.delete_multiple)

    def get_many(self, *, to_start_time, fields, limit, cursor, request_options):
        self.listed.append((to_start_time, fields, limit, cursor))
        start = int(cursor or 0)
        end = start + self.per_page
        nxt = str(end) if end < len(self.rows) else None
        data = [SimpleNamespace(trace_id=t) for t in self.rows[start:end]]
        return SimpleNamespace(data=data, meta=SimpleNamespace(cursor=nxt))

    def delete_multiple(self, *, trace_ids, request_options):
        self.deleted.append(list(trace_ids))


def client_of(api):
    return SimpleNamespace(api=api)


def test_the_periods_are_pinned():
    assert (retention.CONVERSATION_DAYS, retention.TRACE_DAYS, retention.BATCH, retention.MAX_PAGES) == (
        90,
        30,
        100,
        200,
    )


def test_conversations_report_and_delete_children_first():
    rows = [("t1", "c-1"), ("t2", "c-1"), ("t3", "c-1"), ("t4", "c-2"), ("t5", "c-2")]
    conn = FakeConn(rows)
    assert asyncio.run(retention.conversations(FakePool(conn), False)) == (2, 5)
    assert [c[0] for c in conn.calls if not c[0].startswith("SET")] == [retention.OLD_CONVERSATIONS]
    assert conn.calls[1][1] == (90,) and "max(created_at) < now() - %s::int * interval '1 day'" in conn.calls[1][0]
    conn = FakeConn(rows)
    assert asyncio.run(retention.conversations(FakePool(conn), True)) == (2, 5)
    deletes = [c for c in conn.calls if c[0].startswith("DELETE")]
    assert [d[0].split()[2] for d in deletes] == ["chat.turn_context", "chat.turn_versions", "chat.turns"]
    # keyed by the selected turn ids, never by conversation: a turn committed after the read stays
    assert all(d[1] == (["t1", "t2", "t3", "t4", "t5"],) and "turn_id = ANY(%s)" in d[0] for d in deletes)
    assert not any("conversation_id" in d[0] for d in deletes)


def test_nothing_old_deletes_nothing():
    conn = FakeConn([])
    assert asyncio.run(retention.conversations(FakePool(conn), True)) == (0, 0)
    assert asyncio.run(retention.checkpoint_threads(FakePool(conn), True)) == 0
    assert not [c for c in conn.calls if c[0].startswith("DELETE")]


def test_checkpoint_threads_use_the_newest_checkpoint_and_delete_all_three_tables():
    conn = FakeConn([("th-1",), ("th-2",)])
    assert asyncio.run(retention.checkpoint_threads(FakePool(conn), True)) == 2
    select = conn.calls[1]
    assert "max((checkpoint->>'ts')::timestamptz) < now() - %s::int * interval '1 day'" in select[0]
    assert select[1] == (90,)
    deletes = [c for c in conn.calls if c[0].startswith("DELETE")]
    assert [d[0].split()[2] for d in deletes] == [
        "public.checkpoint_writes",
        "public.checkpoint_blobs",
        "public.checkpoints",
    ]


def test_traces_report_counts_distinct_trace_ids_without_deleting():
    api = FakeApi(250, per_page=300)
    now = datetime(2026, 11, 1, tzinfo=UTC)
    assert retention.traces(client_of(api), False, now) == 250
    assert api.deleted == []
    assert [c[3] for c in api.listed] == [None, "300", "600"]  # cursor paging until none
    assert all(c[0] == now - timedelta(days=30) and c[1] == "core" and c[2] == 1000 for c in api.listed)


def test_traces_are_deleted_in_batches_of_100_once_all_are_listed():
    api = FakeApi(250)
    assert retention.traces(client_of(api), True) == 250
    assert [len(b) for b in api.deleted] == [100, 100, 50]
    assert sum(api.deleted, []) == sorted(set(api.rows))
    empty = FakeApi(0)
    assert retention.traces(client_of(empty), True) == 0 and empty.deleted == []


def test_the_page_cap_stops_listing_and_prints_one_line(monkeypatch, capsys):
    monkeypatch.setattr(retention, "MAX_PAGES", 2)
    api = FakeApi(250, per_page=100)
    assert retention.traces(client_of(api), True) == 67  # 200 observations of 3 per trace
    assert len(api.listed) == 2
    assert "page cap of 2 reached" in capsys.readouterr().out
    assert sum(len(b) for b in api.deleted) == 67


def test_report_lines():
    found = {"conversations": (4, 17), "checkpoints": 2, "traces": 30}
    assert (
        retention.report(found, AGES, True)
        == "[Retention] conversations -4 (turns -17), checkpoint threads -2, traces -30; "
        "kept: <1d 12, 1-7d 3, 7-30d 0, 30-365d 0, >365d 0"
    )
    assert retention.report(found, AGES, False).startswith(
        "[Retention] report only, would go: conversations 4 (turns 17), checkpoint threads 2, traces 30; turns by age: <1d 12"
    )
    skipped = {"conversations": (0, 0), "checkpoints": "skipped", "traces": "skipped"}
    assert retention.report(skipped, "unknown", True) == (
        "[Retention] conversations -0 (turns -0), checkpoint threads skipped, traces skipped; kept: unknown"
    )


def test_a_denied_checkpoint_step_is_skipped_with_one_line_and_the_others_run(monkeypatch, capsys):
    conn = FakeConn([("x", 1)], fail_on="public.checkpoint")

    @asynccontextmanager
    async def pool(*a, **k):
        yield FakePool(conn)

    class Store:
        def __init__(self, p):
            pass

        async def ages(self):
            return AGES

    monkeypatch.setattr(retention, "read_pool", pool)
    monkeypatch.setattr(retention, "store_conninfo", lambda db: "")
    monkeypatch.setattr(retention, "TurnStore", Store)
    monkeypatch.setattr(retention, "traces", lambda client, apply: 7)
    monkeypatch.setattr(retention, "_trace_client", lambda url: SimpleNamespace(shutdown=lambda: None))
    settings = SimpleNamespace(overlay=SimpleNamespace(database=None, tracing=SimpleNamespace(url="http://lf")))
    line, complete = asyncio.run(retention.run(settings, True))
    assert not complete
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "[Retention] checkpoints skipped: InsufficientPrivilege: permission denied for table checkpoint_writes"
    ]
    assert line.startswith(
        "[Retention] conversations -1 (turns -1), checkpoint threads skipped, traces -7; kept: <1d 12"
    )


def test_main_prints_the_line_and_exits_by_completeness(monkeypatch, capsys):
    async def run(settings, apply):
        return "[Retention] line apply=%s" % apply, apply

    monkeypatch.setattr(retention, "load_settings", lambda: object())
    monkeypatch.setattr(retention, "run", run)
    assert retention.main([]) == 1 and capsys.readouterr().out.strip() == "[Retention] line apply=False"
    assert retention.main(["--apply"]) == 0
    with pytest.raises(SystemExit):
        retention.main(["--dry-run"])
