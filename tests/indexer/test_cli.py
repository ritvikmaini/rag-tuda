"""The indexer CLI: the five commands with their
output and exit codes, and publish, which puts a staging generation live, reloads the backend and never housekeeps. A fake
connection factory; the generations functions are replaced at the connection level, the SQL runs in
tests/pgvector/test_generations_postgres.py."""

import json

import pytest

from indexer import cli, gates
from indexer.index import generations as gen


class Conn:
    def close(self):
        pass


class Lock:
    busy = False

    def __init__(self, target, holder, connect_fn):
        self.holder = holder

    def acquire(self):
        if Lock.busy:
            raise gen.BuildBusy("process-corpus", "2026-10-12T10:00:00")
        return self

    def release(self):
        pass


@pytest.fixture
def run(db_settings, monkeypatch, tmp_path):
    """run(*argv) -> (exit code, printed lines)."""
    monkeypatch.setattr(gen, "BuildLock", Lock)
    monkeypatch.setattr(Lock, "busy", False)

    def call(*argv):
        lines = []
        code = cli.main(list(argv), settings=db_settings, connect_fn=lambda application_name: Conn(), out=lines.append)
        return code, lines

    call.stats = tmp_path / "state" / gates.STATS_NAME
    return call


def rows(*statuses):
    return [
        {"generation": n, "status": s, "published_at": None, "tables": {}, "layout": "v2"}
        for n, s in enumerate(statuses, start=1)
    ]


def test_status_prints_the_registry_json(run, monkeypatch):
    generations = rows("previous", "live")
    monkeypatch.setattr(gen, "generations", lambda conn, target: generations)
    monkeypatch.setattr(gen, "lock_status", lambda conn, target: {"locked": False, "holder": None, "started_at": None,
                                                                  "pid": None})  # fmt: skip
    monkeypatch.setattr(gen, "typed_view_tables", lambda conn, target: {})
    monkeypatch.setattr(gen, "stale_count", lambda conn, target: 0)
    code, lines = run("status")
    printed = json.loads(lines[0])
    assert code == 0 and len(lines) == 1 and lines[0] == json.dumps(printed, indent=1, default=str)
    assert list(printed) == ["status", "lock", "typed_views", "live", "previous",
                             "generations", "stale_tables"]  # fmt: skip
    assert (printed["lock"]["locked"], printed["live"]["generation"]) == (False, 2)


def test_compare_housekeep_and_rollback_keep_their_output_and_exit_codes(run, monkeypatch):
    report = {"generations": {1: {"parents": 3, "children": 6}, 2: {"parents": 3, "children": 6}},
              "parents": {"only_a": 0, "only_b": 1}, "children": {"only_a": 0, "only_b": 0}, "sources": {},
              "samples": [], "identical": False}  # fmt: skip
    monkeypatch.setattr(gen, "compare", lambda conn, target, a, b: report)
    code, lines = run("compare", "1", "2")
    assert code == 1 and lines[-1] == "RESULT DIFFERENT" and lines[0] == "generation 1: 3 parents, 6 children"
    monkeypatch.setattr(gen, "compare", lambda conn, target, a, b: {**report, "identical": True})
    assert run("compare", "1", "2")[0] == 0

    result = {"dropped_tables": ["example_g0001"], "dropped_views": [], "kept": ["3 (live)"], "errors": ["x: locked"],
              "freed_bytes": 2e9}  # fmt: skip
    asked = []
    monkeypatch.setattr(gen, "housekeep", lambda conn, target, stale=False: asked.append(stale) or result)
    assert run("housekeep", "--stale") == (
        1,
        ["dropped table example_g0001", "kept x: locked", "kept generations 3 (live); 1 errors; freed 2.0 GB"],
    )
    assert asked == [True]
    monkeypatch.setattr(gen, "housekeep", lambda conn, target, stale=False: {**result, "errors": []})
    assert run("housekeep")[0] == 0

    monkeypatch.setattr(gen, "rollback", lambda conn, target: {"live": 3, "previous": 4})
    assert run("rollback") == (0, ["rollback: generation 3 live, generation 4 previous"])
    monkeypatch.setattr(
        gen, "rollback", lambda conn, target: (_ for _ in ()).throw(gen.RollbackRefused("no previous generation"))
    )
    assert run("rollback") == (1, ["refused: no previous generation"])

    Lock.busy = True
    code, lines = run("housekeep")
    assert code == 1 and lines == ["busy: a build holds the lock since 2026-10-12T10:00:00 (process-corpus)"]
    Lock.busy = False
    for argv in ([], ["drop"], ["compare", "1"], ["publish"], ["publish", "x"], ["publish", "1", "2"]):
        assert run(*argv) == (2, [cli.USAGE])


def test_publish_puts_a_staging_generation_live_and_does_not_housekeep(run, monkeypatch):
    calls = []
    monkeypatch.setattr(
        gen, "generations", lambda conn, target: rows("previous", "previous", "live", "staging", "staging")
    )
    monkeypatch.setattr(
        gen,
        "verify",
        lambda conn, target, n: calls.append(("verify", n)) or {"children": 9, "parents": 7},
    )
    monkeypatch.setattr(gen, "publish", lambda conn, target, n: calls.append(("publish", n)) or (4, 0.123))
    monkeypatch.setattr(gen, "annotate", lambda conn, target, n, counts=None, timings=None: calls.append(
        ("annotate", n, counts, timings)))  # fmt: skip
    monkeypatch.setattr(gen, "housekeep", lambda *a, **k: pytest.fail("publish must not housekeep"))
    code, lines = run("publish", "5")
    assert code == 0
    assert calls == [
        ("verify", 5),
        ("publish", 5),
        ("annotate", 5, {"children": 9, "parents": 7}, {"publish_s": 0.123}),
    ]
    assert lines == ["published generation 5 (previous 4) in 0.123 s"]
    stats = json.loads(run.stats.read_text())
    assert (stats["parents"], stats["children"]) == (7, 9)


def test_publish_still_records_the_stats_when_annotate_fails_after_the_commit(run, monkeypatch):
    monkeypatch.setattr(gen, "generations", lambda conn, target: rows("live", "staging"))
    monkeypatch.setattr(gen, "verify", lambda conn, target, n: {"children": 9, "parents": 7})
    monkeypatch.setattr(gen, "publish", lambda conn, target, n: (1, 0.5))
    monkeypatch.setattr(gen, "annotate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("connection lost")))
    code, lines = run("publish", "2")
    assert code == 0
    assert lines == [
        "warning: annotate: connection lost",
        "published generation 2 (previous 1) in 0.5 s",
    ]
    assert json.loads(run.stats.read_text())["parents"] == 7


def test_publish_refuses_a_generation_that_is_not_staging(run, monkeypatch):
    monkeypatch.setattr(gen, "generations", lambda conn, target: rows("live", "previous", "failed", "dropped"))
    monkeypatch.setattr(gen, "publish", lambda *a: pytest.fail("must not publish"))
    for n, word in ((1, "live"), (2, "previous"), (3, "failed"), (4, "dropped"), (9, "unknown")):
        assert run("publish", str(n)) == (1, [f"refused: generation {n} is {word}, not staging"])
    monkeypatch.setattr(gen, "generations", lambda conn, target: rows("staging"))
    monkeypatch.setattr(
        gen, "verify", lambda conn, target, n: (_ for _ in ()).throw(gen.GenerationError("verify failed: x"))
    )
    assert run("publish", "1") == (1, ["refused: verify failed: x"])
    assert not run.stats.exists()


def test_publish_takes_the_build_lock(run, monkeypatch):
    monkeypatch.setattr(gen, "generations", lambda conn, target: pytest.fail("no read without the lock"))
    Lock.busy = True
    assert run("publish", "5") == (1, ["busy: a build holds the lock since 2026-10-12T10:00:00 (process-corpus)"])
