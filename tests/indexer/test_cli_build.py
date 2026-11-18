"""`indexer build`: one build of every source the profile lists, an atomic swap, the stats file, the counts
per source on stdout and exit 0, or exit 1 when the lock is held, a gate refuses or a step fails. build_corpus and
write_and_publish are replaced (they have their own tests); the lock and the connection are fakes."""

import json

import pytest

from indexer import build, cli
from indexer.index import generations as gen

BUILT = {
    "status": "success",
    "sources": {"web": {"documents": 3}, "pdfs": {"documents": 5, "with_documents": 2}, "maps": {"documents": 2}},
    "gates": {"size": {"count": 7, "last": None, "ratio": 0.7, "passed": True}},
    "parents": 7,
    "units": ["p"] * 7,
    "children": 9,
    "embedding": None,
    "minutes": {"collect": 0.1, "extract": 0.2, "chunk": 0.0},
    "minutes_total": 0.4,
}


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
def run(db_settings, fakes, monkeypatch):
    """run() -> (exit code, printed lines) of `indexer build`; the swap's arguments are in run.published, a failure of build_corpus is set with run.error."""
    monkeypatch.setattr(gen, "BuildLock", Lock)
    monkeypatch.setattr(Lock, "busy", False)
    published = []

    def build_corpus(settings, sources, limit, reuse_inputs, *, embedding):
        if call.error:
            raise call.error
        return BUILT

    async def write_and_publish(settings, target, connect, embedding, parents, *, swap, build_minutes, prov):
        published.append({"parents": parents, "swap": swap, "build_minutes": build_minutes})
        return {
            "generation": 5,
            "index_counts": {"children": 9, "parents": 7},
            "embeddings": {"cached": 8, "computed": 1},
            "changes": {"added": 1, "changed": 2, "removed": 0},
            "swapped": True,
            "previous_generation": 4,
            "housekeeping": {"dropped_tables": [], "dropped_views": [], "kept": [], "errors": [], "freed_bytes": 0},
        }

    monkeypatch.setattr(build, "build_corpus", build_corpus)
    monkeypatch.setattr(build, "write_and_publish", write_and_publish)

    def call():
        lines = []
        code = cli.main(
            ["build"],
            settings=db_settings,
            connect_fn=lambda application_name: Conn(),
            embedding=fakes.Embedding(),
            sources={},
            out=lines.append,
        )
        return code, lines

    call.error, call.published, call.stats = None, published, fakes.stats
    return call


def test_build_command_swaps_generation_and_prints_counts(run):
    code, lines = run()
    assert code == 0
    assert run.published == [{"parents": ["p"] * 7, "swap": True, "build_minutes": 0.4}]
    assert lines == [
        "source web: 3 documents",
        "source pdfs: 5 documents",
        "source maps: 2 documents",
        "generation 5 published (previous 4): 7 sections, 9 chunks, 1 embedded, 8 cached; 1 added, 2 changed, 0 removed",
    ]
    stats = json.loads(run.stats.read_text())
    assert (stats["parents"], stats["children"]) == (7, 9)
    assert stats["documents"] == {"web": 3, "pdfs": 5, "maps": 2}


def test_build_command_exits_1_and_swaps_nothing_when_busy_refused_or_failed(run):
    Lock.busy = True
    assert run() == (1, ["busy: a build holds the lock since 2026-10-12T10:00:00 (process-corpus)"])
    Lock.busy = False
    run.error = build.BuildRefused("only 0 maps documents (minimum 1); swap refused")
    assert run() == (1, ["refused: only 0 maps documents (minimum 1); swap refused"])
    run.error = RuntimeError("connector down")
    assert run() == (1, ["failed: RuntimeError: connector down"])
    assert run.published == [] and not run.stats.exists()


def test_help_lists_the_commands_without_settings(monkeypatch):
    monkeypatch.delenv("RAG_PROFILE", raising=False)
    lines = []
    assert cli.main(["--help"], out=lines.append) == 0
    assert lines == [cli.USAGE] and "build" in cli.USAGE


def test_a_missing_database_secret_file_is_named_and_exits_1(settings, fakes, monkeypatch):
    out = []
    code = cli.main(["build"], settings=settings, embedding=fakes.Embedding(None), sources={}, out=out.append)
    assert code == 1 and "pgdatabase" in out[-1]
