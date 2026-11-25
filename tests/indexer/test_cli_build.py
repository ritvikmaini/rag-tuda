"""`indexer build`: one build of every source the profile lists, an atomic swap, the stats file, the counts
per source on stdout and exit 0, or exit 1 when the lock is held, a gate refuses or a step fails. build_corpus and
write_and_publish are replaced (they have their own tests); the lock and the connection are fakes."""

import json
from pathlib import Path

import pytest

from indexer import build, cli
from indexer.index import generations as gen

BUILT = {
    "status": "success",
    "sources": {"web": {"documents": 3}, "pdfs": {"documents": 5, "with_documents": 2}, "maps": {"documents": 2}},
    "gates": {"size": {"count": 7, "last": None, "ratio": 0.7, "passed": True}},
    "parents": 7,
    "profile": "example",
    "source_set": {"set": "core", "names": ["maps", "pdfs", "web"]},
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
    fails = None  # an exception acquire raises (the database is down)
    at_release = []  # what each release saw: whether the stats file existed

    def __init__(self, target, holder, connect_fn):
        self.holder = holder

    def acquire(self):
        if Lock.fails:
            raise Lock.fails
        if Lock.busy:
            raise gen.BuildBusy("process-corpus", "2026-10-12T10:00:00")
        return self

    def release(self):
        Lock.at_release.append(Lock.stats.exists())


@pytest.fixture
def run(db_settings, fakes, monkeypatch):
    """run() -> (exit code, printed lines) of `indexer build`; the swap's arguments are in run.published, a failure of build_corpus is set with run.error."""
    monkeypatch.setattr(gen, "BuildLock", Lock)
    monkeypatch.setattr(Lock, "busy", False)
    monkeypatch.setattr(Lock, "fails", None)
    monkeypatch.setattr(Lock, "at_release", [])
    monkeypatch.setattr(Lock, "stats", fakes.stats, raising=False)
    published = []

    def build_corpus(settings, sources, limit, reuse_inputs, *, embedding, options=None):
        call.options.append(options)
        call.reused.append(reuse_inputs)
        if call.error:
            raise call.error
        return {**BUILT, "sources": call.report} if call.report else BUILT

    async def write_and_publish(settings, target, connect, embedding, parents, *, swap, build_minutes, prov):
        published.append({"parents": parents, "swap": swap, "build_minutes": build_minutes})
        return {
            "generation": 5,
            "index_counts": {"children": 9, "parents": 7},
            "embeddings": {"cached": 8, "computed": 1},
            "changes": {"added": 1, "changed": 2, "removed": 0},
            "swapped": swap,
            "previous_generation": 4,
            "housekeeping": {"dropped_tables": [], "dropped_views": [], "kept": [], "errors": [], "freed_bytes": 0},
        }

    monkeypatch.setattr(build, "build_corpus", build_corpus)
    monkeypatch.setattr(build, "write_and_publish", write_and_publish)

    def call(*flags):
        lines = []
        code = cli.main(
            ["build", *flags],
            settings=db_settings,
            connect_fn=lambda application_name: Conn(),
            embedding=fakes.Embedding(),
            sources={},
            out=lines.append,
        )
        return code, lines

    call.error, call.published, call.stats, call.options, call.report = None, published, fakes.stats, [], None
    call.reused = []
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
    assert (stats["profile"], stats["sources"]) == ("example", BUILT["source_set"])


def test_a_build_whose_stats_could_not_be_written_says_so(run):
    run.stats.mkdir()  # the file cannot be replaced
    code, lines = run()
    assert code == 0
    assert lines[-1] == "warning: the build stats were not written; the next build has no size gate"


def test_a_skipped_size_gate_is_said_once(run, monkeypatch):
    skipped = {**BUILT["gates"]["size"], "skipped": "the last build was of another profile or source set"}
    monkeypatch.setitem(BUILT, "gates", {"size": skipped})
    code, lines = run()
    assert code == 0 and lines.count("note: no size gate: the last build was of another profile or source set") == 1


def test_build_command_exits_1_and_swaps_nothing_when_busy_refused_or_failed(run):
    Lock.busy = True
    assert run() == (1, ["busy: a build holds the lock since 2026-10-12T10:00:00 (process-corpus)"])
    Lock.busy = False
    run.error = build.BuildRefused("only 0 maps documents (minimum 1); swap refused")
    assert run() == (1, ["refused: only 0 maps documents (minimum 1); swap refused"])
    run.error = RuntimeError("connector down")
    assert run() == (1, ["failed: RuntimeError: connector down"])
    assert run.published == [] and not run.stats.exists()


def test_a_database_that_is_down_is_a_failed_line_not_a_traceback(run):
    Lock.fails = ConnectionError("connection refused")
    assert run() == (1, ["failed: ConnectionError: connection refused"])
    assert run.published == []


def test_the_stats_are_written_before_the_lock_is_released(run):
    assert run()[0] == 0
    assert Lock.at_release == [True]


REPORT = {
    "empty_pages": [{"url": "https://x.example/hub", "problem": "main content selector '.main' matched nothing"}],
    "not_indexed": [{"url": "https://x.example/mhb.pdf", "missing": ["programme", "po_version"]}],
    "copies": ["https://x.example/copy/apb.pdf"],
    "skipped": [{"url": "https://x.example/down", "reason": "unreachable"}],
    "pdf_sync": {"retained": ["https://x.example/old.pdf"]},
}
LINES = [
    "warning: web: empty page https://x.example/hub (main content selector '.main' matched nothing)",
    "warning: web: not indexed, lacks programme, po_version: https://x.example/mhb.pdf",
    "note: web: copy skipped, its original is indexed: https://x.example/copy/apb.pdf",
    "note: web: not fetched (unreachable): https://x.example/down",
]
RETAINED = "warning: web: PDF kept though this crawl did not list it: https://x.example/old.pdf"


def test_the_summary_names_the_pages_a_source_reports(run):
    run.report = {"web": {"documents": 3, **REPORT}}
    code, lines = run()
    assert code == 0 and lines[:6] == ["source web: 3 documents", *LINES[:2], RETAINED, *LINES[2:]]


def test_a_refused_build_still_names_the_pages_before_the_refusal(run):
    run.error = build.BuildRefused("corpus shrank to 3 parents (last build 9); swap refused", {"web": REPORT})
    assert run() == (
        1,
        [*LINES[:2], RETAINED, *LINES[2:], "refused: corpus shrank to 3 parents (last build 9); swap refused"],
    )


def test_the_summary_lists_at_most_twenty_pages_of_a_kind(run):
    many = [{"url": f"https://x.example/{i}", "reason": "http 503"} for i in range(23)]
    run.report = {"web": {"documents": 1, "skipped": many}}
    lines = run()[1]
    assert lines[1:22] == [f"note: web: not fetched (http 503): https://x.example/{i}" for i in range(20)] + [
        "note: web: 3 more not fetched"
    ]


def test_exclusions_do_not_hide_a_real_failure_in_the_summary(run):
    left_out = [{"url": f"https://x.example/out{i}", "reason": "excluded"} for i in range(30)]
    failed = {"url": "https://x.example/down", "reason": "http 503"}
    run.report = {"web": {"documents": 1, "skipped": [*left_out, failed]}}
    lines = run()[1]
    assert "note: web: not fetched (http 503): https://x.example/down" in lines
    assert lines.count("note: web: not fetched (excluded): https://x.example/out0") == 1
    assert "note: web: 10 more excluded" in lines


def test_the_snapshot_manifest_is_written_after_the_swap_and_a_failure_to_write_it_is_a_warning(run, db_settings):
    run.report = {
        "web": {
            "documents": 1,
            "manifest": [{"url": "https://x.example/a", "content_hash": "h", "fetched": "2026-11-20", "metadata": {}}],
        }
    }
    code, lines = run()
    assert code == 0 and not [line for line in lines if line.startswith("warning")]
    crawl = Path(db_settings.overlay.paths.crawl_dir)
    written = json.loads((crawl.parent / "snapshots" / "5.json").read_text(encoding="utf-8"))  # generation 5
    assert written["documents"][0]["url"] == "https://x.example/a"
    (crawl.parent / "snapshots" / "5.json").unlink()
    (crawl.parent / "snapshots").rmdir()
    (crawl.parent / "snapshots").write_text("a file where the folder should be", encoding="utf-8")
    code, lines = run()
    assert code == 0 and any(
        line.startswith("warning: the snapshot manifest of generation 5 was not written") for line in lines
    )


def test_a_staged_generation_gets_its_snapshot_manifest_for_the_publish_that_follows(run, db_settings):
    run.report = {
        "web": {
            "documents": 1,
            "manifest": [{"url": "https://x.example/a", "content_hash": "h", "fetched": "2026-11-20", "metadata": {}}],
        },
        "laws": {
            "documents": 1,
            "manifest": [{"url": "https://g.example/l", "content_hash": "k", "fetched": "2026-11-21", "metadata": {}}],
        },
    }
    assert run("--no-swap")[0] == 0 and run.published[-1]["swap"] is False
    crawl = Path(db_settings.overlay.paths.crawl_dir)
    written = json.loads((crawl.parent / "snapshots" / "5.json").read_text(encoding="utf-8"))
    assert [d["url"] for d in written["documents"]] == ["https://g.example/l", "https://x.example/a"]  # every source's


def test_the_core_set_and_a_fresh_crawl_are_the_defaults(run):
    assert run()[0] == 0
    assert run.options == [{"sources": "core", "resume": False}]


@pytest.mark.parametrize(
    "flags, options",
    [
        (("--sources", "full"), {"sources": "full", "resume": False}),
        (("--sources=full", "--resume"), {"sources": "full", "resume": True}),
        (("--resume", "--sources", "core"), {"sources": "core", "resume": True}),
    ],
)
def test_the_source_set_and_resume_come_from_the_command_line(run, flags, options):
    assert run(*flags)[0] == 0
    assert run.options == [options]


def test_frozen_inputs_come_from_the_command_line_and_reach_the_build_not_the_connectors(run):
    assert run()[0] == 0 and run("--reuse-inputs")[0] == 0
    assert run.reused == [False, True] and run.options == [{"sources": "core", "resume": False}] * 2


def test_a_build_without_a_swap_stages_the_generation_and_records_nothing(run):
    code, lines = run("--no-swap")
    assert code == 0 and run.published == [{"parents": ["p"] * 7, "swap": False, "build_minutes": 0.4}]
    assert lines[-1] == (
        "generation 5 staged (not published): 7 sections, 9 chunks, 1 embedded, 8 cached; 1 added, 2 changed, 0 removed"
    )
    assert not run.stats.exists() and Lock.at_release == [False]


@pytest.mark.parametrize(
    "flags", [("--sources",), ("--sources", "all"), ("--sources=",), ("--again",), ("core",), ("--no-swap=1",)]
)
def test_a_bad_build_flag_prints_the_usage_and_builds_nothing(run, flags):
    assert run(*flags) == (2, [cli.USAGE])
    assert run.options == [] and run.published == []


def test_help_lists_the_commands_without_settings(monkeypatch):
    monkeypatch.delenv("RAG_PROFILE", raising=False)
    lines = []
    assert cli.main(["--help"], out=lines.append) == 0
    assert lines == [cli.USAGE] and "build" in cli.USAGE


def test_a_missing_database_secret_file_is_named_and_exits_1(settings, fakes, monkeypatch):
    out = []
    code = cli.main(["build"], settings=settings, embedding=fakes.Embedding(None), sources={}, out=out.append)
    assert code == 1 and "pgdatabase" in out[-1]


def test_the_summary_warns_about_a_folder_that_is_gone():
    report = {"files": 0, "missing_folder": "/home/me/docs"}
    assert cli.summary_lines("house", report) == [
        "warning: house: folder not found, its documents are not indexed: /home/me/docs"
    ]
