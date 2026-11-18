"""Index generations without a database: the names, the verify gate's matcher
over the index definitions Postgres 16 prints for the writer's DDL, the viewdef parser, the CLI's usage and the source
rules (no config, no cascade). The caller passes the target, the connection factory and the refresh URL. The SQL runs in tests/pgvector/test_generations_postgres.py."""

import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from indexer.index import generations as gen  # noqa: E402

SOURCE = open(
    os.path.join(ROOT, "apps", "indexer", "src", "indexer", "index", "generations.py"), encoding="utf-8"
).read()
# pg_get_indexdef of a v2 pair the writer built (ddl.create_tables + ddl.create_indexes; no legacy column)
CHILDREN = [
    ("CREATE INDEX example_g0014_chunks_fts_de ON public.example_g0014_chunks USING gin (fts_de)", True, False),
    ("CREATE INDEX example_g0014_chunks_fts_en ON public.example_g0014_chunks USING gin (fts_en)", True, False),
    (
        "CREATE UNIQUE INDEX example_g0014_chunks_pkey ON public.example_g0014_chunks USING btree (chunk_id)",
        True,
        True,
    ),
    (
        "CREATE INDEX example_g0014_chunks_section ON public.example_g0014_chunks USING btree (section_id)",
        True,
        False,
    ),
]
PARENTS = [
    (
        "CREATE UNIQUE INDEX example_g0014_sections_pkey ON public.example_g0014_sections USING btree (section_id)",
        True,
        True,
    ),
    (
        "CREATE INDEX example_g0014_sections_document ON public.example_g0014_sections USING btree (document_id)",
        True,
        False,
    ),
]


def test_generation_names_and_the_live_views():
    target = gen.Target("example")
    assert target.table(1) == "example_g0001"
    assert target.tables(3) == {"children": "example_g0003_chunks", "parents": "example_g0003_sections"}
    assert gen.Target.for_collection("example") == target


def test_only_plain_lower_case_names_reach_the_sql():
    assert gen.ident("example_g0001_parents") == '"example_g0001_parents"'
    for bad in ("Example", "x; drop", 'a"b', "", "a" * 64, None):
        with pytest.raises(ValueError):
            gen.ident(bad)


def test_the_gate_accepts_the_writers_indexes_by_definition_whatever_their_names():
    assert gen.missing_indexes("children", CHILDREN) == []
    assert gen.missing_indexes("parents", PARENTS) == []


def test_the_gate_names_a_missing_invalid_or_non_unique_index():
    assert gen.missing_indexes("children", CHILDREN[1:]) == ["gin on fts_de"]
    invalid = [(d, "fts_en" not in d, u) for d, v, u in CHILDREN]
    assert gen.missing_indexes("children", invalid) == ["gin on fts_en"]
    not_unique = [(d, v, False) for d, v, u in PARENTS]
    assert gen.missing_indexes("parents", not_unique) == ["primary section_id"]
    assert gen.missing_indexes("parents", []) == ["primary section_id", "btree on document_id"]


def test_the_viewdef_parser_finds_the_table():
    assert gen.viewdef_table(" SELECT langchain_id,\n    content\n   FROM example_g0003;") == "example_g0003"
    assert gen.viewdef_table(' SELECT a\n   FROM public."example_g0004_parents";') == "example_g0004_parents"
    assert gen.viewdef_table("") is None


def test_the_lock_and_status_constants():
    assert (gen.LOCK_NAMESPACE, gen.LOCK_TIMEOUT, gen.PUBLISH_ATTEMPTS, gen.RETRY_SLEEP_S) == (2, "5s", 5, 2)
    assert gen.STATUSES == ("staging", "live", "previous", "failed", "dropped")


def test_a_lock_timeout_is_recognised_by_its_sqlstate():
    class LockNotAvailable(Exception):
        sqlstate = "55P03"

    assert gen.is_lock_timeout(LockNotAvailable()) and not gen.is_lock_timeout(RuntimeError("x"))


def test_busy_names_the_holder_and_the_start():
    from datetime import datetime, timezone

    busy = gen.BuildBusy("process-corpus", datetime(2026, 10, 8, 18, 5, tzinfo=timezone.utc))
    assert str(busy) == "a build holds the lock since 2026-10-08T18:05:00+00:00 (process-corpus)"


def test_the_cli_prints_its_usage_without_touching_the_database():
    target = gen.Target.for_collection("example")
    for argv in ([], ["drop"], ["compare", "1"], ["compare", "1", "x"], ["housekeep", "--all"], ["status", "extra"]):
        lines = []
        assert gen.main(argv, target=target, connect_fn=None, out=lines.append) == 2
        assert lines == [gen.USAGE]
    assert gen.USAGE == (
        "usage: indexer build [--sources core|full] [--resume] [--reuse-inputs] [--no-swap] | status | "
        "housekeep [--stale] | rollback | compare <a> <b> | publish <n>"
    )


def test_the_module_reads_no_config_opens_no_connection_of_its_own_and_never_cascades():
    assert "yaml" not in SOURCE and "/run/secrets" not in SOURCE and "config.yml" not in SOURCE
    assert "def connect(" not in SOURCE and "__main__" not in SOURCE
    assert not re.search(r"^import psycopg|^from psycopg", SOURCE, re.M)  # imported where a connection opens
    assert "CASCADE" not in SOURCE and "DROP VIEW" not in SOURCE  # plain drops; the live views are never dropped


def test_a_rollback_whose_check_fails_after_the_commit_still_reports_the_move(monkeypatch):
    """The views moved (the publish committed): the endpoint must reload the backend and say so, not answer 500."""
    rows = [
        {"generation": 2, "status": "live", "published_at": None},
        {"generation": 1, "status": "previous", "published_at": None},
    ]
    monkeypatch.setattr(gen, "generations", lambda conn, target: rows)

    def unchecked(conn, target, n, note=None):
        raise gen.PublishedUnchecked("generation 1 is published, but the check after the commit failed: x", 2, 0.1)

    monkeypatch.setattr(gen, "publish", unchecked)
    moved = gen.rollback(None, gen.Target("example"))
    assert (moved["live"], moved["previous"]) == (1, 2)
    assert "check after the commit failed" in moved["warning"]
