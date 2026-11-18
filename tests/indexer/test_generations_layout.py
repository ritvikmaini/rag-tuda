"""Generations per layout without a database: the v2 table names, begin's
row and refusals, the typed views, and the SQL a v2 publish issues. A fake connection records the SQL; the real thing runs
in tests/pgvector/test_generations_postgres.py."""

import pytest

from indexer.index import ddl
from indexer.index import generations as gen

TARGET = gen.Target("example")


class Conn:
    """execute() records the SQL and answers from `answers`: the first (substring, rows) whose substring is in the SQL."""

    def __init__(self, answers=()):
        self.answers, self.sql = list(answers), []

    def execute(self, sql, params=None):
        self.sql.append((sql, params))
        for needle, rows in self.answers:
            if needle in sql:
                return Result(rows(params) if callable(rows) else rows)
        return Result([(None,)])


class Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


def test_tables_follow_the_layout():
    assert (
        TARGET.tables(12)
        == ddl.v2_tables("example_g0012")
        == {
            "children": "example_g0012_chunks",
            "parents": "example_g0012_sections",
        }
    )
    assert TARGET.typed_views == ("example_chunks", "example_sections")


def test_the_gate_uses_the_required_indexes_of_the_layout():
    assert gen.missing_indexes("parents", []) == ["primary section_id", "btree on document_id"]


def test_begin_records_a_v2_row_with_the_v2_tables():
    conn = Conn([("to_regclass", [(True,)]), ("max(generation)", [(2,)]), ("pg_class", [])])
    n, converted = gen.begin(conn, TARGET, {"config_hash": "x"})
    assert (n, converted) == (2, None)
    sql, params = next(c for c in conn.sql if c[0].startswith("INSERT INTO index_generations"))
    assert "'staging', %s, %s::jsonb" in sql
    assert params[:4] == (
        "example",
        2,
        "v2",
        '{"children": "example_g0002_chunks", "parents": "example_g0002_sections"}',
    )


def test_begin_refuses_when_a_v2_name_exists():
    # relkind answers 'r' only for the chunks table
    def relkind(params):
        return [("r",)] if params == ("example_g0002_chunks",) else []

    conn = Conn([("to_regclass", [(True,)]), ("max(generation)", [(2,)]), ("c.relkind", relkind)])
    with pytest.raises(gen.GenerationError, match="example_g0002_chunks exists"):
        gen.begin(conn, TARGET)


def test_a_v2_publish_repoints_only_the_typed_views():
    conn = Conn()
    gen._create_views(conn, TARGET, {"layout": "v2", "tables": TARGET.tables(3)})
    assert [sql.split(" AS ")[0] for sql, _ in conn.sql] == [
        'CREATE OR REPLACE VIEW "example_chunks"',
        'CREATE OR REPLACE VIEW "example_sections"',
        ddl.view_grant("example"),
    ]
    assert not [sql for sql, _ in conn.sql if "example_production" in sql or "langchain" in sql]


def test_ensure_tables_grants_the_registry_whenever_the_reader_lacks_select():
    for readable, grants in ((False, 1), (True, 0)):
        conn = Conn([("has_table_privilege", [(readable,)])])
        gen.ensure_tables(conn)
        assert [sql for sql, _ in conn.sql].count(ddl.REGISTRY_GRANT) == grants, readable


def test_the_check_after_a_v2_publish_reads_only_the_typed_views(monkeypatch):
    monkeypatch.setattr(
        gen,
        "typed_view_tables",
        lambda conn, target: {
            "example_chunks": "example_g0003_chunks",
            "example_sections": "example_g0003_sections",
        },
    )
    conn = Conn([("SELECT 1 FROM", [(1,)])])
    gen._check_views(conn, TARGET, {"layout": "v2", "tables": TARGET.tables(3)})
    assert [sql for sql, _ in conn.sql] == [
        'SELECT 1 FROM "example_chunks" LIMIT 1',
        'SELECT 1 FROM "example_sections" LIMIT 1',
    ]


def test_typed_view_tables_reads_only_the_typed_views_that_exist(monkeypatch):
    kinds = {
        "example_chunks": "v",
        "example_sections": None,
    }
    monkeypatch.setattr(gen, "relkind", lambda conn, name: kinds[name])
    conn = Conn(
        [("pg_get_viewdef", lambda p: [(f" SELECT a\n   FROM {p[0].replace('example_', 'example_g0002_')};",)])]
    )
    assert gen.typed_view_tables(conn, TARGET) == {"example_chunks": "example_g0002_chunks"}
