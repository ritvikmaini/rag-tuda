"""The index contract between the indexer and the assistant on a fresh cluster: after a publish as rag_indexer, rag_app
reads the collection's two typed views and the registry and nothing else (no legacy view or column, no privilege on a
generation's tables or on a table created later, no CREATE on public), and every statement of rag_core.adapters.pg_index
runs against the views as rag_app. RUN_DOCKER_TESTS=1."""

import os
import shutil
import uuid

import pytest
from psycopg import errors, sql
from rag_testing import index_view

from indexer.index import ddl
from indexer.index import generations as gen
from rag_core.adapters import pg_index

TARGET = gen.Target.for_collection("rag")
TEXTS = (
    "Die Fahrradgarage am Bahnhof hat dreihundert Stellplaetze",
    "Die Stadtbuecherei oeffnet montags um neun Uhr",
    "Der Wertstoffhof nimmt Sperrmuell und Gruenschnitt an",
    "Die Garage im Rathaus ist fuer Dienstwagen reserviert",
    "Stellplaetze fuer Fahrraeder gibt es am Rathaus",
    "Die Oeffnungszeiten des Buergerbueros im Rathaus",
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="image tests run only with RUN_DOCKER_TESTS=1 and a running Docker",
)


def vec(i, dims=768):
    v = [0.1] * dims
    v[i % dims] = 1.0
    return "[" + ",".join(str(x) for x in v) + "]"


@pytest.fixture(scope="module")
def tables(port):
    """Generation 1 of the collection rag, two sections of three chunks, verified and published as rag_indexer."""
    import psycopg

    conn = psycopg.connect(
        host="127.0.0.1", port=port, dbname="vector_db", user="rag_indexer", password="i" * 64, autocommit=True
    )
    n, _ = gen.begin(conn, TARGET, {"config_hash": "x"})
    names = TARGET.tables(n)
    for statement in ddl.create_tables(names["parents"], names["children"], 768):
        conn.execute(statement)
    for s in range(2):
        section_id = f"s{s}"
        conn.execute(
            f'INSERT INTO "{names["parents"]}" (section_id, document_id, collection, source, audience, part_index, '
            "content, metadata, lang) VALUES (%s, 'doc', 'rag', 'example.web', 'public', %s, %s, %s::jsonb, 'de')",
            (section_id, s, " ".join(TEXTS[3 * s : 3 * s + 3]), "{}"),
        )
        for i in range(3 * s, 3 * s + 3):
            conn.execute(
                f'INSERT INTO "{names["children"]}" (chunk_id, section_id, document_id, collection, source, audience, '
                "content, embedding, lang) VALUES (%s, %s, 'doc', 'rag', 'example.web', 'public', %s, %s::vector, 'de')",
                (uuid.UUID(int=i).hex, section_id, TEXTS[i], vec(i)),
            )
    for statement in ddl.create_indexes(names["parents"], names["children"]):
        conn.execute(statement)
    gen.verify(conn, TARGET, n, {"children": 6, "parents": 2})
    gen.publish(conn, TARGET, n)
    yield names
    conn.close()


def test_a_fresh_index_has_no_legacy_relation_or_column(connect, tables):
    su = connect("vector_user")
    assert su.execute("SELECT count(*) FROM pg_class WHERE relname LIKE 'rag_production%'").fetchone()[0] == 0
    assert (
        su.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE column_name IN ('langchain_id', 'langchain_metadata', 'content_tsv')"
        ).fetchone()[0]
        == 0
    )


def test_rag_app_reads_the_two_views_and_the_registry_and_nothing_else(connect, tables):
    su = connect("vector_user")
    got = su.execute(
        index_view.PRIVILEGES, (index_view.CHILDREN, index_view.PARENTS, tables["children"], tables["parents"])
    ).fetchone()
    assert got == (True, True, True, False, False, False, 0)
    connect().execute("CREATE TABLE zz_probe (x int)")  # a table created later: no default privilege
    assert not su.execute("SELECT has_table_privilege('rag_app', 'zz_probe', 'SELECT')").fetchone()[0]
    connect().execute("DROP TABLE zz_probe")
    with pytest.raises(errors.InsufficientPrivilege):
        connect("rag_app").execute("CREATE TABLE zz_app_probe (x int)")


def test_every_statement_of_the_adapter_runs_as_rag_app(connect, tables):
    app = connect("rag_app")

    def run(text, view, params=None):
        return app.execute(sql.SQL(text).format(sql.Identifier(view), where=sql.SQL("")), params).fetchall()

    near = "[" + ",".join(["1.0", "0.5"] + ["0.1"] * 766) + "]"
    assert len(run(pg_index.VECTOR, "rag_chunks", {"vector": near, "k": 5})) == 5
    words = {"lang": "german", "query": "Garage Rathaus"}
    assert run(pg_index.FTS_AND, "rag_chunks", {**words, "k": 3})
    assert run(pg_index.FTS_OR, "rag_chunks", {**words, "or_terms": "'garag' | 'stellplaetz'", "remaining": 4})
    assert run(pg_index.COUNT_TERM, "rag_chunks", {"lang": "german", "term": "rathaus"})[0][0] > 0
    assert run(pg_index.COUNT_ALL, "rag_chunks")[0][0] == 6
    assert app.execute(pg_index.TSQUERY, words).fetchone()[0] == "'garag' & 'rathaus'"
    assert run(pg_index.PING, "rag_sections") and run(pg_index.CHILD_PROBE, "rag_chunks") == []
    rows = run(pg_index.SECTIONS, "rag_sections", {"section_ids": ["s0", "s1", "x"]})
    assert sorted(r[0] for r in rows) == ["s0", "s1"]
