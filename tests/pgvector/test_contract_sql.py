"""The index contract between the indexer and the assistant on a fresh cluster: after a publish as rag_indexer, rag_app
reads the collection's two typed views and the registry and nothing else (no privilege on a
generation's tables or on a table created later, no CREATE on public), and every statement of rag_core.adapters.pg_index
runs against the views as rag_app. RUN_DOCKER_TESTS=1."""

import os
import shutil
import uuid

import pytest
from psycopg import errors
from rag_testing import index_view

from indexer.index import ddl
from indexer.index import generations as gen
from rag_core.adapters import pg_index

TARGET = gen.Target.for_collection("rag")
TEXTS = (
    "Die Fahrradgarage am Campus hat dreihundert Stellplaetze",
    "Die Universitaetsbibliothek oeffnet montags um neun Uhr",
    "Die Druckerei nimmt Plakate und Poster an",
    "Die Garage in der Mensa ist fuer Dienstwagen reserviert",
    "Stellplaetze fuer Fahrraeder gibt es an der Mensa",
    "Die Oeffnungszeiten des Studienbueros in der Mensa",
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

    def run(text, view, params=None, **parts):
        return app.execute(pg_index._on(text, view, **parts), params).fetchall()

    near = "[" + ",".join(["1.0", "0.5"] + ["0.1"] * 766) + "]"
    assert len(run(pg_index.VECTOR, "rag_chunks", {"vector": near, "k": 5})) == 5
    words = {"lang": "german", "query": "Garage Mensa"}
    legs = [
        pg_index.Leg("fts_de", "german", "'garag' & 'mensa'", ()),
        pg_index.Leg("fts_en", "english", "'garag'", ()),
    ]
    and_params = {"fts_de_and": legs[0].and_text, "fts_en_and": legs[1].and_text}
    parts = {"rank": pg_index._rank(legs, "and"), "match": pg_index._match(legs, "and")}
    assert run(pg_index.FTS_AND, "rag_chunks", {**and_params, "k": 3}, **parts)
    seen = {"rank": pg_index._rank(legs, "or"), "match": pg_index._match(legs, "or"), "seen": parts["match"]}
    or_params = {**and_params, "fts_de_or": "'garag' | 'stellplaetz'", "fts_en_or": "'garag'", "remaining": 4}
    assert run(
        pg_index.FTS_OR, "rag_chunks", or_params, **seen
    )  # the German rows have a NULL fts_en and still come back
    assert run(pg_index.COUNT_TERM, "rag_chunks", {"term": "'mensa'"}, column="fts_de")[0][0] > 0
    assert run(pg_index.COUNT_ALL, "rag_chunks", column="fts_de")[0][0] == 6
    assert run(pg_index.COUNT_ALL, "rag_chunks", column="fts_en")[0][0] == 0  # no English row
    assert app.execute(pg_index.TSQUERY, words).fetchone()[0] == "'garag' & 'mensa'"
    groups = {"lang": "german", "terms": ["PO", "Mensa", "Garage", "der"], "owners": [0, 0, 1, 1]}
    # one row per group, its terms OR-ed; "der" is a stopword and leaves the second group with one term
    assert [r[0] for r in app.execute(pg_index.GROUP_QUERIES, groups).fetchall()] == ["'po' | 'mensa'", "'garag'"]
    assert run(pg_index.PING, "rag_sections") and run(pg_index.CHILD_PROBE, "rag_chunks") == []
    rows = run(pg_index.SECTIONS, "rag_sections", {"section_ids": ["s0", "s1", "x"]})
    assert sorted(r[0] for r in rows) == ["s0", "s1"]
