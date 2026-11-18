"""The cut and its reverse on a real cluster: the state
before the cut (two v2 generations with the legacy columns and four legacy views), the assistant's new statements
against the old ones on the same tables, deploy/sql/contract.sql (its refusal, the drop, the grants, run twice), a publish
after the cut read through the grants alone, deploy/sql/contract_reverse.sql (the frozen types, the old reads,
the old indexer's repoint). The tests run in file order on this module's own cluster, each from the state the one before
left. RUN_DOCKER_TESTS=1."""

import json
import os
import shutil
import uuid
from pathlib import Path

import pytest
from psycopg import errors, sql

from indexer.index import ddl
from indexer.index import generations as gen
from rag_core.adapters import pg_index

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (ROOT / "deploy" / "sql" / "contract.sql").read_text(encoding="utf-8")
REVERSE = (ROOT / "deploy" / "sql" / "contract_reverse.sql").read_text(encoding="utf-8")
TARGET = gen.Target.for_collection("rag")

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="image tests run only with RUN_DOCKER_TESTS=1 and a running Docker",
)
# the legacy views as the earlier DDL built them, verbatim: the legacy views before the cut, and the old indexer's repoint after a reverse
LEGACY_VIEWS = (
    'CREATE OR REPLACE VIEW "rag_production" AS SELECT langchain_id, content, embedding, langchain_metadata, '
    'fts AS content_tsv FROM "{children}"',
    'CREATE OR REPLACE VIEW "rag_production_parents" AS SELECT section_id AS paragraph_iid, content, metadata '
    'FROM "{parents}"',
)
# the assistant's old statements over the legacy views, verbatim
OLD_VECTOR = (
    'SELECT "langchain_id", "content", "embedding", "langchain_metadata", cosine_distance("embedding", %(vector)s::vector) '
    'as distance FROM "public".{} ORDER BY "embedding" <=> %(vector)s::vector LIMIT %(k)s'
)
OLD_FTS_AND = (
    "SELECT content, langchain_metadata as metadata, ts_rank(content_tsv, plainto_tsquery(%(lang)s, %(query)s)) as rank "
    "FROM {} WHERE content_tsv @@ plainto_tsquery(%(lang)s, %(query)s) ORDER BY rank DESC LIMIT %(k)s"
)
OLD_FTS_OR = (
    "SELECT content, langchain_metadata as metadata, ts_rank(content_tsv, to_tsquery(%(lang)s, %(or_terms)s)) as rank "
    "FROM {} WHERE content_tsv @@ to_tsquery(%(lang)s, %(or_terms)s) AND NOT content_tsv @@ plainto_tsquery(%(lang)s, "
    "%(query)s) ORDER BY rank DESC LIMIT %(remaining)s"
)
OLD_COUNT_TERM = "SELECT COUNT(*) FROM {} WHERE content_tsv @@ to_tsquery(%(lang)s, %(term)s)"
OLD_SECTIONS = "SELECT paragraph_iid, content as text, metadata FROM {} WHERE paragraph_iid = ANY(%(section_ids)s)"
TEXTS = (
    "Die Fahrradgarage am Bahnhof hat dreihundert Stellplaetze",
    "Die Stadtbuecherei oeffnet montags um neun Uhr",
    "Der Wertstoffhof nimmt Sperrmuell und Gruenschnitt an",
    "Die Garage im Rathaus ist fuer Dienstwagen reserviert",
    "Stellplaetze fuer Fahrraeder gibt es am Rathaus",
    "Die Oeffnungszeiten des Buergerbueros im Rathaus",
)
FROZEN_CHILDREN = [
    "langchain_id uuid",
    "content text",
    "embedding vector(768)",
    "langchain_metadata json",
    "content_tsv tsvector",
]
FROZEN_PARENTS = ["paragraph_iid text", "content text", "metadata jsonb"]


def vec(i, dims=768):
    v = [0.1] * dims
    v[i % dims] = 1.0
    return "[" + ",".join(str(x) for x in v) + "]"


def columns(conn, relation):
    return [
        r[0] + " " + r[1]
        for r in conn.execute(
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod) FROM pg_attribute a WHERE a.attrelid = %s::regclass "
            "AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum",
            (relation,),
        ).fetchall()
    ]


def generation(conn, n, legacy):
    """Generation n: two sections of three chunks; with legacy=True also the two legacy columns."""
    tables = TARGET.tables(n)
    for statement in ddl.create_tables(tables["parents"], tables["children"], "german", 768):
        conn.execute(statement)
    if legacy:
        conn.execute(
            f'ALTER TABLE "{tables["children"]}" ADD COLUMN langchain_id uuid UNIQUE, ADD COLUMN langchain_metadata json'
        )
    for s in range(2):
        section_id = f"g{n}s{s}"
        conn.execute(
            f'INSERT INTO "{tables["parents"]}" (section_id, document_id, collection, source, audience, part_index, '
            "content, metadata) VALUES (%s, 'doc', 'rag', 'example.web', 'public', %s, %s, %s::jsonb)",
            (
                section_id,
                s,
                " ".join(TEXTS[3 * s : 3 * s + 3]),
                json.dumps({"paragraph_iid": section_id, "title": f"T{s}"}),
            ),
        )
        for i in range(3 * s, 3 * s + 3):
            cid = uuid.UUID(int=n * 100 + i)
            if legacy:
                conn.execute(
                    f'INSERT INTO "{tables["children"]}" (chunk_id, section_id, document_id, collection, source, '
                    "audience, content, embedding, langchain_id, langchain_metadata) VALUES (%s, %s, 'doc', 'rag', "
                    "'example.web', 'public', %s, %s::vector, %s, %s::json)",
                    (
                        str(cid),
                        section_id,
                        TEXTS[i],
                        vec(i),
                        cid,
                        json.dumps({"paragraph_iid": section_id, "chunk_index": i % 3}),
                    ),
                )
            else:
                conn.execute(
                    f'INSERT INTO "{tables["children"]}" (chunk_id, section_id, document_id, collection, source, '
                    "audience, content, embedding) VALUES (%s, %s, 'doc', 'rag', 'example.web', 'public', %s, "
                    "%s::vector)",
                    (str(cid), section_id, TEXTS[i], vec(i)),
                )
    for statement in ddl.create_indexes(tables["parents"], tables["children"]):
        conn.execute(statement)
    return tables


@pytest.fixture(scope="module")
def box(port):
    """The state before the cut: generation 1 previous, 2 live, both v2 with the legacy columns; the four views on 2."""
    import psycopg

    # the pre-cut privileges, set here first and explicitly (idempotent on the current init scripts): default privileges
    # apply only to tables created after them, and a fresh cluster may be grant-only
    with psycopg.connect(
        host="127.0.0.1", port=port, dbname="vector_db", user="vector_user", password="m" * 64, autocommit=True
    ) as su:
        su.execute("GRANT CREATE ON SCHEMA public TO rag_app")
        su.execute("ALTER DEFAULT PRIVILEGES FOR ROLE rag_indexer IN SCHEMA public GRANT SELECT ON TABLES TO rag_app")
    conn = psycopg.connect(
        host="127.0.0.1", port=port, dbname="vector_db", user="rag_indexer", password="i" * 64, autocommit=True
    )
    gen.ensure_tables(conn)
    for n, status in ((1, "previous"), (2, "live")):
        tables = generation(conn, n, legacy=True)
        conn.execute(
            "INSERT INTO index_generations (collection, generation, status, layout, tables, published_at) "
            "VALUES ('rag', %s, %s, 'v2', %s::jsonb, now())",
            (n, status, json.dumps(tables)),
        )
    for statement in LEGACY_VIEWS:
        conn.execute(statement.format(**tables))
    for statement in ddl.typed_views("rag", tables["children"], tables["parents"]):
        conn.execute(statement)
    yield
    conn.close()


def run_as_superuser(connect, text):
    su = connect("vector_user")
    with su.transaction():
        su.execute(text)


def state(connect):
    su = connect("vector_user")
    one = lambda q: su.execute(q).fetchone()[0]  # noqa: E731
    return {
        "legacy views": one(
            "SELECT count(*) FROM pg_class WHERE relname IN ('rag_production', 'rag_production_parents')"
        ),
        "legacy columns": one(
            "SELECT count(*) FROM information_schema.columns WHERE table_name LIKE 'rag_g%%_chunks' "
            "AND column_name IN ('langchain_id', 'langchain_metadata')"
        ),
        "default select": one(
            "SELECT count(*) FROM pg_default_acl d, aclexplode(d.defaclacl) a WHERE d.defaclrole = "
            "'rag_indexer'::regrole AND a.grantee = 'rag_app'::regrole AND a.privilege_type = 'SELECT'"
        ),
        "create on public": one("SELECT has_schema_privilege('rag_app', 'public', 'CREATE')"),
    }


def pairs(app, legacy_chunks, legacy_sections, chunks, sections):
    """Every old statement on the legacy views and the new one on the typed views, as comparable rows."""
    out = []
    q = "[" + ",".join(["1.0", "0.5"] + ["0.1"] * 766) + "]"  # nearest to chunk 0, then chunk 1
    run = lambda text, view, params: app.execute(sql.SQL(text).format(sql.Identifier(view)), params).fetchall()  # noqa: E731
    old = run(OLD_VECTOR, legacy_chunks, {"vector": q, "k": 5})
    new = run(pg_index.VECTOR, chunks, {"vector": q, "k": 5})
    out.append(([(c, m["paragraph_iid"], d) for _, c, _, m, d in old], [tuple(r) for r in new]))
    for query in ("Fahrradgarage Bahnhof", "Rathaus", "Garage Rathaus"):
        params = {"lang": "german", "query": query, "k": 3}
        old = run(OLD_FTS_AND, legacy_chunks, params)
        new = run(pg_index.FTS_AND, chunks, params)
        out.append(([(c, m["paragraph_iid"], r) for c, m, r in old], [tuple(r) for r in new]))
    params = {
        "lang": "german",
        "query": "Stellplaetze Rathaus Garage",
        "or_terms": "'rathaus' | 'garag'",
        "remaining": 4,
    }
    old = run(OLD_FTS_OR, legacy_chunks, params)
    new = run(pg_index.FTS_OR, chunks, params)
    out.append(([(c, m["paragraph_iid"], r) for c, m, r in old], [tuple(r) for r in new]))
    for term in ("rathaus", "garag", "stellplaetz"):
        params = {"lang": "german", "term": term}
        out.append((run(OLD_COUNT_TERM, legacy_chunks, params), run(pg_index.COUNT_TERM, chunks, params)))
    ids = {
        "section_ids": [
            r[0] for r in app.execute(sql.SQL("SELECT section_id FROM {}").format(sql.Identifier(sections))).fetchall()
        ]
        + ["x"]
    }
    out.append((sorted(run(OLD_SECTIONS, legacy_sections, ids)), sorted(run(pg_index.SECTIONS, sections, ids))))
    return out


def test_the_cut_refuses_a_legacy_generation_and_changes_nothing(connect, box):
    indexer = connect()
    indexer.execute("UPDATE index_generations SET layout = 'legacy' WHERE collection = 'rag' AND generation = 1")
    before = state(connect)
    try:
        with pytest.raises(errors.RaiseException, match="contract refused: generation 1"):
            run_as_superuser(connect, CONTRACT)
        assert (
            state(connect)
            == before
            == {"legacy views": 2, "legacy columns": 4, "default select": 1, "create on public": True}
        )
    finally:
        indexer.execute("UPDATE index_generations SET layout = 'v2' WHERE collection = 'rag' AND generation = 1")


def test_the_cut_refuses_a_chunk_whose_ids_differ_from_its_legacy_columns(connect, box):
    indexer = connect()
    chunks = TARGET.tables(1)["children"]
    chunk_id, kept = indexer.execute(
        f'SELECT chunk_id, langchain_metadata::text FROM "{chunks}" ORDER BY chunk_id LIMIT 1'
    ).fetchone()
    indexer.execute(
        f'UPDATE "{chunks}" SET langchain_metadata = %s::json WHERE chunk_id = %s',
        (json.dumps({"paragraph_iid": "x"}), chunk_id),
    )
    before = state(connect)
    try:
        with pytest.raises(errors.RaiseException, match="contract refused: 1 rows of rag_g0001_chunks differ"):
            run_as_superuser(connect, CONTRACT)
        assert (
            state(connect)
            == before
            == {"legacy views": 2, "legacy columns": 4, "default select": 1, "create on public": True}
        )
    finally:
        indexer.execute(f'UPDATE "{chunks}" SET langchain_metadata = %s::json WHERE chunk_id = %s', (kept, chunk_id))


def test_the_typed_statements_return_the_legacy_rows_in_the_legacy_order(connect, box):
    app = connect("rag_app")
    compared = pairs(app, "rag_production", "rag_production_parents", "rag_chunks", "rag_sections")
    for old, new in compared:
        assert old == new
    # vector, "Rathaus", "Garage Rathaus", the count of the stem "stellplaetz", sections: none of them empty on both sides
    assert (
        len(compared[0][1]) == 5
        and compared[2][1]
        and compared[3][1]
        and compared[7][1][0][0] > 0
        and len(compared[-1][1]) == 2
    )


def test_the_reverse_refuses_before_the_cut_and_changes_nothing(connect, box):
    before = state(connect)
    su = connect("vector_user")
    viewdef = lambda: su.execute("SELECT pg_get_viewdef('rag_production'::regclass)").fetchone()[0]  # noqa: E731
    before_view = viewdef()
    with pytest.raises(errors.RaiseException, match="reverse refused: rag_g0002_chunks still has langchain_id"):
        run_as_superuser(connect, REVERSE)
    assert (
        state(connect)
        == before
        == {"legacy views": 2, "legacy columns": 4, "default select": 1, "create on public": True}
    )
    assert viewdef() == before_view
    assert "langchain_id" in before_view and "chunk_id" not in before_view


def test_the_cut_drops_the_legacy_relations_and_grants_the_typed_views(connect, box):
    run_as_superuser(connect, CONTRACT)
    after = {"legacy views": 0, "legacy columns": 0, "default select": 0, "create on public": False}
    assert state(connect) == after
    run_as_superuser(connect, CONTRACT)  # a second run changes nothing
    assert state(connect) == after
    app = connect("rag_app")
    assert app.execute("SELECT count(*) FROM rag_chunks").fetchone()[0] == 6
    assert (
        app.execute("SELECT generation FROM index_generations WHERE collection = 'rag' AND status = 'live'").fetchone()[
            0
        ]
        == 2
    )
    connect().execute("CREATE TABLE zz_contract_probe (x int)")  # a new table of rag_indexer: no default privilege
    assert (
        not connect("vector_user")
        .execute("SELECT has_table_privilege('rag_app', 'zz_contract_probe', 'SELECT')")
        .fetchone()[0]
    )
    connect().execute("DROP TABLE zz_contract_probe")
    with pytest.raises(errors.InsufficientPrivilege):
        app.execute("CREATE TABLE zz_app_probe (x int)")


def test_a_publish_after_the_cut_is_read_through_the_grants_alone(connect, box):
    indexer = connect()
    n, _ = gen.begin(indexer, TARGET, {"config_hash": "x"})
    assert n == 3
    tables = generation(indexer, n, legacy=False)
    gen.verify(indexer, TARGET, n, {"children": 6, "parents": 2})
    gen.publish(indexer, TARGET, n)
    assert gen.typed_view_tables(indexer, TARGET) == {
        "rag_chunks": tables["children"],
        "rag_sections": tables["parents"],
    }
    assert gen.relkind(indexer, "rag_production") is None and gen.relkind(indexer, "rag_production_parents") is None
    assert "langchain_id uuid" not in columns(indexer, tables["children"])
    su = connect("vector_user")
    assert (
        su.execute("SELECT has_table_privilege('rag_app', %s, 'SELECT')", (tables["children"],)).fetchone()[0] is False
    )
    app = connect("rag_app")
    assert app.execute("SELECT count(*) FROM rag_chunks c JOIN rag_sections s USING (section_id)").fetchone()[0] == 6


def test_the_reverse_serves_the_old_images(connect, box):
    run_as_superuser(connect, REVERSE)
    indexer, app, su = connect(), connect("rag_app"), connect("vector_user")
    assert columns(su, "rag_production") == FROZEN_CHILDREN
    assert columns(su, "rag_production_parents") == FROZEN_PARENTS
    owners = su.execute(
        "SELECT string_agg(pg_get_userbyid(relowner), ' ') FROM pg_class WHERE relname IN "
        "('rag_production', 'rag_production_parents')"
    ).fetchone()[0]
    assert owners == "rag_indexer rag_indexer"
    assert state(connect) == {"legacy views": 2, "legacy columns": 0, "default select": 1, "create on public": True}
    for old, new in pairs(app, "rag_production", "rag_production_parents", "rag_chunks", "rag_sections"):
        assert old == new  # the old image's reads equal the new ones on generation 3
    # the old indexer builds generation 4 with its legacy columns and repoints the legacy views
    tables = generation(indexer, 4, legacy=True)
    for statement in LEGACY_VIEWS:
        indexer.execute(statement.format(**tables))
    assert app.execute("SELECT count(*) FROM rag_production").fetchone()[0] == 6
    assert (
        su.execute("SELECT has_table_privilege('rag_app', %s, 'SELECT')", (tables["children"],)).fetchone()[0] is True
    )
