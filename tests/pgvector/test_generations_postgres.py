"""Index generations on a real pgvector cluster. A fresh
container runs the init scripts; every step runs as rag_indexer from the Mac through psycopg (no superuser), the reads
as rag_app. Each test uses its own collection, so the tests share one cluster. Needs a running Docker (Colima on the
Mac) and psycopg in the test venv: RUN_DOCKER_TESTS=1."""

import datetime
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
from indexer.index import ddl  # noqa: E402
from indexer.index import generations as gen  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="image tests run only with RUN_DOCKER_TESTS=1 and a running Docker",
)


def statuses(conn, target):
    return {r["generation"]: r["status"] for r in gen.generations(conn, target)}


def test_publish_repoints_both_views_keeps_the_view_and_the_previous(connect):
    conn, target = connect(), gen.Target("c2")
    build_v2(conn, target, ("a", "b"))
    app = connect("rag_app")
    oid = conn.execute("SELECT 'c2_chunks'::regclass::oid").fetchone()[0]
    assert app.execute("SELECT count(*) FROM c2_chunks", prepare=True).fetchone()[0] == 4
    n = build_v2(conn, target, ("a", "b", "c", "d"))
    assert n == 2 and statuses(conn, target) == {1: "previous", 2: "live"}
    assert gen.typed_view_tables(conn, target) == {"c2_chunks": "c2_g0002_chunks", "c2_sections": "c2_g0002_sections"}
    assert conn.execute("SELECT 'c2_chunks'::regclass::oid").fetchone()[0] == oid  # CREATE OR REPLACE keeps the view
    assert (
        app.execute("SELECT count(*) FROM c2_chunks", prepare=True).fetchone()[0] == 8
    )  # a prepared statement re-plans
    assert gen.relkind(conn, "c2_g0001_chunks") == "r"  # the previous generation stays


def test_rollback_repoints_to_the_previous_and_back(connect):
    conn, target = connect(), gen.Target("c3")
    with pytest.raises(gen.RollbackRefused, match="no previous generation"):  # kept: stateable on v2
        gen.rollback(conn, target)
    build_v2(conn, target)
    build_v2(conn, target)
    assert gen.rollback(conn, target) == {"live": 1, "previous": 2}
    assert gen.typed_view_tables(conn, target)["c3_chunks"] == "c3_g0001_chunks"
    assert connect("rag_app").execute("SELECT count(*) FROM c3_chunks").fetchone()[0] == 6
    assert gen.rollback(conn, target) == {"live": 2, "previous": 1}
    assert gen.typed_view_tables(conn, target)["c3_chunks"] == "c3_g0002_chunks"


def test_a_failed_check_after_the_commit_says_the_generation_is_published(connect):
    conn, target = connect(), gen.Target("c10")
    build_v2(conn, target)
    n, _ = gen.begin(conn, target, {})
    tables = target.tables(n)
    make_v2_pair(conn, tables["children"], tables["parents"], contents=())  # no row: the check after the commit fails
    with pytest.raises(gen.PublishedUnchecked, match="is published, but the check after the commit failed") as caught:
        gen.publish(conn, target, n)
    assert (caught.value.previous, caught.value.seconds >= 0) == (1, True)
    assert gen.typed_view_tables(conn, target) == {
        "c10_chunks": "c10_g0002_chunks",
        "c10_sections": "c10_g0002_sections",
    }
    assert statuses(conn, target) == {1: "previous", 2: "live"}


def test_a_reader_holding_the_view_delays_the_publish_and_a_failed_publish_changes_nothing(connect, monkeypatch):
    conn, target = connect(), gen.Target("c4")
    build_v2(conn, target)
    n = build_v2(conn, target, ("y",), publish=False)
    reader = connect("rag_app", autocommit=False)
    reader.execute("SELECT count(*) FROM c4_chunks").fetchone()  # an open transaction holds the view
    monkeypatch.setattr(gen, "LOCK_TIMEOUT", "300ms")
    monkeypatch.setattr(gen, "PUBLISH_ATTEMPTS", 2)
    monkeypatch.setattr(gen, "RETRY_SLEEP_S", 0)
    with pytest.raises(gen.GenerationError, match="the views are unchanged"):
        gen.publish(conn, target, n)
    assert gen.typed_view_tables(conn, target)["c4_chunks"] == "c4_g0001_chunks"
    assert statuses(conn, target) == {1: "live", 2: "staging"}
    reader.rollback()
    gen.publish(conn, target, n)
    assert gen.typed_view_tables(conn, target) == {"c4_chunks": "c4_g0002_chunks", "c4_sections": "c4_g0002_sections"}


def test_housekeeping_keeps_live_and_the_newest_previous_and_never_cascades(connect):
    conn, target = connect(), gen.Target("c5")
    for _ in range(4):
        build_v2(conn, target)
    assert statuses(conn, target) == {1: "previous", 2: "previous", 3: "previous", 4: "live"}
    conn.execute("CREATE VIEW c5_keepme AS SELECT * FROM c5_g0002_chunks")  # a view that still reads generation 2
    make_v2_pair(conn, "c5_staging_1700000000", "c5_staging_1700000000_parents")
    make_v2_pair(conn, "c5_old_1690000000", "c5_old_1690000000_parents")
    result = gen.housekeep(conn, target)
    assert sorted(result["dropped_tables"]) == ["c5_g0001_chunks", "c5_g0001_sections", "c5_g0002_sections"]
    assert result["kept"] == ["4 (live)", "3 (previous)"] and result["freed_bytes"] > 0
    assert len(result["errors"]) == 1 and result["errors"][0].startswith(
        "c5_g0002_chunks: cannot drop table c5_g0002_chunks"
    )
    assert statuses(conn, target) == {1: "dropped", 2: "previous", 3: "previous", 4: "live"}
    assert gen.relkind(conn, "c5_staging_1700000000") == "r"  # stale tables only on request
    assert gen.status(conn, target)["stale_tables"] == 4
    result = gen.housekeep(conn, target, stale=True)
    assert sorted(result["dropped_tables"]) == [
        "c5_old_1690000000",
        "c5_old_1690000000_parents",
        "c5_staging_1700000000",
        "c5_staging_1700000000_parents",
    ]
    assert gen.relkind(conn, "c5_g0002_chunks") == "r" and gen.relkind(conn, "c5_g0003_chunks") == "r"
    conn.execute("DROP VIEW c5_keepme")
    assert gen.housekeep(conn, target)["dropped_tables"] == ["c5_g0002_chunks"]
    assert statuses(conn, target) == {1: "dropped", 2: "dropped", 3: "previous", 4: "live"}


def test_the_build_lock_is_exclusive_queryable_and_shared_with_the_cli(connect):
    conn, target = connect(), gen.Target("c7")
    first = gen.BuildLock(target, "process-corpus reuse_inputs", connect).acquire()
    with pytest.raises(gen.BuildBusy, match="a build holds the lock since .*process-corpus reuse_inputs"):
        gen.BuildLock(target, "rollback", connect).acquire()
    lock = gen.status(conn, target)["lock"]
    assert (
        lock["locked"]
        and lock["holder"] == "process-corpus reuse_inputs"
        and lock["pid"] == first.conn.info.backend_pid
    )
    lines = []
    assert gen.main(["housekeep"], target=target, connect_fn=connect, out=lines.append) == 1
    assert lines[0].startswith("busy: a build holds the lock since")
    first.release()
    assert gen.status(conn, target)["lock"] == {"locked": False, "holder": None, "started_at": None, "pid": None}
    crashed = gen.BuildLock(target, "crashed build", connect).acquire()
    crashed.conn.close()  # the session lock dies with its connection; the stale row is ignored
    time.sleep(0.5)
    assert gen.status(conn, target)["lock"]["locked"] is False
    gen.BuildLock(target, "next", connect).acquire().release()


def test_a_generation_that_fails_verify_is_marked_and_dropped_at_the_next_housekeeping(connect):
    conn, target = connect(), gen.Target("c8")
    build_v2(conn, target)
    n, _ = gen.begin(conn, target, {"git_commit": None, "config_hash": "abc"})
    tables = target.tables(n)
    written = make_v2_pair(conn, tables["children"], tables["parents"])
    conn.execute(f'DROP INDEX "{tables["children"]}_fts_de"')
    with pytest.raises(gen.GenerationError, match="no valid gin on fts_de index"):
        gen.verify(conn, target, n, written)
    with pytest.raises(gen.GenerationError, match="the writer inserted"):
        gen.verify(conn, target, 1, {"children": 7, "parents": 3})
    gen.mark_failed(conn, target, n, "verify failed: no valid gin on fts_de index")
    row = gen.generations(conn, target)[1]
    assert (row["status"], row["detail"], row["config_hash"]) == (
        "failed",
        "verify failed: no valid gin on fts_de index",
        "abc",
    )
    assert gen.housekeep(conn, target)["dropped_tables"] == ["c8_g0002_chunks", "c8_g0002_sections"]
    assert gen.status(conn, target)["live"]["generation"] == 1


def test_compare_reports_identical_and_different(connect):
    conn, target = connect(), gen.Target("c9")
    build_v2(conn, target, ("alpha", "beta", "gamma"), publish=False)
    build_v2(conn, target, ("alpha", "beta", "gamma"), publish=False)
    build_v2(conn, target, ("alpha", "beta", "delta"), publish=False)
    same = gen.compare(conn, target, 1, 2)
    assert same["identical"] and same["generations"] == {
        1: {"parents": 3, "children": 6},
        2: {"parents": 3, "children": 6},
    }
    assert gen.compare_lines(same, 1, 2)[-1] == "RESULT identical"
    other = gen.compare(conn, target, 1, 3)
    assert not other["identical"]
    assert other["parents"] == {"only_a": 1, "only_b": 1} and other["children"] == {"only_a": 2, "only_b": 2}
    lines = gen.compare_lines(other, 1, 3)
    assert "source archive/pdf: 3 | 3" in lines and lines[-1] == "RESULT DIFFERENT"
    assert "only in 1: https://docs.example.org/doc.pdf page 3" in lines


def test_cli_publish_puts_a_staging_generation_live_and_keeps_the_other_staging(connect, monkeypatch, tmp_path):
    """Two staging builds, then publish one; nothing is housekept."""
    from indexer import cli, gates

    conn, target = connect(), gen.Target("c11")
    first = build_v2(conn, target)
    a, b = build_v2(conn, target, publish=False), build_v2(conn, target, ("x", "y"), publish=False)
    lines = []
    assert cli.publish(target, b, connect, lines.append, str(tmp_path)) == 0
    assert statuses(conn, target) == {first: "previous", a: "staging", b: "live"}
    assert gen.typed_view_tables(conn, target)["c11_chunks"] == target.tables(b)["children"]
    assert gen.relkind(conn, target.tables(a)["children"]) == "r"
    assert lines[0].startswith(f"published generation {b} (previous {first}) in ")
    assert len(lines) == 1
    assert json.loads((tmp_path / gates.STATS_NAME).read_text())["parents"] == 2
    assert cli.publish(target, b, connect, lines.append, str(tmp_path)) == 1
    assert lines[-1] == f"refused: generation {b} is live, not staging"


# --- schema v2: generations of typed tables ----------------------------------------------


def make_v2_pair(conn, children, parents, contents=("alpha", "beta", "gamma"), dims=768, lang="de"):
    """A v2 pair from ddl plus hand INSERTs (no writer): two chunks per section. Returns the
    written counts."""
    sections, chunks = ddl.create_tables(parents, children, dims)
    conn.execute(sections)
    conn.execute(chunks)
    vector = "[" + ",".join(["0.1"] * dims) + "]"
    for page, text in enumerate(contents, start=1):
        section_id = str(uuid.uuid4())
        meta = {
            "url": "https://docs.example.org/doc.pdf",
            "page_number": page,
            "part_index": page - 1,
            "origin": "archive",
            "source_type": "pdf",
            "doc_key": "doc",
            "section_id": section_id,
            "scrapedAt": str(time.time()),
        }
        conn.execute(
            f'INSERT INTO "{parents}" (section_id, document_id, collection, source, audience, part_index, content, '
            f"metadata, lang) VALUES (%s, 'doc', 'example', 'archive', 'public', %s, %s, %s::jsonb, %s)",
            (section_id, page - 1, text, json.dumps(meta), lang),
        )
        for index in range(2):
            conn.execute(
                f'INSERT INTO "{children}" (chunk_id, section_id, document_id, collection, source, audience, content, '
                f"embedding, lang) VALUES (%s, %s, 'doc', 'example', 'archive', 'public', %s, %s::vector, %s)",
                (f"{section_id}:{index}", section_id, f"Titel: {text} {index}", vector, lang),
            )
    for statement in ddl.create_indexes(parents, children):
        conn.execute(statement)
    return {"children": 2 * len(contents), "parents": len(contents)}


def build_v2(conn, target, contents=("alpha", "beta", "gamma"), publish=True):
    n, _ = gen.begin(conn, target, {"config_hash": "x"})
    tables = target.tables(n)
    written = make_v2_pair(conn, tables["children"], tables["parents"], contents)
    gen.verify(conn, target, n, written)
    if publish:
        gen.publish(conn, target, n)
    return n


def test_ensure_tables_grants_the_registry_and_a_publish_grants_the_two_views(connect):
    conn, target, app = connect(), gen.Target("c31"), connect("rag_app")
    gen.ensure_tables(conn)
    can_read = "SELECT has_table_privilege(current_user, 'index_generations', 'SELECT')"
    assert app.execute(can_read).fetchone() == (True,)
    conn.execute("REVOKE SELECT ON index_generations FROM rag_app")  # a registry whose grant was lost (killed indexer)
    assert app.execute(can_read).fetchone() == (False,)
    gen.ensure_tables(conn)
    assert app.execute(can_read).fetchone() == (True,)
    n = build_v2(conn, target)  # begin, write, verify and publish one v2 generation
    got = app.execute(
        "SELECT has_table_privilege(current_user, 'c31_chunks', 'SELECT'), "
        "has_table_privilege(current_user, 'c31_sections', 'SELECT'), "
        "has_table_privilege(current_user, %s, 'SELECT'), has_schema_privilege(current_user, 'public', 'CREATE')",
        (target.tables(n)["children"],),
    ).fetchone()
    assert got == (True, True, False, False)
    assert app.execute("SELECT count(*) FROM c31_sections").fetchone()[0] > 0


def test_a_type_mismatch_leaves_every_view_unchanged(connect):
    conn, target = connect(), gen.Target("c22")
    build_v2(conn, target)
    n, _ = gen.begin(conn, target, {})
    tables = target.tables(n)
    make_v2_pair(conn, tables["children"], tables["parents"], dims=384)
    with pytest.raises(gen.GenerationError, match="the views are unchanged"):
        gen.publish(conn, target, n)
    assert gen.typed_view_tables(conn, target) == {
        "c22_chunks": "c22_g0001_chunks",
        "c22_sections": "c22_g0001_sections",
    }
    assert statuses(conn, target) == {1: "live", 2: "staging"}


def test_fts_de_and_fts_en_generated(connect):
    """fts_de is the German vector of every chunk, fts_en the English one of the chunks with lang = en and NULL for
    the rest; the gate finds a GIN index on each."""
    conn = connect()
    make_v2_pair(conn, "c23_chunks", "c23_sections", contents=("Prüfungen", "Prüfungen"), lang="de")
    make_v2_pair(conn, "c23en_chunks", "c23en_sections", contents=("examinations", "examinations"), lang="en")
    check = "SELECT bool_and(fts_de = to_tsvector('german', content)), count(fts_en), count(*) FROM "
    assert conn.execute(check + "c23_chunks").fetchone() == (True, 0, 4)
    assert conn.execute(check + "c23en_chunks").fetchone() == (True, 4, 4)
    assert conn.execute("SELECT bool_and(fts_en = to_tsvector('english', content)) FROM c23en_chunks").fetchone() == (
        True,
    )
    assert (
        conn.execute("SELECT count(*) FROM c23_chunks WHERE fts_de @@ to_tsquery('german', 'prufung')").fetchone()[0]
        == 4
    )
    assert (
        conn.execute("SELECT count(*) FROM c23en_chunks WHERE fts_en @@ to_tsquery('english', 'examin')").fetchone()[0]
        == 4
    )
    rows = conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'c23en_chunks'").fetchall()
    assert gen.missing_indexes("children", [(r[0], True, "UNIQUE" in r[0]) for r in rows]) == []


def test_verify_v2_counts_orphans_and_indexes(connect):
    conn, target = connect(), gen.Target("c24")
    n, _ = gen.begin(conn, target, {})
    tables = target.tables(n)
    written = make_v2_pair(conn, tables["children"], tables["parents"])
    assert gen.verify(conn, target, n, written) == {"children": 6, "parents": 3}
    conn.execute(
        f'UPDATE "{tables["children"]}" SET section_id = %s WHERE chunk_id = (SELECT min(chunk_id) FROM "{tables["children"]}")',
        ("nowhere",),
    )
    with pytest.raises(gen.GenerationError, match="verify failed: .*orphan chunks 1"):
        gen.verify(conn, target, n, written)
    m, _ = gen.begin(conn, target, {})
    other = target.tables(m)
    make_v2_pair(conn, other["children"], other["parents"])
    conn.execute(f'DROP INDEX "{other["children"]}_fts_en"')
    with pytest.raises(gen.GenerationError, match="no valid gin on fts_en index"):
        gen.verify(conn, target, m)


def test_compare_of_two_v2_generations_joins_chunks_by_section(connect):
    conn, target = connect(), gen.Target("c30")
    build_v2(conn, target)
    build_v2(conn, target, publish=False)
    report = gen.compare(conn, target, 1, 2)
    assert report["identical"] and report["generations"] == {
        1: {"parents": 3, "children": 6},
        2: {"parents": 3, "children": 6},
    }
    conn.execute(
        'UPDATE "c30_g0002_chunks" SET content = content || \' x\' WHERE chunk_id = (SELECT min(chunk_id) FROM "c30_g0002_chunks")'
    )
    assert gen.compare(conn, target, 1, 2)["children"] == {"only_a": 1, "only_b": 1}


def test_housekeep_drops_v2_tables_by_the_row_names(connect):
    conn, target = connect(), gen.Target("c27")
    for _ in range(3):
        build_v2(conn, target)
    result = gen.housekeep(conn, target)
    assert sorted(result["dropped_tables"]) == ["c27_g0001_chunks", "c27_g0001_sections"]
    assert statuses(conn, target) == {1: "dropped", 2: "previous", 3: "live"}


def test_v2_rows_keep_the_keys_older_images_read(connect):
    conn, target = connect(), gen.Target("c28")
    build_v2(conn, target)
    row = gen.generations(conn, target)[0]
    assert row["layout"] == "v2" and sorted(row["tables"]) == ["children", "parents"]
    assert [gen.relkind(conn, t) for t in row["tables"].values()] == ["r", "r"]


def test_write_v2_fills_a_pair_the_typed_views_read(connect, tmp_path):  # write_v2 on a real cluster
    from indexer.chunking.units import Child, Parent
    from indexer.index import cache, writer
    from indexer.settings import load_settings
    from rag_core.settings.profile import Source

    profile = load_settings(
        {
            "RAG_PROFILE": str(ROOT / "profiles" / "example" / "profile.yaml"),
            "RAG_OVERLAY": str(ROOT / "deploy" / "env" / "ci-example.yaml"),
        },
        secrets_dir=str(tmp_path / "secrets"),
    ).profile
    extra = {  # the example profile lists one public source; these two give the source column other values
        "pdfs": Source(connector="example.pdfs", audience="public"),
        "maps": Source(connector="example.maps", audience="public"),
    }
    sources = {**profile.ingest.sources, **extra}
    profile = profile.model_copy(update={"ingest": profile.ingest.model_copy(update={"sources": sources})})
    parents = [
        Parent(
            "Öffnungszeiten\x00 in der Bibliothek",
            {
                "section_id": "s1",
                "part_index": 0,
                "filename": "Bibliothek",
                "origin": "website",
                "source_type": "page",
                "url": "https://example.org/r",
                "doc_key": "https://example.org/r",
                "page_number": -1,
            },
            (Child("Bibliothek: Montag"), Child("Bibliothek: Montag"), Child("Bibliothek: Dienstag")),
            "Bibliothek: ",
            "folder",
        ),
        Parent(
            "§ 1 Geltung",
            {
                "section_id": "s2",
                "part_index": 0,
                "filename": "DA",
                "origin": "archive",
                "source_type": "pdf",
                "doc_key": "https://docs.example.org/da.pdf",
                "page_number": 2,
                "doc_year": "2024",
                "tags": ("a", "b"),
                "regulation": "po",
                "programme": ("bsc-informatik", "msc-informatik"),
                "po_version": (2023,),
                "valid_from": "2022-01-13",
                "lang": "en",
                "twin_key": "https://docs.example.org/da#s",
            },
            (Child("DA · § 1: Geltung"),),
            "DA · § 1: ",
            "example.pdfs",
        ),
        Parent(
            "Wertstoffhof",
            {
                "section_id": "s3",
                "part_index": 0,
                "filename": "Wertstoffhof",
                "origin": "website",
                "source_type": "map",
                "doc_key": "https://map.example.org/#uid=1",
            },
            (Child("Wertstoffhof"),),
            "",
            "example.maps",
        ),
    ]
    vector = [1 / 3, 0.1, 2.0000001] + [0.5] * 765

    async def embed(texts):
        return [vector for _ in texts]

    conn, target = connect(), gen.Target("c29")
    n, _ = gen.begin(conn, target, {"config_hash": "x"})
    key = cache.key(target.collection, profile.retrieval.embedding, None)
    written = writer.write_v2(conn, embed, target.tables(n), parents, profile, generation=n, cache_key=key)
    assert {k: written[k] for k in ("children", "parents")} == {"children": 4, "parents": 3}
    assert gen.verify(conn, target, n, written) == {"children": 4, "parents": 3}
    gen.publish(conn, target, n)
    app = connect("rag_app")
    joined = app.execute(
        "SELECT s.content, s.metadata->>'source', count(c.*), "
        "bool_and(c.fts_de = to_tsvector('german', c.content)) FROM c29_sections s "
        "JOIN c29_chunks c ON c.section_id = s.section_id GROUP BY 1, 2"
    ).fetchall()
    assert sorted(joined) == sorted([("Öffnungszeiten in der Bibliothek", "Bibliothek", 2, True), ("§ 1 Geltung", "DA", 1, True),
                                     ("Wertstoffhof", "Wertstoffhof", 1, True)])  # fmt: skip
    assert app.execute("SELECT metadata->'tags' FROM c29_sections WHERE content = '§ 1 Geltung'").fetchone()[0] == [
        "a",
        "b",
    ]
    typed = app.execute("SELECT source, audience, year, page, header FROM c29_sections").fetchall()
    assert sorted(typed) == sorted([("folder", "public", None, None, "Bibliothek: "),
                                    ("example.pdfs", "public", 2024, 2, "DA · § 1: "),
                                    ("example.maps", "public", None, None, None)])  # fmt: skip
    assert (
        app.execute("SELECT count(*) FROM c29_chunks WHERE embedding = %s::vector", (str(vector),)).fetchone()[0] == 4
    )
    # the document metadata went through COPY into typed arrays and dates; a document without it has NULL, never {}
    assert app.execute(
        "SELECT regulation, programme, po_version, degree, valid_from, lang, twin_key, archived FROM c29_sections "
        "WHERE content = '§ 1 Geltung'"
    ).fetchone() == ("po", ["bsc-informatik", "msc-informatik"], [2023], None, datetime.date(2022, 1, 13), "en",
                     "https://docs.example.org/da#s", False)  # fmt: skip
    assert app.execute(
        "SELECT regulation, programme, po_version, valid_from, twin_key FROM c29_sections WHERE content = 'Wertstoffhof'"
    ).fetchone() == (None, None, None, None, None)
    assert app.execute(
        "SELECT count(*), count(fts_en), count(programme) FILTER (WHERE programme = ARRAY['bsc-informatik', "
        "'msc-informatik']) FROM c29_chunks WHERE section_id = (SELECT section_id FROM c29_sections "
        "WHERE content = '§ 1 Geltung')"
    ).fetchone() == (1, 1, 1)  # the chunk carries the document's metadata and its English vector
