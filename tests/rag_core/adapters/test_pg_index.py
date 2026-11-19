"""The psycopg reads: the statements over the typed views, their result shapes, one pool without prepared
statements. A fake pool records what would be sent."""

import asyncio
import datetime as dt
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from psycopg_pool import AsyncConnectionPool
from rag_core.adapters import pg_index
from rag_core.domain.filters import Filter


class Cursor:
    def __init__(self, rows):
        self.rows = rows

    async def fetchall(self):
        return self.rows

    async def fetchone(self):
        return self.rows[0] if self.rows else None


class Conn:
    def __init__(self, answers):
        self.answers, self.sent, self.binary = answers, [], []

    async def execute(self, query, params=None, **kw):  # binary=True on the keyword rows
        text = query.as_string(None) if hasattr(query, "as_string") else query
        self.sent.append((text, dict(params or {})))
        if kw.get("binary"):
            self.binary.append(text)
        for marker, rows in self.answers:
            if marker in text:
                if isinstance(rows, Exception):
                    raise rows
                return Cursor(rows(params) if callable(rows) else rows)
        return Cursor([])


class Pool:
    def __init__(self, answers):
        self.conn = Conn(answers)

    def connection(self):
        conn = self.conn

        class Context:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return Context()


def test_the_statements_read_the_typed_views():
    pool: Any = Pool([("cosine_distance", [("Kind", "p1", 0.25)])])
    hits = asyncio.run(pg_index.PgVectorStore(pool, "example_chunks").search([0.5, 1.0], 30))
    text, params = pool.conn.sent[0]
    assert text == (
        'SELECT "content", "section_id", cosine_distance("embedding", %(vector)s::vector) as distance '
        'FROM "public"."example_chunks" ORDER BY "embedding" <=> %(vector)s::vector LIMIT %(k)s'
    )
    assert params == {"vector": "[0.5, 1.0]", "k": 30}
    assert [(h.page_content, dict(h.metadata)) for h in hits] == [
        ("Kind", {"section_id": "p1", "retriever": "semantic", "child_chunk_semantic_search_score": 0.25})
    ]
    assert "fts_de @@ plainto_tsquery(%(lang)s, %(query)s){where} ORDER BY rank DESC LIMIT %(k)s" in pg_index.FTS_AND
    assert "section_id = ANY(%(section_ids)s){where}" in pg_index.SECTIONS


def test_a_filter_binds_its_values_into_both_legs_and_the_section_lookup():
    flt = Filter(po_version=(2023,), regulation=("mhb",), date=dt.date(2026, 10, 1))
    where = (
        " AND archived = false"
        " AND (po_version && %(f_po_version)s::int[] OR po_version IS NULL)"
        " AND (regulation = ANY(%(f_regulation)s::text[]) OR regulation IS NULL)"
        " AND (valid_from <= %(f_date)s::date OR valid_from IS NULL OR po_version IS NULL)"
    )
    bound = {"f_po_version": [2023], "f_regulation": ["mhb"], "f_date": dt.date(2026, 10, 1)}
    pool: Any = Pool(
        [("cosine_distance", []), ("ts_rank(fts_de, plainto_tsquery", []), ("ts_rank(fts_de, to_tsquery", [])]
    )
    asyncio.run(pg_index.PgVectorStore(pool, "v").search([0.5], 3, filter=flt))
    text, params = pool.conn.sent[0]
    assert text.endswith(f'FROM "public"."v" WHERE{where[4:]} ORDER BY "embedding" <=> %(vector)s::vector LIMIT %(k)s')
    assert params == {"vector": "[0.5]", "k": 3, **bound}
    pool = Pool(
        [
            ("ts_rank(fts_de, plainto_tsquery", []),
            ("::text as tsq", [("'garag'",)]),
            ('COUNT(*) FROM "v" WHERE', [(0,)]),
        ]
    )
    asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Garage", 2, flt))
    and_sql, and_params = pool.conn.sent[0]
    or_sql, or_params = next((t, p) for t, p in pool.conn.sent if "to_tsquery(%(lang)s, %(or_terms)s)" in t)
    assert f"plainto_tsquery(%(lang)s, %(query)s){where} ORDER BY rank" in and_sql  # the OR fallback carries it too
    assert f"plainto_tsquery(%(lang)s, %(query)s){where} ORDER BY rank" in or_sql
    assert and_params.items() >= bound.items() and or_params.items() >= bound.items()
    pool = Pool([("section_id = ANY", [])])
    asyncio.run(pg_index.PgSections(pool, "v").get_many(["a"], flt))
    text, params = pool.conn.sent[0]
    assert text.endswith(f"WHERE section_id = ANY(%(section_ids)s){where}")
    assert params == {"section_ids": ["a"], **bound}


def test_no_filter_adds_no_condition_and_an_empty_filter_only_leaves_archived_rows_out():
    assert pg_index.where_of(None) == ("", {})
    assert pg_index.where_of(Filter()) == (" AND archived = false", {})
    assert pg_index.where_of(Filter(include_archived=True)) == ("", {})
    assert pg_index.where_of(Filter(degree=("master",)), " WHERE") == (
        " WHERE archived = false AND (degree && %(f_degree)s::text[] OR degree IS NULL)",
        {"f_degree": ["master"]},
    )


def test_no_statement_names_a_legacy_relation_or_column():  # no legacy read is left behind
    names = ("VECTOR", "FTS_AND", "FTS_OR", "TSQUERY", "COUNT_ALL", "COUNT_TERM", "SECTIONS", "PING", "CHILD_PROBE")
    statements = [getattr(pg_index, n) for n in names]
    for legacy in ("langchain", "content_tsv", "paragraph_iid", "_production", "_parents"):
        assert not [s for s in statements if legacy in s], legacy


def test_max_distance_cuts_far_hits():
    pool: Any = Pool([("cosine_distance", [("near", "a", 0.2), ("far", "b", 0.9)])])
    hits = asyncio.run(pg_index.PgVectorStore(pool, "v").search([0.0], 2, max_distance=0.5))
    assert [h.page_content for h in hits] == ["near"]


def test_an_empty_vector_result_and_a_cached_term():  # the two branches the others miss
    empty: Any = Pool([])
    assert asyncio.run(pg_index.PgVectorStore(empty, "v").search([0.0], 3)) == []
    pool: Any = Pool(
        [
            ("::text as tsq", [("'garag'",)]),
            ('SELECT COUNT(*) FROM "v" WHERE', [(1,)]),
            ('SELECT COUNT(*) FROM "v"', [(100,)]),
        ]
    )
    fts = pg_index.PgFullText(pool, "v", "german")
    for _ in range(2):
        asyncio.run(fts.search("Garage", 2))
    assert (
        len([t for t, _ in pool.conn.sent if t.startswith('SELECT COUNT(*) FROM "v" WHERE')]) == 1
    )  # the second: from the cache


def test_the_or_fallback_supplements_and_skips_seen_texts():
    pool: Any = Pool(
        [
            ("ts_rank(fts_de, plainto_tsquery", [("eins", "a", 0.5)]),
            ("::text as tsq", [("'garag' & 'stellplatz'",)]),
            ('SELECT COUNT(*) FROM "v" WHERE', [(1,)]),
            ('SELECT COUNT(*) FROM "v"', [(100,)]),
            ("ts_rank(fts_de, to_tsquery", [("eins", "a", 0.4), ("zwei", "b", 0.3)]),
        ]
    )
    hits = asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Garage Stellplatz", 3))
    assert [
        (h.page_content, h.metadata["section_id"], h.metadata["retriever"], h.metadata["fts_rank"]) for h in hits
    ] == [
        ("eins", "a", "fts", 0.5),
        ("zwei", "b", "fts", 0.3),
    ]
    or_params = [p for t, p in pool.conn.sent if "to_tsquery(%(lang)s, %(or_terms)s)" in t][0]
    assert or_params == {
        "lang": "german",
        "query": "Garage Stellplatz",
        "or_terms": "'garag' | 'stellplatz'",
        "remaining": 2,
    }
    # the AND and OR rows are read in binary (text would round fts_rank)
    assert len(pool.conn.binary) == 2 and all("ts_rank(fts_de" in t for t in pool.conn.binary)


def test_the_or_fallback_logs_no_terms_of_the_question(caplog):  # no question text in any log
    caplog.set_level(
        logging.INFO, logger="rag_core.adapters.pg_index"
    )  # a test of the evals CLI may leave "rag_core" at WARNING
    pool: Any = Pool(
        [
            ("ts_rank(fts_de, plainto_tsquery", []),
            ("::text as tsq", [("'garag' & 'stellplatz'",)]),
            ('SELECT COUNT(*) FROM "v" WHERE', [(1,)]),
            ('SELECT COUNT(*) FROM "v"', [(100,)]),
            ("ts_rank(fts_de, to_tsquery", [("eins", "a", 0.4)]),
        ]
    )
    assert asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Garage Stellplatz", 3))
    assert caplog.records == []


def test_a_filtered_frequent_term_is_logged_as_a_count_never_as_the_word(
    caplog,
):  # no question text in any log
    caplog.set_level(logging.INFO, logger="rag_core.adapters.pg_index")
    counts = lambda p: [(50,)] if p["term"] in ("stadt", "amt") else [(1,)]  # noqa: E731
    pool: Any = Pool(
        [
            ("ts_rank(fts_de, plainto_tsquery", []),
            ("::text as tsq", [("'stadt' & 'amt' & 'garag'",)]),
            ('SELECT COUNT(*) FROM "v" WHERE', counts),
            ('SELECT COUNT(*) FROM "v"', [(100,)]),
            ("ts_rank(fts_de, to_tsquery", []),
        ]
    )
    asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Stadt Amt Garage", 3))
    assert [r.getMessage() for r in caplog.records] == ["[FTS] Filtered 2 high-freq terms"]


def test_a_failed_search_logs_the_class_never_the_message(caplog):  # a tsquery error quotes the terms of the question
    down: Any = Pool([("ts_rank", RuntimeError('syntax error in tsquery: "garage"'))])
    with pytest.raises(RuntimeError):
        asyncio.run(pg_index.PgFullText(down, "v", "german").search("Garage", 2))
    assert [r.getMessage() for r in caplog.records] == ["[PostgresFTSRetriever] Error during search: RuntimeError"]


def test_an_and_query_that_fills_k_needs_no_or_fallback():
    rows = [("eins", "a", 0.5), ("zwei", "b", 0.4)]
    pool: Any = Pool([("ts_rank(fts_de, plainto_tsquery", rows)])
    hits = asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Garage", 2))
    assert [h.page_content for h in hits] == ["eins", "zwei"]
    assert len(pool.conn.sent) == 1  # no tsquery lookup, no term counts, no OR query


def test_the_term_cache_is_per_instance():
    pool: Any = Pool([])  # Any for basedpyright
    a, b = pg_index.PgFullText(pool, "v", "german"), pg_index.PgFullText(pool, "v", "german")
    a.term_is_frequent["x"] = True
    assert b.term_is_frequent == {} and a.total_rows is None


def test_sections_and_their_failure_rule():
    pool: Any = Pool([("ANY(", [("p1", "Abschnitt", {"title": "T"})])])
    found = asyncio.run(pg_index.PgSections(pool, "example_sections").get_many(["p1", "p2"]))
    assert {k: (s.section_id, s.text, dict(s.metadata)) for k, s in found.items()} == {
        "p1": ("p1", "Abschnitt", {"title": "T"})
    }
    assert pool.conn.sent[0] == (
        'SELECT section_id, content as text, metadata FROM "example_sections" WHERE section_id = ANY(%(section_ids)s)',
        {"section_ids": ["p1", "p2"]},
    )
    assert asyncio.run(pg_index.PgSections(pool, "v").get_many([])) == {}


def test_the_edge_rules_of_the_keyword_search_and_the_lookup():
    counts = lambda p: [(50,)] if p["term"] == "stadt" else [(1,)]  # noqa: E731 - "stadt" is in 50 % of 100 children
    mixed: Any = Pool(
        [
            ("ts_rank(fts_de, plainto_tsquery", []),
            ("::text as tsq", [("'stadt' & 'garag'",)]),
            ('SELECT COUNT(*) FROM "v" WHERE', counts),
            ('SELECT COUNT(*) FROM "v"', [(100,)]),
        ]
    )
    asyncio.run(pg_index.PgFullText(mixed, "v", "german").search("Stadt Garage", 2))
    assert [p["or_terms"] for t, p in mixed.conn.sent if "%(or_terms)s" in t] == ["'garag'"]
    every: Any = Pool(
        [
            ("::text as tsq", [("'stadt'",)]),
            ('SELECT COUNT(*) FROM "v" WHERE', [(50,)]),
            ('SELECT COUNT(*) FROM "v"', [(100,)]),
        ]
    )
    asyncio.run(pg_index.PgFullText(every, "v", "german").search("Stadt", 2))
    assert [p["or_terms"] for t, p in every.conn.sent if "%(or_terms)s" in t] == ["'stadt'"]  # all frequent: all kept
    empty: Any = Pool([("::text as tsq", [(None,)])])
    assert asyncio.run(pg_index.PgFullText(empty, "v", "german").search("der", 2)) == []
    assert not [t for t, _ in empty.conn.sent if "%(or_terms)s" in t]  # no lexeme: no OR query
    down: Any = Pool([("ts_rank", RuntimeError("down"))])
    try:
        asyncio.run(pg_index.PgFullText(down, "v", "german").search("x", 2))
        raise AssertionError("an outage must not look like no hits")
    except RuntimeError as exc:
        assert str(exc) == "down"
    broken: Any = Pool([("ANY(", RuntimeError("down"))])
    assert asyncio.run(pg_index.PgSections(broken, "v").get_many(["p"])) == {}


class OpenedPool(Pool):
    """read_pool's pool, opened without waiting; close() is recorded."""

    def __init__(self, answers):
        super().__init__(answers)
        self.opened_wait, self.closed = None, False

    async def open(self, wait=True):
        self.opened_wait = wait

    async def close(self):
        self.closed = True


def open_index(monkeypatch, pool):
    made: dict[str, Any] = {}

    def fake_read_pool(conninfo):
        made["conninfo"] = conninfo
        return pool

    monkeypatch.setattr(pg_index, "read_pool", fake_read_pool)
    monkeypatch.setattr(pg_index, "credentials", lambda db: ("rag_app", "pw", "vector_db"))
    profile: Any = SimpleNamespace(retrieval=SimpleNamespace(fts_language="german", fts_extra_stopwords=()))
    database: Any = SimpleNamespace(host="pgvector", port=5432)
    return asyncio.run(pg_index.open_index("example", profile, database)), made


def test_open_index_builds_the_three_reads_on_the_typed_views_after_probing_both(monkeypatch):
    pool = OpenedPool([("SELECT 1", [(1,)])])
    handles, made = open_index(monkeypatch, pool)
    assert pool.opened_wait is False and handles.engines == (pool,) and not pool.closed
    assert all(part in made["conninfo"] for part in ("host=pgvector", "port=5432", "dbname=vector_db", "user=rag_app"))
    assert [t for t, _ in pool.conn.sent] == [
        'SELECT 1 FROM "example_sections" LIMIT 1',
        'SELECT "content", "section_id", "embedding" FROM "example_chunks" LIMIT 0',
    ]
    assert handles.vector.view == handles.full_text.view == "example_chunks"
    asyncio.run(handles.sections.ping())
    assert pool.conn.sent[-1][0] == 'SELECT 1 FROM "example_sections" LIMIT 1'


def test_open_index_fails_and_closes_the_pool_when_the_index_is_missing_or_the_database_is_down(monkeypatch):
    # the opener once failed here (PGVectorStore.create read the view's columns): a start without an index is degraded, /readyz
    # says "missing", a question gets the not_ready frame, and a failed reload leaves the previous graph serving
    for marker in ("LIMIT 1", "LIMIT 0"):  # the section view is missing / the child view is missing
        pool = OpenedPool([(marker, RuntimeError("relation does not exist"))])
        with pytest.raises(RuntimeError, match="does not exist"):
            open_index(monkeypatch, pool)
        assert pool.closed


def test_keyword_terms_drop_the_stop_terms_unless_nothing_is_left():
    assert pg_index.keyword_terms("'haushaltsplan' & '2026' & 'a'") == ["haushaltsplan", "2026", "a"]
    tsq = "'haushaltsplan' & '2026' & 'a' & 'telefon'"
    assert pg_index.keyword_terms(tsq, ("telefon",)) == ["haushaltsplan"]
    assert pg_index.keyword_terms("'2026' & 'a'", ("telefon",)) == ["2026", "a"]  # nothing left: keep all


def test_the_pool_prepares_nothing_and_checks_connections():
    pool = pg_index.read_pool("host=pgvector port=5432 dbname=d user=u password=p")
    assert isinstance(pool, AsyncConnectionPool)
    assert pool.kwargs["prepare_threshold"] is None and pool.max_size == 20 and pool.min_size == 2
    assert pool.max_idle == 240.0 and pool.timeout == 10.0 and pool._check is not None


def test_the_pool_check_sweeps_every_dead_idle_connection_on_the_first_failure(monkeypatch):
    class RecordingPool:
        swept = 0

        def __init__(self, conninfo, **kwargs):
            self.kwargs = kwargs

        async def check_connection(self, conn):
            if conn == "dead":
                raise OSError("connection closed")

        async def check(self):
            self.swept += 1

    monkeypatch.setattr(pg_index, "AsyncConnectionPool", RecordingPool)
    pool: Any = pg_index.read_pool("host=pgvector port=5432 dbname=d user=u password=p")
    asyncio.run(pool.kwargs["check"]("alive"))
    assert pool.swept == 0
    with pytest.raises(OSError):
        asyncio.run(pool.kwargs["check"]("dead"))
    assert pool.swept == 1
