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


# markers of the keyword statements in a fake pool's answers
TSQ = "plainto_tsquery(%(lang)s, %(query)s)::text as tsq"  # the lexemes of the question's words
AND = "ts_rank(fts_de, %(fts_de_and)s"
OR = "ts_rank(fts_de, %(fts_de_or)s"
COUNT_TERM = 'SELECT COUNT(*) FROM "v" WHERE fts_de @@'
COUNT_ALL = 'SELECT COUNT(fts_de) FROM "v"'


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
    assert "WHERE ({match}){where} ORDER BY rank DESC LIMIT %(k)s" in pg_index.FTS_AND
    assert "section_id = ANY(%(section_ids)s){where}" in pg_index.SECTIONS


def test_a_filter_binds_its_values_into_both_legs_and_the_section_lookup():
    flt = Filter(po_version=(2023,), regulation=("mhb",), date=dt.date(2026, 10, 1))
    where = (
        " AND (archived = false OR po_version && %(f_po_version)s::int[])"
        " AND (po_version && %(f_po_version)s::int[] OR po_version IS NULL)"
        " AND (regulation = ANY(%(f_regulation)s::text[]) OR regulation IS NULL)"
        " AND (valid_from <= %(f_date)s::date OR valid_from IS NULL OR po_version IS NULL)"
    )
    bound = {"f_po_version": [2023], "f_regulation": ["mhb"], "f_date": dt.date(2026, 10, 1)}
    pool: Any = Pool([("cosine_distance", [])])
    asyncio.run(pg_index.PgVectorStore(pool, "v").search([0.5], 3, filter=flt))
    text, params = pool.conn.sent[0]
    assert text.endswith(f'FROM "public"."v" WHERE{where[4:]} ORDER BY "embedding" <=> %(vector)s::vector LIMIT %(k)s')
    assert params == {"vector": "[0.5]", "k": 3, **bound}
    pool = Pool([(TSQ, [("'garag'",)]), (COUNT_TERM, [(0,)])])
    asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Garage", 2, flt))
    and_sql, and_params = next((t, p) for t, p in pool.conn.sent if AND in t)
    or_sql, or_params = next((t, p) for t, p in pool.conn.sent if OR in t)
    assert f"::tsquery){where} ORDER BY rank" in and_sql  # the OR fallback carries it too
    assert f"IS NOT TRUE{where} ORDER BY rank" in or_sql
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
    assert pg_index.where_of(Filter(po_version=(2015,), include_archived=True))[0].startswith(
        " AND (po_version &&"
    )  # a history request keeps archived rows without a PO version
    assert pg_index.where_of(Filter(degree=("master",)), " WHERE") == (
        " WHERE archived = false AND (degree && %(f_degree)s::text[] OR degree IS NULL)",
        {"f_degree": ["master"]},
    )


def test_max_distance_cuts_far_hits():
    pool: Any = Pool([("cosine_distance", [("near", "a", 0.2), ("far", "b", 0.9)])])
    hits = asyncio.run(pg_index.PgVectorStore(pool, "v").search([0.0], 2, max_distance=0.5))
    assert [h.page_content for h in hits] == ["near"]


def test_an_empty_vector_result_and_a_cached_term():  # the two branches the others miss
    empty: Any = Pool([])
    assert asyncio.run(pg_index.PgVectorStore(empty, "v").search([0.0], 3)) == []
    pool: Any = Pool(
        [
            (TSQ, [("'garag'",)]),
            (COUNT_TERM, [(1,)]),
            (COUNT_ALL, [(100,)]),
        ]
    )
    fts = pg_index.PgFullText(pool, "v", "german")
    for _ in range(2):
        asyncio.run(fts.search("Garage", 2))
    assert len([t for t, _ in pool.conn.sent if t.startswith(COUNT_TERM)]) == 1  # the second: from the cache


def test_the_or_fallback_supplements_and_skips_seen_texts():
    pool: Any = Pool(
        [
            (AND, [("eins", "a", 0.5)]),
            (TSQ, [("'garag' & 'stellplatz'",)]),
            (COUNT_TERM, [(1,)]),
            (COUNT_ALL, [(100,)]),
            (OR, [("eins", "a", 0.4), ("zwei", "b", 0.3)]),
        ]
    )
    hits = asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Garage Stellplatz", 3))
    assert [
        (h.page_content, h.metadata["section_id"], h.metadata["retriever"], h.metadata["fts_rank"]) for h in hits
    ] == [
        ("eins", "a", "fts", 0.5),
        ("zwei", "b", "fts", 0.3),
    ]
    or_params = [p for t, p in pool.conn.sent if OR in t][0]
    assert or_params == {
        "k": 3,
        "fts_de_and": "'garag' & 'stellplatz'",
        "fts_de_or": "'garag' | 'stellplatz'",
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
            (AND, []),
            (TSQ, [("'garag' & 'stellplatz'",)]),
            (COUNT_TERM, [(1,)]),
            (COUNT_ALL, [(100,)]),
            (OR, [("eins", "a", 0.4)]),
        ]
    )
    assert asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Garage Stellplatz", 3))
    assert caplog.records == []


def test_a_filtered_frequent_term_is_logged_as_a_count_never_as_the_word(
    caplog,
):  # no question text in any log
    caplog.set_level(logging.INFO, logger="rag_core.adapters.pg_index")
    counts = lambda p: [(50,)] if p["term"] in ("'campus'", "'haus'") else [(1,)]  # noqa: E731
    pool: Any = Pool(
        [
            (AND, []),
            (TSQ, [("'campus' & 'haus' & 'garag'",)]),
            (COUNT_TERM, counts),
            (COUNT_ALL, [(100,)]),
            (OR, []),
        ]
    )
    asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Campus Haus Garage", 3))
    assert [r.getMessage() for r in caplog.records] == ["[FTS] Filtered 2 high-freq terms"]


def test_a_failed_search_logs_the_class_never_the_message(caplog):  # a tsquery error quotes the terms of the question
    down: Any = Pool([(TSQ, RuntimeError('syntax error in tsquery: "garage"'))])
    with pytest.raises(RuntimeError):
        asyncio.run(pg_index.PgFullText(down, "v", "german").search("Garage", 2))
    assert [r.getMessage() for r in caplog.records] == ["[PostgresFTSRetriever] Error during search: RuntimeError"]


def test_an_and_query_that_fills_k_needs_no_or_fallback():
    rows = [("eins", "a", 0.5), ("zwei", "b", 0.4)]
    pool: Any = Pool([(TSQ, [("'garag'",)]), (AND, rows)])
    hits = asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Garage", 2))
    assert [h.page_content for h in hits] == ["eins", "zwei"]
    assert [AND in t for t, _ in pool.conn.sent] == [
        False,
        True,
    ]  # the words' tsquery, the AND query; no counts, no OR query


def test_the_term_cache_is_per_instance():
    pool: Any = Pool([])  # Any for basedpyright
    a, b = pg_index.PgFullText(pool, "v", "german"), pg_index.PgFullText(pool, "v", "german")
    a.term_is_frequent[("fts_de", "x")] = True
    assert b.term_is_frequent == {} and a.total_rows == {}


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
    counts = lambda p: [(50,)] if p["term"] == "'campus'" else [(1,)]  # noqa: E731 - "campus" is in 50 % of 100 children
    mixed: Any = Pool(
        [
            (AND, []),
            (TSQ, [("'campus' & 'garag'",)]),
            (COUNT_TERM, counts),
            (COUNT_ALL, [(100,)]),
        ]
    )
    asyncio.run(pg_index.PgFullText(mixed, "v", "german").search("Campus Garage", 2))
    assert [p["fts_de_or"] for t, p in mixed.conn.sent if OR in t] == ["'garag'"]
    every: Any = Pool(
        [
            (TSQ, [("'campus'",)]),
            (COUNT_TERM, [(50,)]),
            (COUNT_ALL, [(100,)]),
        ]
    )
    asyncio.run(pg_index.PgFullText(every, "v", "german").search("Campus", 2))
    assert [p["fts_de_or"] for t, p in every.conn.sent if OR in t] == ["'campus'"]  # all frequent: all kept
    empty: Any = Pool([(TSQ, [(None,)])])
    assert asyncio.run(pg_index.PgFullText(empty, "v", "german").search("der", 2)) == []
    assert len(empty.conn.sent) == 1  # no lexeme: no keyword query at all
    down: Any = Pool([(TSQ, RuntimeError("down"))])
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


def open_index(monkeypatch, pool, acronyms=None, switch=True):
    made: dict[str, Any] = {}

    def fake_read_pool(conninfo):
        made["conninfo"] = conninfo
        return pool

    monkeypatch.setattr(pg_index, "read_pool", fake_read_pool)
    monkeypatch.setattr(pg_index, "credentials", lambda db: ("rag_app", "pw", "vector_db"))
    retrieval = SimpleNamespace(
        fts_language="german", fts_extra_stopwords=(), switches=SimpleNamespace(acronyms=switch)
    )
    profile: Any = SimpleNamespace(retrieval=retrieval)
    database: Any = SimpleNamespace(host="pgvector", port=5432)
    return asyncio.run(pg_index.open_index("example", profile, database, acronyms)), made


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


def test_a_pool_smaller_than_two_is_built_with_its_own_minimum():
    pool = pg_index.read_pool("host=pgvector port=5432 dbname=d user=u password=p", max_size=1)
    assert (pool.min_size, pool.max_size) == (1, 1)


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


GROUPS = (("PO", "Prüfungsordnung"), ("Rücktritt", "Rücktritt von der Prüfung"))
GROUP_ANSWER = "string_agg(plainto_tsquery"
EN_AND = "ts_rank(fts_en, %(fts_en_and)s"


def test_open_index_hands_the_acronym_groups_to_the_keyword_search_unless_the_switch_is_off(monkeypatch):
    acronyms: Any = SimpleNamespace(groups=[SimpleNamespace(terms=("PO", "Prüfungsordnung"))])
    on, _ = open_index(monkeypatch, OpenedPool([("SELECT 1", [(1,)])]), acronyms)
    assert on.full_text.acronyms == (("PO", "Prüfungsordnung"),)
    off, _ = open_index(monkeypatch, OpenedPool([("SELECT 1", [(1,)])]), acronyms, switch=False)
    assert off.full_text.acronyms == ()
    none, _ = open_index(monkeypatch, OpenedPool([("SELECT 1", [(1,)])]))
    assert none.full_text.acronyms == ()


def test_a_named_group_is_one_or_group_in_the_and_query_and_the_words_left_over_stay_required():
    pool: Any = Pool(
        [
            (GROUP_ANSWER, [("'po' | 'prufungsordn'",)]),
            (TSQ, [("'anmeld'",)]),
            (AND, [("eins", "a", 0.5)]),
        ]
    )
    asyncio.run(pg_index.PgFullText(pool, "v", "german", acronyms=GROUPS).search("Anmeldung PO", 1))
    words, group, and_query = pool.conn.sent
    assert words[1]["query"] == "Anmeldung"  # the group's term is cut out of the words
    assert group[1] == {"lang": "german", "terms": ["PO", "Prüfungsordnung"], "owners": [0, 0]}
    assert and_query[1]["fts_de_and"] == "'anmeld' & ('po' | 'prufungsordn')"
    plain: Any = Pool([(TSQ, [("'anmeld' & 'po'",)]), (AND, [("eins", "a", 0.5)])])
    asyncio.run(pg_index.PgFullText(plain, "v", "german").search("Anmeldung PO", 1))
    assert not [t for t, _ in plain.conn.sent if GROUP_ANSWER in t]  # no groups configured: the words as asked


def test_a_group_stays_whole_in_the_or_fallback_while_a_frequent_word_goes():
    pool: Any = Pool(
        [
            (GROUP_ANSWER, [("'po' | 'prufungsordn'",)]),
            (TSQ, [("'antrag' & 'zeugnis'",)]),
            (AND, []),
            (COUNT_TERM, lambda p: [(60,)] if p["term"] == "'antrag'" else [(1,)]),
            (COUNT_ALL, [(100,)]),
            (OR, []),
        ]
    )
    asyncio.run(pg_index.PgFullText(pool, "v", "german", acronyms=GROUPS).search("PO Antrag Zeugnis", 3))
    counted = [p["term"] for t, p in pool.conn.sent if t.startswith(COUNT_TERM)]
    assert counted == ["'antrag'", "'zeugnis'"]  # the group's terms are not counted: nothing to drop
    or_params = next(p for t, p in pool.conn.sent if OR in t)
    assert or_params["fts_de_or"] == "'zeugnis' | ('po' | 'prufungsordn')"
    assert or_params["fts_de_and"] == "'antrag' & 'zeugnis' & ('po' | 'prufungsordn')"


def test_an_english_question_searches_both_columns_and_a_german_one_only_the_german_column():
    english: Any = Pool([(TSQ, [("'withdraw'",)]), (EN_AND, [("eins", "a", 0.5)])])
    asyncio.run(pg_index.PgFullText(english, "v", "german").search("Can I do withdrawing?", 1))
    statements = [(t, p) for t, p in english.conn.sent]
    assert [p["lang"] for t, p in statements if TSQ in t] == [
        "german",
        "english",
    ]  # each column under its own configuration
    text, params = next((t, p) for t, p in statements if EN_AND in t)
    assert "fts_de @@ %(fts_de_and)s::tsquery OR fts_en @@ %(fts_en_and)s::tsquery" in text
    assert "GREATEST(ts_rank(fts_de, %(fts_de_and)s::tsquery), ts_rank(fts_en, %(fts_en_and)s::tsquery))" in text
    assert params["fts_de_and"] == params["fts_en_and"] == "'withdraw'"
    german: Any = Pool([(TSQ, [("'withdraw'",)]), (AND, [("eins", "a", 0.5)])])
    asyncio.run(pg_index.PgFullText(german, "v", "german").search("Was ist withdrawing?", 1))
    assert not [t for t, _ in german.conn.sent if "fts_en" in t]


def test_the_or_fallback_tests_the_seen_rows_with_is_not_true_because_fts_en_is_null_for_german_rows():
    pool: Any = Pool([(TSQ, [("'a' & 'b'",)]), (EN_AND, []), (COUNT_TERM, [(0,)]), (COUNT_ALL, [(100,)])])
    asyncio.run(pg_index.PgFullText(pool, "v", "german").search("Can I do this?", 2))
    text = next(t for t, _ in pool.conn.sent if "%(remaining)s" in t)
    assert (
        "AND (fts_de @@ %(fts_de_and)s::tsquery OR fts_en @@ %(fts_en_and)s::tsquery) IS NOT TRUE" in text
    )  # NOT (NULL) would drop every German row
    assert "NOT (" not in text.replace("IS NOT TRUE", "")
    counted = sorted({t.split("COUNT(")[1].split(")")[0] for t, _ in pool.conn.sent if t.startswith("SELECT COUNT(")})
    assert counted == ["*", "fts_de", "fts_en"]  # term statistics per column
    assert pool.conn.sent[-1][1]["fts_en_or"] == "'a' | 'b'"
