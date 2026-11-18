"""The assistant's index reads on psycopg 3: the vector search, the keyword search and the section lookup over the typed
views `<collection>_chunks` and `<collection>_sections` (SQL contract: tests/pgvector/test_contract_sql.py). The vector is
sent as a `str([float(x) ...])` literal with an explicit ::vector. Everything runs through one pool; nothing is
prepared. A child keeps section_id, its retriever and its score; chunk_index,
total_chunks and unique_id are not carried (they show only in the docs frame of the no-parents fallback, which no consumer
reads). The pool (read_pool) lives here, not in adapters.db: the indexer imports db and installs no psycopg-pool."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, LiteralString, cast

from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from rag_core.adapters.db import credentials
from rag_core.domain.resilience import check_or_sweep
from rag_core.domain.retrieval import Hit, Section
from rag_core.settings.profile import Profile
from rag_core.settings.sections import DatabaseSection

logger = logging.getLogger(__name__)

VECTOR: LiteralString = (
    'SELECT "content", "section_id", cosine_distance("embedding", %(vector)s::vector) as distance '
    'FROM "public".{} ORDER BY "embedding" <=> %(vector)s::vector LIMIT %(k)s'
)
FTS_AND: LiteralString = (
    "SELECT content, section_id, ts_rank(fts_de, plainto_tsquery(%(lang)s, %(query)s)) as rank "
    "FROM {} WHERE fts_de @@ plainto_tsquery(%(lang)s, %(query)s) ORDER BY rank DESC LIMIT %(k)s"
)
FTS_OR: LiteralString = (
    "SELECT content, section_id, ts_rank(fts_de, to_tsquery(%(lang)s, %(or_terms)s)) as rank "
    "FROM {} WHERE fts_de @@ to_tsquery(%(lang)s, %(or_terms)s) AND NOT fts_de @@ plainto_tsquery(%(lang)s, "
    "%(query)s) ORDER BY rank DESC LIMIT %(remaining)s"
)
TSQUERY: LiteralString = "SELECT plainto_tsquery(%(lang)s, %(query)s)::text as tsq"
COUNT_ALL: LiteralString = "SELECT COUNT(*) FROM {}"
COUNT_TERM: LiteralString = "SELECT COUNT(*) FROM {} WHERE fts_de @@ to_tsquery(%(lang)s, %(term)s)"
SECTIONS: LiteralString = "SELECT section_id, content as text, metadata FROM {} WHERE section_id = ANY(%(section_ids)s)"
PING: LiteralString = "SELECT 1 FROM {} LIMIT 1"
CHILD_PROBE: LiteralString = 'SELECT "content", "section_id", "embedding" FROM {} LIMIT 0'
FREQUENT_SHARE = 0.05  # a term in more than 5 % of the children is left out of the OR fallback
READ_POOL_MAX = 20  # two pools live 120 s per reload
ReadPool = AsyncConnectionPool[AsyncConnection[TupleRow]]  # strict typing needs the pool's type argument


def _on(template: LiteralString, view: str) -> sql.Composed:
    return sql.SQL(template).format(sql.Identifier(view))


def _meta(value: Any) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def keyword_terms(tsq_text: str, extra_stopwords: Sequence[str] = ()) -> list[str]:
    """Lexemes of an AND tsquery ("'unterlag' & 'brauch'" -> ['unterlag', 'brauch']) for the OR
    fallback. Only when extra stopwords are set are single characters, pure numbers
    and the stopwords dropped (all terms kept if nothing would remain); otherwise the terms
    stay as they are, so years such as 2026 still count."""
    terms = [t.strip().strip("'") for t in tsq_text.split(" & ")]
    terms = [t for t in terms if t]
    if extra_stopwords:
        terms = [t for t in terms if len(t) >= 2 and not t.isdigit() and t not in extra_stopwords] or terms
    return terms


class PgVectorStore:
    def __init__(self, pool: ReadPool, view: str) -> None:
        self.pool, self.query = pool, _on(VECTOR, view)

    async def search(self, vector: Sequence[float], k: int, *, max_distance: float | None = None) -> list[Hit]:
        literal = f"{[float(x) for x in vector]}"
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(self.query, {"vector": literal, "k": k})).fetchall()
        if rows:
            logger.info(
                "[Semantic] k=%d distance min=%.3f max=%.3f", k, min(r[2] for r in rows), max(r[2] for r in rows)
            )
        return [
            Hit(
                page_content=content,
                metadata={
                    "section_id": section_id,
                    "retriever": "semantic",
                    "child_chunk_semantic_search_score": distance,
                },
            )
            for content, section_id, distance in rows
            if max_distance is None or distance <= max_distance
        ]


class PgFullText:
    def __init__(self, pool: ReadPool, view: str, language: str, extra_stopwords: Sequence[str] = ()) -> None:
        self.pool, self.view, self.language = pool, view, language
        self.extra_stopwords = tuple(extra_stopwords or ())
        self.and_query, self.or_query = _on(FTS_AND, view), _on(FTS_OR, view)
        # per instance: a reload after a publish or a rollback starts with fresh term statistics
        self.term_is_frequent: dict[str, bool] = {}
        self.total_rows: int | None = None

    async def _frequent(self, conn: AsyncConnection[TupleRow], term: str) -> bool:
        if term not in self.term_is_frequent:
            if self.total_rows is None:
                row = await (await conn.execute(_on(COUNT_ALL, self.view))).fetchone()
                self.total_rows = int(row[0]) if row else 0
            row = await (
                await conn.execute(_on(COUNT_TERM, self.view), {"lang": self.language, "term": term})
            ).fetchone()
            df = int(row[0]) if row else 0
            self.term_is_frequent[term] = df > self.total_rows * FREQUENT_SHARE
        return self.term_is_frequent[term]

    async def _or_terms(self, conn: AsyncConnection[TupleRow], query: str) -> str | None:
        row = await (await conn.execute(sql.SQL(TSQUERY), {"lang": self.language, "query": query})).fetchone()
        terms = keyword_terms(row[0], self.extra_stopwords) if row and row[0] else []
        if not terms:
            return None
        kept = [t for t in terms if not await self._frequent(conn, t)] or terms
        if len(kept) < len(terms):  # a count, never the words: a word of the question is question text
            logger.info("[FTS] Filtered %d high-freq terms", len(terms) - len(kept))
        return " | ".join("'" + t.replace("'", "''") + "'" for t in kept)

    async def search(self, query: str, k: int) -> list[Hit]:
        try:
            async with self.pool.connection() as conn:  # one transaction for all statements
                params = {"lang": self.language, "query": query, "k": k}
                # binary rows: ts_rank is real; the text format would round fts_rank
                rows = list(await (await conn.execute(self.and_query, params, binary=True)).fetchall())
                if len(rows) < k:
                    seen = {r[0] for r in rows}
                    or_terms = await self._or_terms(conn, query)
                    if or_terms:
                        more_params = {
                            "lang": self.language,
                            "query": query,
                            "or_terms": or_terms,
                            "remaining": k - len(rows),
                        }
                        more = await (await conn.execute(self.or_query, more_params, binary=True)).fetchall()
                        rows += [r for r in more if r[0] not in seen]
        except Exception as exc:
            logger.error("[PostgresFTSRetriever] Error during search: %s", type(exc).__name__)
            raise  # a database outage must not look like "no keyword hits"
        return [
            Hit(
                page_content=content,
                metadata={"section_id": section_id, "retriever": "fts", "fts_rank": float(rank)},
            )
            for content, section_id, rank in rows
        ]


class PgSections:
    def __init__(self, pool: ReadPool, view: str) -> None:
        self.pool, self.query, self.ping_query = pool, _on(SECTIONS, view), _on(PING, view)

    async def get_many(self, section_ids: Sequence[str]) -> dict[str, Section]:
        if not section_ids:
            return {}
        try:
            async with self.pool.connection() as conn:
                rows = await (await conn.execute(self.query, {"section_ids": list(section_ids)})).fetchall()
        except Exception as exc:
            logger.error("[PostgresParentLookup] Error in batch lookup: %s", exc)
            return {}  # a failed lookup gives no sections, the answer goes on without them
        return {
            section_id: Section(section_id=section_id, text=text, metadata=_meta(meta))
            for section_id, text, meta in rows
        }

    async def ping(self) -> None:
        """Readiness: the section view answers."""
        async with self.pool.connection() as conn:
            await conn.execute(self.ping_query)


@dataclass
class IndexHandles:
    vector: PgVectorStore
    full_text: PgFullText
    sections: PgSections
    # (the pool,): closed a request deadline (120 s) after the next reload's swap, or at shutdown (dispose_engines)
    engines: tuple[Any, ...]


def read_pool(conninfo: str, *, max_size: int = READ_POOL_MAX, name: str = "index-reads") -> ReadPool:
    """The assistant's index reads (the turn store opens one more, named turns): one pool without prepared
    statements; the minimum two connections stay open, a restarted database kills them, and the
    pool's check sweeps the dead ones on the next use (one reconnect); a handed-out connection is checked, and a failed check
    sweeps every dead idle one. Not opened: open_index opens it without waiting and probes the views."""

    async def check(conn: AsyncConnection[TupleRow]) -> None:
        await check_or_sweep(pool, conn)

    pool: ReadPool = AsyncConnectionPool(
        conninfo,
        min_size=2,
        max_size=max_size,
        open=False,
        timeout=10.0,
        max_idle=240.0,
        kwargs={"prepare_threshold": None},
        check=check,
        name=name,
    )
    return pool


async def open_index(collection: str, profile: Profile, database: DatabaseSection) -> IndexHandles:
    """The three reads over the collection's typed views <collection>_chunks and <collection>_sections, on one new pool.
    It fails when the index is missing or the database is down (a probe of both views): a start without an index is degraded, /readyz says "missing", a question
    gets the not_ready frame, and a failed reload leaves the old graph serving. The pool is closed before the error goes on;
    later outages are the pool's own reconnecting."""
    retrieval = profile.retrieval
    user, password, name = credentials(database)
    host, port = database.host, database.port
    logger.info(f"PostgreSQL Connection: {user}@{host}:{port}/{name}")
    pool = read_pool(make_conninfo(host=host, port=port, dbname=name, user=user, password=password))
    await pool.open(wait=False)
    chunks, sections_view = f"{collection}_chunks", f"{collection}_sections"
    sections = PgSections(pool, sections_view)
    try:
        await sections.ping()
        async with pool.connection() as conn:
            await conn.execute(_on(CHILD_PROBE, chunks))
    except BaseException:
        await pool.close()
        raise
    logger.info(f"Using PostgreSQL FTS for keyword search (language: {retrieval.fts_language})")
    logger.info(f"Created PostgresParentLookup for table: {sections_view}")
    return IndexHandles(
        PgVectorStore(pool, chunks),
        PgFullText(pool, chunks, retrieval.fts_language, tuple(retrieval.fts_extra_stopwords)),
        sections,
        (pool,),
    )
