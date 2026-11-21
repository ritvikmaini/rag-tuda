"""The assistant's index reads on psycopg 3: the vector search, the keyword search and the section lookup over the typed
views `<collection>_chunks` and `<collection>_sections` (SQL contract: tests/pgvector/test_contract_sql.py). The vector is
sent as a `str([float(x) ...])` literal with an explicit ::vector. Everything runs through one pool; nothing is
prepared. A child keeps section_id, its retriever and its score; chunk_index,
total_chunks and unique_id are not carried (they show only in the docs frame of the no-parents fallback, which no consumer
reads). The pool (read_pool) lives here, not in adapters.db: the indexer imports db and installs no psycopg-pool.

The keyword search reads fts_de for every question and fts_en besides for an English one (the English column is NULL for German
rows). A named acronym group is one OR group inside the AND tsquery, built from `plainto_tsquery` per term: the question's text
reaches a tsquery only as lexemes the server itself produced."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, LiteralString, cast

from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from rag_core.adapters.db import credentials
from rag_core.domain.acronyms import expand
from rag_core.domain.filters import Filter
from rag_core.domain.language import detect
from rag_core.domain.resilience import check_or_sweep
from rag_core.domain.retrieval import Hit, Section
from rag_core.settings.profile import Acronyms, Profile
from rag_core.settings.sections import DatabaseSection

logger = logging.getLogger(__name__)

VECTOR: LiteralString = (
    'SELECT "content", "section_id", cosine_distance("embedding", %(vector)s::vector) as distance '
    'FROM "public".{}{where} ORDER BY "embedding" <=> %(vector)s::vector LIMIT %(k)s'
)
# {rank} and {match} are built per search from the legs read: one ts_rank (GREATEST of them) and one `column @@ query` each.
# fts_en is NULL for a German row, so its `@@` is NULL there: {seen} is tested with IS NOT TRUE, never negated with NOT.
FTS_AND: LiteralString = (
    "SELECT content, section_id, GREATEST({rank}) as rank FROM {} WHERE ({match}){where} ORDER BY rank DESC LIMIT %(k)s"
)
FTS_OR: LiteralString = (
    "SELECT content, section_id, GREATEST({rank}) as rank FROM {} WHERE ({match}) AND ({seen}) IS NOT TRUE{where} "
    "ORDER BY rank DESC LIMIT %(remaining)s"
)
TSQUERY: LiteralString = "SELECT plainto_tsquery(%(lang)s, %(query)s)::text as tsq"
# one row per acronym group (terms[i] belongs to owners[i]): its terms as an OR of tsqueries; a term of stopwords only is left out
GROUP_QUERIES: LiteralString = (
    "SELECT string_agg(plainto_tsquery(%(lang)s, term)::text, ' | ') as tsq "
    "FROM unnest(%(terms)s::text[], %(owners)s::int[]) AS t(term, owner) "
    "WHERE plainto_tsquery(%(lang)s, term)::text <> '' GROUP BY owner ORDER BY owner"
)
COUNT_ALL: LiteralString = "SELECT COUNT({column}) FROM {}"
COUNT_TERM: LiteralString = "SELECT COUNT(*) FROM {} WHERE {column} @@ %(term)s::tsquery"
SECTIONS: LiteralString = (
    "SELECT section_id, content as text, metadata FROM {} WHERE section_id = ANY(%(section_ids)s){where}"
)
# the sections that hold a paragraph: it is in the parent's recorded `paragraphs` (a merged parent keeps only the prefix of
# its sections' paths) or it heads the section path (a long section's later parts carry no record). {narrow} are the
# programme and PO version conditions; a NULL row passes them like every filter condition
REF_SECTIONS: LiteralString = (
    "SELECT section_id, content as text, metadata FROM {} WHERE regulation = %(ref_regulation)s "
    "AND ((metadata -> 'paragraphs') @> jsonb_build_array(%(ref_paragraph)s::text) "
    "OR (metadata ->> 'section_path') ~ %(ref_path)s){narrow}{where} ORDER BY url, part_index, section_id LIMIT %(ref_limit)s"
)
REF_PROGRAMME: LiteralString = " AND (programme && %(ref_programme)s::text[] OR programme IS NULL)"
REF_PO_VERSION: LiteralString = " AND (po_version && %(ref_po_version)s::int[] OR po_version IS NULL)"
REF_LIMIT = 10  # the sections one reference may bring
PING: LiteralString = "SELECT 1 FROM {} LIMIT 1"
CHILD_PROBE: LiteralString = 'SELECT "content", "section_id", "embedding" FROM {} LIMIT 0'
FREQUENT_SHARE = 0.05  # a term in more than 5 % of the children is left out of the OR fallback
READ_POOL_MAX = 20  # two pools live 120 s per reload
ReadPool = AsyncConnectionPool[AsyncConnection[TupleRow]]  # strict typing needs the pool's type argument


# The hard filter's conditions, over the typed columns of both views. A NULL column passes every filter on it; a row bound
# to a PO version (po_version set) is kept for a date only once its valid_from has come. Values are bound parameters.
ARCHIVED: LiteralString = "archived = false"
DEGREE: LiteralString = "(degree && %(f_degree)s::text[] OR degree IS NULL)"
PROGRAMME: LiteralString = "(programme && %(f_programme)s::text[] OR programme IS NULL)"
PO_VERSION: LiteralString = "(po_version && %(f_po_version)s::int[] OR po_version IS NULL)"
REGULATION: LiteralString = "(regulation = ANY(%(f_regulation)s::text[]) OR regulation IS NULL)"
VALID_FROM: LiteralString = "(valid_from <= %(f_date)s::date OR valid_from IS NULL OR po_version IS NULL)"


def where_of(filter: Filter | None, lead: LiteralString = " AND") -> tuple[LiteralString, dict[str, Any]]:
    """The filter's conditions as one clause starting with `lead` (" AND" after a WHERE, " WHERE" without one) and the
    parameters it binds; None (no filtering) gives no clause. An empty Filter keeps archived rows out."""
    if filter is None:
        return "", {}
    conditions: list[LiteralString] = [] if filter.include_archived else [ARCHIVED]
    params: dict[str, Any] = {}
    wanted: tuple[tuple[LiteralString, str, Any], ...] = (
        (DEGREE, "f_degree", filter.degree),
        (PROGRAMME, "f_programme", filter.programme),
        (PO_VERSION, "f_po_version", filter.po_version),
        (REGULATION, "f_regulation", filter.regulation),
        (VALID_FROM, "f_date", filter.date),
    )
    for condition, name, value in wanted:
        if value is not None:
            conditions.append(condition)
            params[name] = list(value) if isinstance(value, tuple) else value
    if not conditions:
        return "", {}
    return lead + " " + " AND ".join(conditions), params


def _on(template: LiteralString, view: str, where: LiteralString = "", **parts: LiteralString) -> sql.Composed:
    return sql.SQL(template).format(
        sql.Identifier(view), where=sql.SQL(where), **{k: sql.SQL(v) for k, v in parts.items()}
    )


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
        self.pool, self.view = pool, view

    async def search(
        self, vector: Sequence[float], k: int, *, max_distance: float | None = None, filter: Filter | None = None
    ) -> list[Hit]:
        literal = f"{[float(x) for x in vector]}"
        where, bound = where_of(filter, " WHERE")
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(_on(VECTOR, self.view, where), {"vector": literal, "k": k, **bound})
            ).fetchall()
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


def _quoted(lexeme: str) -> str:
    """A lexeme the server produced, as a quoted tsquery operand."""
    return "'" + lexeme.replace("\\", "\\\\").replace("'", "''") + "'"


@dataclass(frozen=True)
class Leg:
    """One full-text column searched for a question: the tsquery texts (server output) of the question's words and of each
    acronym group it names, under the column's text search configuration."""

    column: Literal["fts_de", "fts_en"]
    config: str
    words: str
    groups: tuple[str, ...]

    @property
    def and_text(self) -> str:
        """The words and the groups, all required; a group is satisfied by any of its terms."""
        return " & ".join([*([self.words] if self.words else []), *(f"({g})" for g in self.groups)])


def _rank(legs: Sequence[Leg], key: Literal["and", "or"]) -> LiteralString:
    return ", ".join(f"ts_rank({leg.column}, %({leg.column}_{key})s::tsquery)" for leg in legs)


def _match(legs: Sequence[Leg], key: Literal["and", "or"]) -> LiteralString:
    return " OR ".join(f"{leg.column} @@ %({leg.column}_{key})s::tsquery" for leg in legs)


class PgFullText:
    def __init__(
        self,
        pool: ReadPool,
        view: str,
        language: str,
        extra_stopwords: Sequence[str] = (),
        acronyms: Sequence[Sequence[str]] = (),
    ) -> None:
        """`language` is the text search configuration of fts_de (the profile's); `acronyms` are the groups of terms that
        widen a named term to the whole group, none by default."""
        self.pool, self.view, self.language = pool, view, language
        self.extra_stopwords = tuple(extra_stopwords or ())
        self.acronyms = tuple(tuple(group) for group in acronyms)
        # per instance: a reload after a publish or a rollback starts with fresh term statistics, per column
        self.term_is_frequent: dict[tuple[str, str], bool] = {}
        self.total_rows: dict[str, int] = {}

    async def _frequent(self, conn: AsyncConnection[TupleRow], leg: Leg, term: str) -> bool:
        key = (leg.column, term)
        if key not in self.term_is_frequent:
            if leg.column not in self.total_rows:
                row = await (await conn.execute(_on(COUNT_ALL, self.view, column=leg.column))).fetchone()
                self.total_rows[leg.column] = int(row[0]) if row else 0
            row = await (
                await conn.execute(_on(COUNT_TERM, self.view, column=leg.column), {"term": _quoted(term)})
            ).fetchone()
            df = int(row[0]) if row else 0
            self.term_is_frequent[key] = df > self.total_rows[leg.column] * FREQUENT_SHARE
        return self.term_is_frequent[key]

    async def _legs(
        self, conn: AsyncConnection[TupleRow], words: str, named: Sequence[Sequence[str]], english: bool
    ) -> list[Leg]:
        """The columns this question searches, with their tsqueries; a column whose query is empty (only stopwords) is left out."""
        configs: list[tuple[Literal["fts_de", "fts_en"], str]] = [("fts_de", self.language)]
        if english:
            configs.append(("fts_en", "english"))
        terms = [term for group in named for term in group]
        owners = [i for i, group in enumerate(named) for _ in group]
        legs: list[Leg] = []
        for column, config in configs:
            row = await (await conn.execute(sql.SQL(TSQUERY), {"lang": config, "query": words})).fetchone()
            groups: list[str] = []
            if named:
                params = {"lang": config, "terms": terms, "owners": owners}
                groups = [r[0] for r in await (await conn.execute(sql.SQL(GROUP_QUERIES), params)).fetchall()]
            leg = Leg(column, config, (row[0] if row else None) or "", tuple(groups))
            if leg.and_text:
                legs.append(leg)
        return legs

    async def _or_text(self, conn: AsyncConnection[TupleRow], leg: Leg) -> str | None:
        """The OR fallback's tsquery for one column: the question's words without the frequent ones (all of them when every
        one is frequent), and every acronym group whole, whatever its terms' share."""
        terms = keyword_terms(leg.words, self.extra_stopwords) if leg.words else []
        kept = [t for t in terms if not await self._frequent(conn, leg, t)] or terms
        if len(kept) < len(terms):  # a count, never the words: a word of the question is question text
            logger.info("[FTS] Filtered %d high-freq terms", len(terms) - len(kept))
        return " | ".join([*(_quoted(t) for t in kept), *(f"({g})" for g in leg.groups)]) or None

    async def search(self, query: str, k: int, filter: Filter | None = None) -> list[Hit]:
        where, bound = where_of(
            filter
        )  # both statements carry it: the OR fallback never brings back a filtered-out row
        words, named = expand(query, self.acronyms)
        try:
            async with self.pool.connection() as conn:  # one transaction for all statements
                legs = await self._legs(conn, words, named, english=detect(query) == "en")
                if not legs:
                    return []
                params = {"k": k, **bound, **{f"{leg.column}_and": leg.and_text for leg in legs}}
                statement = _on(FTS_AND, self.view, where, rank=_rank(legs, "and"), match=_match(legs, "and"))
                # binary rows: ts_rank is real; the text format would round fts_rank
                rows = list(await (await conn.execute(statement, params, binary=True)).fetchall())
                if len(rows) < k:
                    seen = {r[0] for r in rows}
                    alternatives = [(leg, await self._or_text(conn, leg)) for leg in legs]
                    or_legs = [leg for leg, text in alternatives if text]
                    if or_legs:
                        more_params = {
                            **params,
                            "remaining": k - len(rows),
                            **{f"{leg.column}_or": text for leg, text in alternatives if text},
                        }
                        more_statement = _on(
                            FTS_OR,
                            self.view,
                            where,
                            rank=_rank(or_legs, "or"),
                            match=_match(or_legs, "or"),
                            seen=_match(legs, "and"),
                        )
                        more = await (await conn.execute(more_statement, more_params, binary=True)).fetchall()
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
        self.pool, self.view, self.ping_query = pool, view, _on(PING, view)

    async def get_many(self, section_ids: Sequence[str], filter: Filter | None = None) -> dict[str, Section]:
        if not section_ids:
            return {}
        where, bound = where_of(filter)
        try:
            async with self.pool.connection() as conn:
                params = {"section_ids": list(section_ids), **bound}
                rows = await (await conn.execute(_on(SECTIONS, self.view, where), params)).fetchall()
        except Exception as exc:
            logger.error("[PostgresParentLookup] Error in batch lookup: %s", exc)
            return {}  # a failed lookup gives no sections, the answer goes on without them
        return {
            section_id: Section(section_id=section_id, text=text, metadata=_meta(meta))
            for section_id, text, meta in rows
        }

    async def sections_by_ref(
        self,
        regulation: str,
        paragraph: str,
        programme: Sequence[str] | None,
        po_version: Sequence[int] | None,
        filter: Filter | None = None,
    ) -> list[Section]:
        """The sections of the regulation that hold the paragraph ("§ 22"), under the question's filter; a programme or PO
        version given narrows them. A failed lookup raises: the caller decides what an unresolved reference costs."""
        number = re.escape(paragraph.removeprefix("§").strip())
        where, bound = where_of(filter)
        params = {
            "ref_regulation": regulation,
            "ref_paragraph": paragraph,
            # the number is a whole word: "§ 22" is not "§ 220" or "§ 22a"
            "ref_path": rf"(^| > )§[[:space:]]*{number}([^[:alnum:]]|$)",
            "ref_limit": REF_LIMIT,
            **bound,
        }
        narrow: LiteralString = ""
        if programme:
            params["ref_programme"] = list(programme)
            narrow += REF_PROGRAMME
        if po_version:
            params["ref_po_version"] = list(po_version)
            narrow += REF_PO_VERSION
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(_on(REF_SECTIONS, self.view, where, narrow=narrow), params)).fetchall()
        return [Section(section_id=section_id, text=text, metadata=_meta(meta)) for section_id, text, meta in rows]

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
        min_size=min(2, max_size),
        max_size=max_size,
        open=False,
        timeout=10.0,
        max_idle=240.0,
        kwargs={"prepare_threshold": None},
        check=check,
        name=name,
    )
    return pool


async def open_index(
    collection: str, profile: Profile, database: DatabaseSection, acronyms: Acronyms | None = None
) -> IndexHandles:
    """The three reads over the collection's typed views <collection>_chunks and <collection>_sections, on one new pool.
    It fails when the index is missing or the database is down (a probe of both views): a start without an index is degraded, /readyz says "missing", a question
    gets the not_ready frame, and a failed reload leaves the old graph serving. The pool is closed before the error goes on;
    later outages are the pool's own reconnecting. The keyword search widens the groups of `acronyms` unless the profile's
    acronyms switch is off."""
    retrieval = profile.retrieval
    user, password, name = credentials(database)
    host, port = database.host, database.port
    logger.info(f"PostgreSQL Connection: {user}@{host}:{port}/{name}")
    groups = tuple(g.terms for g in acronyms.groups) if acronyms and retrieval.switches.acronyms else ()
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
        PgFullText(pool, chunks, retrieval.fts_language, tuple(retrieval.fts_extra_stopwords), groups),
        sections,
        (pool,),
    )
