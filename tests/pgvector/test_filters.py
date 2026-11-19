"""The hard filter on both search legs and the section lookup, on a fresh cluster as rag_app. Five documents: the PO 2015
(archived), the PO 2023, the module handbook of PO 2023, an info page that applies to every programme, and an archived
document with every typed column NULL. RUN_DOCKER_TESTS=1."""

import asyncio
import datetime as dt
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from psycopg.conninfo import make_conninfo

from indexer.index import ddl
from indexer.index import generations as gen
from rag_core.adapters import pg_index
from rag_core.domain.filters import Filter
from rag_core.services.retrieval import RetrievalService
from rag_core.settings import load_profile
from rag_core.settings.profile import Profile
from rag_core.settings.sections import Timeouts

PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "example" / "profile.yaml"
TARGET = gen.Target.for_collection("rag")
DIMS = 768
QUESTION = "Anerkennung Studienleistung"
# section_id, regulation, degree, programme, po_version, archived, valid_from
DOCUMENTS = (
    ("s15", "po", ["bachelor"], ["bsc-informatik"], [2015], True, "2015-10-01"),
    ("s23", "po", ["bachelor"], ["bsc-informatik"], [2023], False, "2023-10-01"),
    ("smhb", "mhb", ["bachelor"], ["bsc-informatik"], [2023], False, "2023-10-01"),
    ("sinfo", "info", None, None, None, False, "2030-01-01"),  # no degree, programme or PO: not bound to a version
    ("sold", None, None, None, None, True, None),
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="image tests run only with RUN_DOCKER_TESTS=1 and a running Docker",
)


def vec(i):
    v = [0.1] * DIMS
    v[i] = 1.0
    return "[" + ",".join(str(x) for x in v) + "]"


@pytest.fixture(scope="module")
def conninfo(port):
    import psycopg

    conn = psycopg.connect(
        host="127.0.0.1", port=port, dbname="vector_db", user="rag_indexer", password="i" * 64, autocommit=True
    )
    n, _ = gen.begin(conn, TARGET, {"config_hash": "x"})
    names = TARGET.tables(n)
    for statement in ddl.create_tables(names["parents"], names["children"], DIMS):
        conn.execute(statement)
    columns = "regulation, degree, programme, po_version, archived, valid_from, lang"
    for i, (section_id, *typed) in enumerate(DOCUMENTS):
        conn.execute(
            f'INSERT INTO "{names["parents"]}" (section_id, document_id, collection, source, audience, part_index, content,'
            f" metadata, {columns}) VALUES (%s, %s, 'rag', 'example.web', 'public', 0, %s, '{{}}'::jsonb, %s, %s, %s, %s,"
            " %s, %s, 'de')",
            (section_id, section_id, f"{QUESTION} im Abschnitt {section_id}", *typed),
        )
        conn.execute(
            f'INSERT INTO "{names["children"]}" (chunk_id, section_id, document_id, collection, source, audience, content,'
            f" embedding, {columns}) VALUES (%s, %s, %s, 'rag', 'example.web', 'public', %s, %s::vector, %s, %s, %s, %s,"
            " %s, %s, 'de')",
            (uuid.UUID(int=i).hex, section_id, section_id, f"{QUESTION} im Abschnitt {section_id}", vec(i), *typed),
        )
    for statement in ddl.create_indexes(names["parents"], names["children"]):
        conn.execute(statement)
    gen.verify(conn, TARGET, n, {"children": len(DOCUMENTS), "parents": len(DOCUMENTS)})
    gen.publish(conn, TARGET, n)
    conn.close()
    return make_conninfo(host="127.0.0.1", port=port, dbname="vector_db", user="rag_app", password="s" * 64)


@asynccontextmanager
async def legs(conninfo):
    pool = pg_index.read_pool(conninfo, max_size=4)
    await pool.open(wait=True)
    try:
        yield (
            pg_index.PgVectorStore(pool, "rag_chunks"),
            pg_index.PgFullText(pool, "rag_chunks", "german"),
            pg_index.PgSections(pool, "rag_sections"),
        )
    finally:
        await pool.close()


def found(conninfo, filter):
    """The section ids each leg and the section lookup return under the filter."""

    async def run():
        async with legs(conninfo) as (vector, full_text, sections):
            semantic = await vector.search([float(x) for x in vec(0)[1:-1].split(",")], 10, filter=filter)
            lexical = await full_text.search(QUESTION, 10, filter=filter)
            looked_up = await sections.get_many([d[0] for d in DOCUMENTS], filter=filter)
            return (
                {h.metadata["section_id"] for h in semantic},
                {h.metadata["section_id"] for h in lexical},
                set(looked_up),
            )

    return asyncio.run(run())


def test_filter_returns_2023_and_null_from_both_legs(conninfo):
    both = Filter(po_version=(2023,), include_archived=True)
    assert found(conninfo, both) == ({"s23", "smhb", "sinfo", "sold"},) * 3  # not the PO 2015; NULL rows stay
    assert found(conninfo, None) == ({d[0] for d in DOCUMENTS},) * 3  # no filter: no WHERE at all
    assert found(conninfo, Filter(po_version=(2015,), include_archived=True)) == ({"s15", "sinfo", "sold"},) * 3


def test_an_empty_filter_hides_archived_rows(conninfo):
    assert found(conninfo, Filter()) == ({"s23", "smhb", "sinfo"},) * 3


def test_empty_filtered_result_stays_empty(conninfo):
    nothing = Filter(po_version=(2031,), regulation=("mhb",))  # the NULL row is archived, the info page is no handbook
    assert found(conninfo, nothing) == (set(),) * 3
    embedder = SimpleNamespace(aembed_query=lambda text: asyncio.sleep(0, [float(x) for x in vec(0)[1:-1].split(",")]))
    profile = load_profile(Profile, PROFILE)

    async def reranker_rerank(queries, texts, timeout_s):
        return [(1.0, t) for t in sorted(set(texts))]

    async def retrieve(filter):
        async with legs(conninfo) as (vector, full_text, sections):
            service = RetrievalService(
                embedder, vector, full_text, sections, SimpleNamespace(rerank=reranker_rerank), profile.retrieval,
                profile.ranking, Timeouts(retrieval_s=10, parent_lookup_s=5, reranker_s=15), 8, 20, 1, chars_per_token=3.3,
            )  # fmt: skip
            return await service.retrieve([QUESTION], filter, question=QUESTION)

    assert asyncio.run(retrieve(nothing)) == []  # no unfiltered retry
    answered = asyncio.run(retrieve(Filter(regulation=("mhb",))))
    assert {h.metadata["section_id"] for h in answered} == {"smhb"}  # the parents of the children found, no other row


def test_regulation_filter_only_when_named(conninfo):
    assert found(conninfo, Filter()) == ({"s23", "smhb", "sinfo"},) * 3  # none named: po, mhb and info alike
    assert found(conninfo, Filter(regulation=("mhb",))) == ({"smhb"},) * 3
    assert found(conninfo, Filter(regulation=("mhb",), include_archived=True)) == ({"smhb", "sold"},) * 3  # NULL passes
    assert found(conninfo, Filter(regulation=("po", "mhb"))) == ({"s23", "smhb"},) * 3


def test_valid_from_le_date(conninfo):
    on = lambda day: found(conninfo, Filter(date=dt.date.fromisoformat(day), include_archived=True))  # noqa: E731
    assert on("2014-01-01") == ({"sinfo", "sold"},) * 3  # no version-bound row was in force yet
    assert on("2016-01-01") == ({"s15", "sinfo", "sold"},) * 3
    assert on("2023-10-01") == ({"s15", "s23", "smhb", "sinfo", "sold"},) * 3  # the day itself counts
