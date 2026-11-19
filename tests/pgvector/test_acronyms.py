"""Acronym groups and language routing on the keyword leg, on a fresh cluster as rag_app: a group is one OR group inside the
main AND query and is not dropped as frequent; an English question searches the English column besides the German one.
RUN_DOCKER_TESTS=1."""

import asyncio
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from psycopg.conninfo import make_conninfo

from indexer.index import ddl
from indexer.index import generations as gen
from rag_core.adapters import pg_index
from rag_core.domain.filters import Filter
from rag_core.settings import load_profile
from rag_core.settings.profile import Acronyms

ACRONYMS = load_profile(Acronyms, Path(__file__).resolve().parents[2] / "profiles" / "tu-informatik" / "acronyms.yaml")
GROUPS = tuple(group.terms for group in ACRONYMS.groups)
TARGET = gen.Target.for_collection("rag")
DIMS = 768
# antrag-n: forty sections that all say "Prüfungsordnung" and "Antrag", far above the frequent-term share
FILLER = tuple((f"antrag-{i}", f"Die Prüfungsordnung regelt den Antrag Nummer {i}.", "de", "po") for i in range(40))
# section_id, content, lang, regulation
DOCUMENTS = (
    ("long", "Die Prüfungsordnung regelt die Bachelorarbeit.", "de", "po"),
    ("short", "Gemäß PO beginnt die Masterarbeit im Sommer.", "de", "mhb"),
    ("plain", "Die Bachelorarbeit braucht eine Betreuung.", "de", "info"),
    ("zeugnis", "Das Zeugnis wird nach der letzten Prüfung ausgestellt.", "de", "info"),
    ("rule", "Die Prüfungsordnung regelt den Rücktritt von einer Prüfung.", "de", "po"),  # German only: no fts_en
    ("english", "Withdrawal from an examination is possible until the deadline.", "en", "info"),
    *FILLER,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="image tests run only with RUN_DOCKER_TESTS=1 and a running Docker",
)


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
    columns = "regulation, lang"
    for i, (section_id, content, lang, regulation) in enumerate(DOCUMENTS):
        conn.execute(
            f'INSERT INTO "{names["parents"]}" (section_id, document_id, collection, source, audience, part_index, content,'
            f" metadata, {columns}) VALUES (%s, %s, 'rag', 'example.web', 'public', 0, %s, '{{}}'::jsonb, %s, %s)",
            (section_id, section_id, content, regulation, lang),
        )
        conn.execute(
            f'INSERT INTO "{names["children"]}" (chunk_id, section_id, document_id, collection, source, audience, content,'
            f" embedding, {columns}) VALUES (%s, %s, %s, 'rag', 'example.web', 'public', %s, %s::vector, %s, %s)",
            (
                uuid.UUID(int=i).hex,
                section_id,
                section_id,
                content,
                "[" + ",".join(["0.1"] * DIMS) + "]",
                regulation,
                lang,
            ),
        )
    for statement in ddl.create_indexes(names["parents"], names["children"]):
        conn.execute(statement)
    gen.verify(conn, TARGET, n, {"children": len(DOCUMENTS), "parents": len(DOCUMENTS)})
    gen.publish(conn, TARGET, n)
    conn.close()
    return make_conninfo(host="127.0.0.1", port=port, dbname="vector_db", user="rag_app", password="s" * 64)


@asynccontextmanager
async def full_text(conninfo, groups):
    pool = pg_index.read_pool(conninfo, max_size=4)
    await pool.open(wait=True)
    try:
        yield pg_index.PgFullText(pool, "rag_chunks", "german", acronyms=groups)
    finally:
        await pool.close()


def found(conninfo, query, k=100, groups=GROUPS, filter=None):
    """The section ids the keyword leg returns for the query, best first."""

    async def run():
        async with full_text(conninfo, groups) as leg:
            return [h.metadata["section_id"] for h in await leg.search(query, k, filter)]

    return asyncio.run(run())


def test_po_matches_pruefungsordnung(conninfo):
    # the question says PO, the section says Prüfungsordnung (the one section with Bachelorarbeit that does): the group
    # is inside the AND query, so the section fills k=1 without the OR fallback that "plain" would join
    assert found(conninfo, "PO Bachelorarbeit", k=1) == ["long"]
    assert found(conninfo, "Prüfungsordnung Masterarbeit", k=1) == ["short"]  # and the other way round
    assert found(conninfo, "Was regelt die Prüfungsordnung zur Bachelorarbeit?", k=2)[0] == "long"


def test_group_not_dropped_as_frequent(conninfo):
    # "Prüfungsordnung" is in 42 of 46 children: as a plain term the OR fallback would drop it, as a group it stays
    hits = found(conninfo, "PO Zeugnis")
    assert "zeugnis" in hits and {"antrag-0", "antrag-39", "long"} <= set(hits)
    assert found(conninfo, "Antrag Zeugnis") == ["zeugnis"]  # the control: a plain frequent term is still dropped


def test_english_question_searches_fts_de(conninfo):
    # the rule is in German only, so fts_en holds nothing of it; the English question still finds it, through the group
    assert "rule" in found(conninfo, "May I find the rule on withdrawing from an exam in the PO?")
    assert found(conninfo, "Can I do withdrawing?") == [
        "english"
    ]  # fts_en: "withdrawing" and "Withdrawal" share a stem there
    assert found(conninfo, "Was ist withdrawing?") == []  # a German question searches fts_de alone, where they differ


def test_the_filter_binds_the_acronym_query_too(conninfo):
    assert {"long", "short"} <= set(found(conninfo, "PO Bachelorarbeit Masterarbeit"))
    assert found(conninfo, "PO Bachelorarbeit", filter=Filter(regulation=("mhb",))) == [
        "short"
    ]  # the others are po, info


HOSTILE = ("&", "|", "!", ":*", "(", ")", "'", "\\", "''", "'; DROP TABLE rag_chunks; --", "<->", "a:*b", ")(|&!:*'\\")


@pytest.mark.parametrize("hostile", HOSTILE)
def test_operators_in_the_question_are_words_never_tsquery_syntax(conninfo, hostile):
    # no SQL or tsquery error, and the same hits as the question without the characters
    assert found(conninfo, f"{hostile} Zeugnis {hostile}") == found(conninfo, "Zeugnis") == ["zeugnis"]
    assert found(conninfo, f"PO {hostile} Bachelorarbeit", k=1) == ["long"]  # a group beside them is still one group
    found(conninfo, f"Can I do {hostile} withdrawing?")  # the English route reads both columns: no error
    assert found(conninfo, "Zeugnis") == ["zeugnis"]  # the table is still there
    if not any(c.isalpha() for c in hostile):  # only characters: no words, so nothing changes or matches
        assert found(conninfo, f"Can I do {hostile} withdrawing?") == ["english"]
        assert found(conninfo, hostile) == []


@pytest.mark.parametrize("empty", ("", "   ", "der die das"))
def test_an_empty_or_stopword_only_question_finds_nothing(conninfo, empty):
    assert found(conninfo, empty) == []
