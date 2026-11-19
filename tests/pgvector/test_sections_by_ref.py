"""The reference lookup over the sections view, on a fresh cluster as rag_app: a section is found by its regulation and by
the "§ n" its parent records (`paragraphs`) or its section path carries, narrowed by programme and PO version where given and
by the question's filter. RUN_DOCKER_TESTS=1."""

import asyncio
import json
import os
import shutil
import uuid

import pytest
from psycopg.conninfo import make_conninfo

from indexer.index import ddl
from indexer.index import generations as gen
from rag_core.adapters import pg_index
from rag_core.domain.filters import Filter

TARGET = gen.Target.for_collection("rag")
DIMS = 768
BSC, MSC = ["bsc-informatik"], ["msc-informatik"]
# section_id, regulation, programme, po_version, archived, part_index, section_path, paragraphs
DOCUMENTS = (
    ("apb22", "apb", None, None, False, 0, "APB > § 22 Anerkennung", ["§ 22"]),
    ("apb22b", "apb", None, None, False, 1, "APB > § 22 Anerkennung", None),  # the part after the heading's part
    ("apb220", "apb", None, None, False, 2, "APB > § 220 Anderes", ["§ 220"]),  # a longer number is another paragraph
    (
        "apb5-6",
        "apb",
        None,
        None,
        False,
        3,
        "APB > Abschnitt II",
        ["§ 5", "§ 6"],
    ),  # merged: the path keeps the prefix only
    ("apb22old", "apb", None, None, True, 4, "APB > § 22 Anerkennung", ["§ 22"]),
    ("po23", "po", BSC, [2023], False, 0, "PO > § 5 Fristen", ["§ 5"]),
    ("po15", "po", BSC, [2015], False, 0, "PO > § 5 Fristen", ["§ 5"]),
    ("pomsc", "po", MSC, [2023], False, 0, "PO > § 5 Fristen", ["§ 5"]),
    ("mhb23", "mhb", BSC, [2023], False, 0, "MHB > § 5 Module", ["§ 5"]),
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
    columns = "regulation, programme, po_version, archived, lang"
    for i, (section_id, regulation, programme, po_version, archived, part, path, marked) in enumerate(DOCUMENTS):
        typed = (regulation, programme, po_version, archived)
        metadata = {"section_path": path, **({"paragraphs": marked} if marked else {})}
        conn.execute(
            f'INSERT INTO "{names["parents"]}" (section_id, document_id, collection, source, audience, part_index, content,'
            f" metadata, {columns}) VALUES (%s, %s, 'rag', 'example.web', 'public', %s, %s, %s::jsonb, %s, %s, %s, %s, 'de')",
            (section_id, regulation, part, f"Text von {section_id}", json.dumps(metadata), *typed),
        )
        conn.execute(
            f'INSERT INTO "{names["children"]}" (chunk_id, section_id, document_id, collection, source, audience, content,'
            f" embedding, {columns}) VALUES (%s, %s, %s, 'rag', 'example.web', 'public', %s, %s::vector, %s, %s, %s, %s,"
            " 'de')",
            (
                uuid.UUID(int=i).hex,
                section_id,
                regulation,
                f"Text von {section_id}",
                "[" + ",".join(["0.1"] * DIMS) + "]",
                *typed,
            ),
        )
    for statement in ddl.create_indexes(names["parents"], names["children"]):
        conn.execute(statement)
    gen.verify(conn, TARGET, n, {"children": len(DOCUMENTS), "parents": len(DOCUMENTS)})
    gen.publish(conn, TARGET, n)
    conn.close()
    return make_conninfo(host="127.0.0.1", port=port, dbname="vector_db", user="rag_app", password="s" * 64)


def by_ref(conninfo, regulation, paragraph, programme=None, po_version=None, filter=Filter()):
    """The section ids `sections_by_ref` finds."""

    async def run():
        pool = pg_index.read_pool(conninfo, max_size=2)
        await pool.open(wait=True)
        try:
            found = await pg_index.PgSections(pool, "rag_sections").sections_by_ref(
                regulation, paragraph, programme, po_version, filter
            )
            return [s.section_id for s in found]
        finally:
            await pool.close()

    return asyncio.run(run())


def test_a_paragraph_is_found_by_its_record_or_its_path(conninfo):
    assert by_ref(conninfo, "apb", "§ 22") == ["apb22", "apb22b"]  # the heading's part and the part that continues it
    assert by_ref(conninfo, "apb", "§ 220") == ["apb220"]


def test_merged_paragraph_is_found(conninfo):
    assert by_ref(conninfo, "apb", "§ 6") == ["apb5-6"]  # not in the path, only in the parent's `paragraphs`
    assert by_ref(conninfo, "apb", "§ 5") == ["apb5-6"]
    assert by_ref(conninfo, "apb", "§ 7") == []


def test_programme_and_version_narrow_a_programme_regulation(conninfo):
    assert by_ref(conninfo, "po", "§ 5", BSC, [2023]) == ["po23"]
    assert by_ref(conninfo, "po", "§ 5", BSC, [2015]) == ["po15"]
    assert by_ref(conninfo, "po", "§ 5", MSC, [2023]) == ["pomsc"]
    assert by_ref(conninfo, "po", "§ 5", BSC, [2031]) == []
    assert by_ref(conninfo, "po", "§ 5", BSC, None) == [
        "po15",
        "po23",
    ]  # no version named: every version of the programme
    assert by_ref(conninfo, "mhb", "§ 5", BSC, [2023]) == ["mhb23"]


def test_a_regulation_without_programme_passes_a_programme(conninfo):
    assert by_ref(conninfo, "apb", "§ 22", BSC, [2023]) == ["apb22", "apb22b"]  # NULL rows apply to every programme


def test_the_question_filter_applies(conninfo):
    assert by_ref(conninfo, "apb", "§ 22", filter=Filter(include_archived=True)) == ["apb22", "apb22b", "apb22old"]
    assert by_ref(conninfo, "apb", "§ 22", filter=None) == ["apb22", "apb22b", "apb22old"]  # no filter: no WHERE at all
    assert by_ref(conninfo, "po", "§ 5", filter=Filter(po_version=(2023,))) == ["po23", "pomsc"]
    assert by_ref(conninfo, "po", "§ 5", filter=Filter(programme=tuple(MSC))) == ["pomsc"]
    assert by_ref(conninfo, "apb", "§ 22", filter=Filter(programme=tuple(MSC))) == ["apb22", "apb22b"]  # NULL passes
