"""The index as the harness reads it: the SQL against the typed views and the
generation tables, and the normalised snapshot of sections and chunks. The one file a change of the index contract
edits (the v2 tables, the typed views); the harness tests themselves stay unchanged.

Normalised: the random ids (section_id, unique_id), the run's clock (scrapedAt, crawl_date) and the row
order. What stays is what a structural round must not change: url, page or part, the SHA-256 of every text and
embedding, and every other metadata key and value."""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, LiteralString

PARENTS, CHILDREN = (
    "rag_sections",
    "rag_chunks",
)  # the typed views
RUN_KEYS = frozenset({"section_id", "scrapedAt", "crawl_date", "unique_id"})
LIVE_TABLE = "SELECT table_name FROM information_schema.view_table_usage WHERE view_name = %s"
YEARS = (
    "SELECT count(*) FROM rag_sections "
    "WHERE coalesce(metadata ->> 'doc_year', '') <> '' OR coalesce(metadata ->> 'doc_date', '') <> ''"
)
LOCK = "SELECT pg_try_advisory_lock(2, hashtext('rag'))"  # the indexer's build lock (generations.py)
UNLOCK = "SELECT pg_advisory_unlock(2, hashtext('rag'))"
COLLECTION = "rag"
LAYOUT = "v2"  # the generation row's layout
GENERATION_NAME = re.compile(COLLECTION + r"_g\d{4}")  # the generation prefix of a v2 table name


def generation_table(n: int) -> str:
    return f"{COLLECTION}_g{n:04d}"


def children_of(base: str) -> str:
    return base + "_chunks"  # layout v2


def parents_of(base: str) -> str:
    return base + "_sections"  # layout v2


# {} = a table or view name, filled in as an identifier (psycopg.sql)
PARENTS_SQL: LiteralString = "SELECT t.section_id, t.content, t.metadata FROM {} t"
CHILDREN_SQL: LiteralString = "SELECT content, embedding::text, json_build_object('section_id', section_id) FROM {}"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if k not in RUN_KEYS}


def as_dict(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def snapshot(
    parents: Iterable[tuple[Any, ...]], children: Iterable[tuple[Any, ...]]
) -> dict[str, list[dict[str, Any]]]:
    """parents: (section_id, content, metadata); children: (content, embedding text, {section_id})."""
    rows = []
    for section_id, content, meta in parents:
        meta = as_dict(meta)
        position = meta.get("part_index", meta.get("page_number"))
        rows.append(
            (section_id, {"url": meta.get("url"), "position": position, "sha": sha(content), "metadata": clean(meta)})
        )
    rows.sort(key=lambda r: (str(r[1]["url"]), str(r[1]["position"]), r[1]["sha"]))
    number = {section_id: n for n, (section_id, _) in enumerate(rows)}
    kids = []
    for content, embedding, meta in children:
        meta = as_dict(meta)
        kids.append(
            {
                "parent": number.get(meta.get("section_id"), -1),
                "sha": sha(content),
                "embedding": sha(str(embedding)),
                "metadata": clean(meta),
            }
        )
    kids.sort(key=lambda k: (k["parent"], k["metadata"].get("chunk_index", 0), k["sha"]))
    return {"parents": [row for _, row in rows], "children": kids}


# rag_app's privileges: SELECT on the two views and the registry, none on a generation's tables, no CREATE on public, no
# default privilege from rag_indexer (a default privilege from rag_migrator in schema chat stays). Parameters: chunks view, sections view, a chunks table, a sections table.
PRIVILEGES: LiteralString = (
    "SELECT has_table_privilege('rag_app', %s, 'SELECT'), has_table_privilege('rag_app', %s, 'SELECT'), "
    "has_table_privilege('rag_app', 'index_generations', 'SELECT'), has_table_privilege('rag_app', %s, 'SELECT'), "
    "has_table_privilege('rag_app', %s, 'SELECT'), has_schema_privilege('rag_app', 'public', 'CREATE'), "
    "(SELECT count(*) FROM pg_default_acl d, aclexplode(d.defaclacl) a "
    "WHERE d.defaclrole = 'rag_indexer'::regrole AND a.grantee = 'rag_app'::regrole)"
)
