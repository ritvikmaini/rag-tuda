"""Stable ids and the embedding cache: builds from the inputs of the last build, left staging, keep the live index's section
and chunk ids and embed nothing."""

import re
import subprocess
from typing import Any

import psycopg
from psycopg import sql
from rag_testing import harness, index_view

FROZEN = ("--reuse-inputs", "--no-swap")
STAGED = re.compile(
    r"^generation (\d+) staged \(not published\): (\d+) sections, (\d+) chunks, 0 embedded, (\d+) cached; "
    r"0 added, 0 changed, 0 removed$",
    re.M,
)


def ids(db: psycopg.Connection[Any], base: str) -> tuple[set[str], set[str]]:
    sections = db.execute(sql.SQL("SELECT section_id FROM {}").format(sql.Identifier(index_view.parents_of(base))))
    chunks = db.execute(sql.SQL("SELECT chunk_id FROM {}").format(sql.Identifier(index_view.children_of(base))))
    return {r[0] for r in sections.fetchall()}, {r[0] for r in chunks.fetchall()}


def staged(build: subprocess.CompletedProcess[str]) -> str:
    """The generation table a frozen build left staging; the build embedded nothing and changed nothing, every chunk came from
    the cache."""
    assert build.returncode == 0, build.stdout + build.stderr
    found = STAGED.search(build.stdout)
    assert found, build.stdout
    assert found[3] == found[4], build.stdout
    return index_view.generation_table(int(found[1]))


def test_two_frozen_builds_keep_the_live_ids_and_embed_nothing(
    first_build: dict[str, Any],
    indexer: Any,
    db: psycopg.Connection[Any],
    owner_db: psycopg.Connection[Any],
) -> None:
    live = harness.live_generation(db)
    assert live is not None
    one, two = staged(indexer(*FROZEN)), staged(indexer(*FROZEN))
    assert one != two
    assert ids(owner_db, one) == ids(owner_db, two) == ids(owner_db, index_view.generation_table(live))
    assert harness.live_generation(db) == live  # nothing was published
