"""rag_app on a fresh cluster: the init scripts give it
no default privilege and no CREATE on public; the indexer grants SELECT on the registry when it creates it and on the
collection's two views at every publish."""

from typing import Any

import psycopg
from rag_testing import harness, index_view


def test_rag_app_reads_the_two_views_and_the_registry_and_no_generation_table(
    ready: None, db: psycopg.Connection[Any]
) -> None:
    live = harness.live_generation(db)
    assert live is not None
    base = index_view.generation_table(live)
    got = db.execute(
        index_view.PRIVILEGES,
        (index_view.CHILDREN, index_view.PARENTS, index_view.children_of(base), index_view.parents_of(base)),
    ).fetchone()
    assert got == (True, True, True, False, False, False, 0)
