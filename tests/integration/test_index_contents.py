"""The index contents on the fixture sources:
parents and children normalised (rag_testing.index_view), compared with the committed snapshot; no section dated.

HARNESS_WRITE_SNAPSHOT=1 writes the snapshot instead of comparing (committed after a second run agrees)."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psycopg
import pytest
from rag_testing import index_view

SNAPSHOT = Path(__file__).parent / "snapshots" / "index_v1.json"


def test_the_live_index_equals_the_committed_snapshot(ready: None, snapshot: Callable[..., dict[str, Any]]) -> None:
    taken = snapshot()
    if os.environ.get("HARNESS_WRITE_SNAPSHOT") == "1":
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(taken, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip("snapshot written to %s" % SNAPSHOT)
    assert taken == json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_no_section_carries_a_year_or_a_date(ready: None, db: psycopg.Connection[Any]) -> None:
    row = db.execute(index_view.YEARS).fetchone()
    assert row is not None and row[0] == 0
