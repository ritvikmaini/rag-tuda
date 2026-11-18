"""The example profile's leg: the real indexer and assistant images, built with
PROFILE=example, index the ten law texts of profiles/example/documents through the folder connector into the collection
`rag` and answer from them. rag_app reads the new collection through the grants of its first publish alone."""

import json
from typing import Any

import psycopg
import pytest
from psycopg import errors
from rag_testing import harness, index_view

BASE = "https://www.gesetze-im-internet.de/"
LEAVE_LAW = BASE + "burlg"
LEAVE = "Wie viele Werktage Urlaub stehen Beschäftigten mindestens zu?"
CITE_ALL = {
    "purpose": "answer",
    "reply": "Mindestens vierundzwanzig Werktage " + " ".join(f"[{n}]" for n in range(1, 15)) + ".",
}
ORIGIN = "Gesetze im Internet"


@pytest.fixture(scope="module")
def built(first_build: dict[str, Any]) -> str:
    """What `indexer build` printed on the empty database: the session's first build, shared with the rest of the suite."""
    build = first_build["build"]
    assert build.returncode == 0, build.stdout + build.stderr
    return build.stdout


def test_the_first_build_publishes_generation_1_and_prints_its_counts(built: str, db: psycopg.Connection[Any]) -> None:
    assert harness.live_generation(db) == 1
    lines = built.splitlines()
    assert lines[0] == "source gesetze: 10 documents"
    assert lines[1].startswith("generation 1 published (previous none): ")


def test_every_section_is_the_fixture_connectors_public_and_undated(built: str, db: psycopg.Connection[Any]) -> None:
    rows = db.execute(
        "SELECT collection, source, audience, count(DISTINCT document_id), "
        "count(*) FILTER (WHERE year IS NOT NULL OR date IS NOT NULL) FROM rag_sections GROUP BY 1, 2, 3"
    ).fetchall()
    assert rows == [("rag", "folder", "public", 10, 0)]
    urls = {r[0] for r in db.execute("SELECT DISTINCT url FROM rag_sections").fetchall()}
    assert len(urls) == 10 and all(u.startswith(BASE) for u in urls) and LEAVE_LAW in urls


def test_rag_app_reads_the_example_views_and_the_registry_and_nothing_else(
    built: str, db: psycopg.Connection[Any]
) -> None:
    got = db.execute(
        index_view.PRIVILEGES, ("rag_chunks", "rag_sections", "rag_g0001_chunks", "rag_g0001_sections")
    ).fetchone()
    assert got == (True, True, True, False, False, False, 0)
    with pytest.raises(errors.InsufficientPrivilege):
        db.execute("SELECT 1 FROM rag_g0001_sections LIMIT 1")


def test_a_lookup_cites_the_law_texts(
    built: str, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(CITE_ALL)
    reply = assistant.ask(LEAVE, user)
    assert reply.path == harness.LOOKUP
    entries = reply.first("citations")
    assert LEAVE_LAW in {e["url"] for e in entries}
    assert all(e["url"].startswith(BASE) and e["origin"] == ORIGIN for e in entries)
    calls = llm.calls()
    # positive control: the scanned calls are the real ones
    assert {"analysis", "answer"} <= {c["purpose"] for c in calls}
    sent = json.dumps(llm.calls("answer"), ensure_ascii=False)
    assert f"{ORIGIN} | Gesetz" in sent and "Link: " + LEAVE_LAW in sent  # the source headers reached the model


def test_an_out_of_scope_message_gets_the_direct_reply(
    built: str, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(harness.OUT_OF_SCOPE)
    reply = assistant.ask(harness.JOKE, user)
    assert reply.path == harness.DIRECT
    calls = llm.calls()
    assert {"analysis", "direct"} <= {c["purpose"] for c in calls}
