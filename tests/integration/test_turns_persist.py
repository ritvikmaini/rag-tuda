"""Turns survive a restart: the conversation lives in the turn store, not in
the assistant's memory, so a follow-up after the assistant restarted still sees the first turn, and both turns are rows.
Black box: HTTP, SQL as rag_app and the compose CLI."""

import subprocess
import uuid
from pathlib import Path
from typing import Any

import httpx
import psycopg
from rag_testing import endpoints, harness

COMPOSE = ["docker", "compose", "-f", str(Path(__file__).resolve().parents[2] / "deploy" / "compose.ci.yaml")]


def backend_url() -> str:
    """The backend's host port as Docker publishes it now. In a linked worktree the harness lets Docker pick the ports,
    and a restart may pick a new one, so harness.BACKEND can go stale."""
    out = subprocess.run(COMPOSE + ["port", "backend", "8000"], check=True, capture_output=True, text=True, timeout=60)
    return "http://127.0.0.1:" + out.stdout.strip().splitlines()[0].rsplit(":", 1)[1]


def assistant_ready(url: str) -> bool:
    try:
        return httpx.get(url + endpoints.READYZ, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def test_a_follow_up_after_a_restart_sees_the_first_turn(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str, db: psycopg.Connection[Any]
) -> None:
    llm.script(harness.ANSWER_ONE)
    assistant.ask(harness.GARAGE, user, turn_id=str(uuid.uuid4()))
    subprocess.run(COMPOSE + ["restart", "backend"], check=True, capture_output=True, timeout=180)
    url = backend_url()  # read again after the restart
    harness.wait_until(lambda: assistant_ready(url), 120)
    assistant.client.base_url = httpx.URL(url)  # the session's client (later tests too) follows the backend's port now
    assistant.ask(harness.LIBRARY, user, turn_id=str(uuid.uuid4()))
    assert harness.history(llm, harness.LIBRARY) == harness.turn(harness.GARAGE, "Antwort eins.")
    rows = db.execute(
        "SELECT outcome FROM chat.turns WHERE conversation_id = %s ORDER BY created_at", (user,)
    ).fetchall()
    assert [r[0] for r in rows] == ["completed", "completed"]
