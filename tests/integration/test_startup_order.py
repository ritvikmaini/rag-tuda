"""The backend heals itself after a bad start order: pgvector is down while the
backend starts, so the first load fails and /readyz says 503; once pgvector is back, the backend's own retry loads the index
within about a minute, with no question and no reload call in between. Black box: HTTP, SQL as rag_app and the compose CLI.
conftest.py runs this file late: stopping pgvector kills the session's own database connections."""

import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import psycopg
from rag_testing import endpoints, harness

COMPOSE = ["docker", "compose", "-f", str(Path(__file__).resolve().parents[2] / "deploy" / "compose.ci.yaml")]


# backend_url and assistant_ready as in test_turns_persist.py: importlib mode allows no sibling import
def backend_url() -> str:
    """The backend's host port as Docker publishes it now (a restart may pick a new one)."""
    out = subprocess.run(COMPOSE + ["port", "backend", "8000"], check=True, capture_output=True, text=True, timeout=60)
    return "http://127.0.0.1:" + out.stdout.strip().splitlines()[0].rsplit(":", 1)[1]


def status(url: str) -> int | None:
    try:
        return httpx.get(url + endpoints.READYZ, timeout=2).status_code
    except httpx.HTTPError:
        return None


def assistant_ready(url: str) -> bool:
    return status(url) == 200


def test_the_backend_loads_the_index_by_itself_once_the_database_is_back(
    ready: None,
    assistant: harness.Assistant,
    llm: harness.FakeLLM,
    user: str,
    connect_app: Callable[[int], psycopg.Connection[Any]],
) -> None:
    out = subprocess.run(COMPOSE + ["ps", "-q", "pgvector"], check=True, capture_output=True, text=True, timeout=60)
    pgvector = out.stdout.strip()
    try:
        subprocess.run(["docker", "stop", pgvector], check=True, capture_output=True, timeout=60)
        subprocess.run(COMPOSE + ["restart", "--no-deps", "backend"], check=True, capture_output=True, timeout=180)
        harness.wait_until(lambda: status(backend_url()) == 503, 120)  # the start's load failed (the port may change)
        subprocess.run(["docker", "start", pgvector], check=True, capture_output=True, timeout=60)
        url = backend_url()
        harness.wait_until(lambda: assistant_ready(url), 90)  # no question in between: the heal task does it
        assistant.client.base_url = httpx.URL(url)
        llm.script(harness.ANSWER_ONE)
        assistant.ask(harness.THESIS, user, turn_id=str(uuid.uuid4()))
        port = subprocess.run(
            COMPOSE + ["port", "pgvector", "5432"], check=True, capture_output=True, text=True, timeout=60
        ).stdout.strip()  # pgvector's host port as Docker publishes it now
        with connect_app(int(port.rsplit(":", 1)[1])) as db:
            rows = db.execute("SELECT outcome FROM chat.turns WHERE conversation_id = %s", (user,)).fetchall()
        assert [r[0] for r in rows] == ["completed"]
    finally:
        subprocess.run(["docker", "start", pgvector], check=True, capture_output=True, timeout=60)
        url = backend_url()
        try:
            harness.wait_until(lambda: assistant_ready(url), 120)
        except TimeoutError:  # the stack is left as found
            subprocess.run(COMPOSE + ["restart", "--no-deps", "backend"], check=True, capture_output=True, timeout=180)
            url = backend_url()
            harness.wait_until(lambda: assistant_ready(url), 120)
        assistant.client.base_url = httpx.URL(url)
