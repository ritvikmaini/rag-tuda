"""A question works right after the database restarted: every idle
connection of the backend's pools is dead after a restart of the database, and the first question must still be answered. No /readyz in between: that probe hides the dead
connections. Black box: the compose CLI, HTTP and the fakes. conftest.py runs this file last (a pgvector restart kills the
session's own database connections and, in a linked worktree, may move its host port)."""

import subprocess
import uuid
from pathlib import Path

from rag_testing import harness

COMPOSE = ["docker", "compose", "-f", str(Path(__file__).resolve().parents[2] / "deploy" / "compose.ci.yaml")]


def health(container: str) -> str:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Health.Status}}", container], capture_output=True, text=True, timeout=30
    )
    return out.stdout.strip()


def test_a_question_works_right_after_a_database_restart(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    out = subprocess.run(COMPOSE + ["ps", "-q", "pgvector"], check=True, capture_output=True, text=True, timeout=60)
    container = out.stdout.strip()
    assert container
    try:
        subprocess.run(COMPOSE + ["restart", "pgvector"], check=True, capture_output=True, timeout=180)
        harness.wait_until(lambda: health(container) == "healthy", 90)
        llm.script(harness.ANSWER_ONE)
        reply = assistant.ask(harness.THESIS, user, turn_id=str(uuid.uuid4()), formats="v1")
        assert reply.stream[-1]["type"] == "done" and reply.stream[-1]["content"]["outcome"] == "completed", reply.raw
        assert "Antwort eins" in reply.text
    finally:  # the stack is left as found; --no-recreate: the harness's port variable must not recreate a started container
        subprocess.run(
            COMPOSE + ["up", "-d", "--wait", "--no-recreate", "pgvector"], check=True, capture_output=True, timeout=180
        )
