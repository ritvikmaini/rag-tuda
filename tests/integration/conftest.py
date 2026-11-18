"""The harness's fixtures. `just harness` starts deploy/compose.ci.yaml
(project rag-ci, rag-ci-<id> in a linked worktree) and writes deploy/ci/.generated/ (tokens, secrets, fixtures); the tests
then speak HTTP to the apps, SQL to the database and HTTP to the fakes' control endpoints. Black box: they import no product code
(tests/contracts/test_harness_contracts.py checks it), so they run unchanged across refactors.
Without a running harness (no deploy/ci/.generated/ci.env) pytest collects nothing here."""

import os
import subprocess
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from rag_testing import harness, index_view

GEN = Path(__file__).resolve().parents[2] / "deploy" / "ci" / ".generated"

# collected only while a harness runs for the example profile (`just harness example` sets RAG_CI_PROFILE=example and, by
# default, runs tests/integration)
if not (GEN / "ci.env").exists() or os.environ.get("RAG_CI_PROFILE") != "example":
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """test_startup_order.py and test_database_restart.py run last, in this order: stopping or restarting pgvector kills the
    session-scoped db and owner_db connections and, in a linked worktree, may move pgvector's host port (the harness recipe
    reads it again afterwards)."""
    last = {"test_startup_order.py": 1, "test_database_restart.py": 2}
    items.sort(key=lambda item: last.get(item.path.name, 0))  # stable: every other order stays


def secret(name: str) -> str:
    return (GEN / "secrets" / (name + ".txt")).read_text(encoding="utf-8").strip()


@pytest.fixture(scope="session")
def tokens() -> dict[str, str]:
    lines = (GEN / "ci.env").read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


@pytest.fixture(scope="session")
def assistant(tokens: dict[str, str]) -> harness.Assistant:
    return harness.Assistant(
        tokens["BACKEND_REFRESH_TOKEN"],
        GEN / "frames.jsonl",
        lambda: os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" ", 1)[0],
    )


@pytest.fixture(scope="session")
def indexer() -> Callable[..., subprocess.CompletedProcess[str]]:
    """indexer() runs `indexer build` in a one-off container of the harness project and returns the finished process."""
    return lambda: harness.build_index(GEN.parents[2])


@pytest.fixture
def llm() -> harness.FakeLLM:
    fake = harness.FakeLLM()
    fake.reset()
    return fake


@pytest.fixture
def user() -> str:
    return "harness-" + uuid.uuid4().hex[:12]


@pytest.fixture(scope="session")
def connect_app() -> Callable[[int], psycopg.Connection[Any]]:
    """connect_app(port) opens a new connection as rag_app on pgvector's host port: for a test that restarts the database."""
    return lambda port: psycopg.connect(
        host="127.0.0.1",
        port=port,
        dbname=secret("pgdatabase"),
        user="rag_app",
        password=secret("rag_app_password"),
        autocommit=True,
    )


@pytest.fixture(scope="session")
def db(connect_app: Callable[[int], psycopg.Connection[Any]]) -> Iterator[psycopg.Connection[Any]]:
    """rag_app on pgvector directly: it reads the two views and `index_generations`, granted by the indexer."""
    with connect_app(harness.DB_PORT) as conn:
        yield conn


@pytest.fixture(scope="session")
def owner_db() -> Iterator[psycopg.Connection[Any]]:
    """rag_indexer on pgvector directly: it owns the generation and staging tables, which rag_app does not read."""
    with psycopg.connect(
        host="127.0.0.1",
        port=harness.DB_PORT,
        dbname=secret("pgdatabase"),
        user="rag_indexer",
        password=secret("rag_indexer_password"),
        autocommit=True,
    ) as conn:
        yield conn


@pytest.fixture
def snapshot(owner_db: psycopg.Connection[Any]) -> Callable[..., dict[str, Any]]:
    def take(parents: str = index_view.PARENTS, children: str = index_view.CHILDREN) -> dict[str, Any]:
        return index_view.snapshot(
            owner_db.execute(sql.SQL(index_view.PARENTS_SQL).format(sql.Identifier(parents))).fetchall(),
            owner_db.execute(sql.SQL(index_view.CHILDREN_SQL).format(sql.Identifier(children))).fetchall(),
        )

    return take


@pytest.fixture(scope="session")
def first_build(
    assistant: harness.Assistant,
    indexer: Callable[..., subprocess.CompletedProcess[str]],
    db: psycopg.Connection[Any],
) -> dict[str, Any]:
    """The session's first build on the empty database, with what the assistant said before it and right after it.
    Every test that asks a question depends on it."""
    assert harness.live_generation(db) is None, "the harness database must start empty (just harness ends with down -v)"
    before = assistant.ask(harness.THESIS, "harness-before")
    build = indexer()
    if build.returncode == 0:
        assistant.wait_ready()  # the backend loads the new index by itself
    after = assistant.ask(harness.THESIS, "harness-after")
    return {"before": before, "build": build, "after": after}


@pytest.fixture
def ready(first_build: dict[str, Any]) -> None:
    """A test that asks questions needs the index (the first build) to exist."""
    build = first_build["build"]
    assert build.returncode == 0, build.stdout + build.stderr
