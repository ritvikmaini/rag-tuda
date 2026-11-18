"""The migrate one-shot: the new backend image as rag_migrator, directly on
pgvector, behind profile migrate in production and the harness, so up never starts it and an older backend image runs under
the same files (rollback-safe compose); the harness runs it before its up."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def service(path: str, name: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))["services"][name]


def test_the_migrate_service_is_a_profiled_one_shot_of_the_backend_image():
    # the CI image carries the worktree tag
    for path, image in (
        ("deploy/compose.yaml", "rag-tuda-backend:latest"),
        ("deploy/compose.ci.yaml", "rag-ci-backend${RAG_CI_TAG:-}"),
    ):
        m = service(path, "migrate")
        assert m["image"] == image and m["profiles"] == ["migrate"], path
        assert m["command"] == ["python", "-m", "assistant_api.migrate"], path
        assert sorted(s if isinstance(s, str) else s["source"] for s in m["secrets"]) == [
            "pgdatabase",
            "rag_migrator_password",
        ]
        assert "RAG_MIGRATE_HOST=pgvector" in m["environment"] and "build" not in m and "depends_on" not in m, path


def test_the_backend_does_not_depend_on_the_one_shot():
    for path in ("deploy/compose.yaml", "deploy/compose.ci.yaml"):
        assert "migrate" not in (service(path, "backend").get("depends_on") or {}), path


def test_the_harness_migrates_before_its_up():
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    recipe = justfile.split('\nharness profile="example":', 1)[1].split("\n\n", 1)[0]
    assert recipe.index("compose --profile migrate run --rm --no-deps migrate") < recipe.index(
        "compose up -d --build --wait"
    )
    db_tests = justfile.split("\ndb-tests:", 1)[1].split("\n\n", 1)[0]
    assert "--profile migrate run --rm --no-deps migrate" in db_tests
    assert "tests/assistant_api/test_migrations.py" in db_tests
    assert "tests/assistant_api/chat/test_turns_store.py" in recipe  # it needs the index the leg built


def test_the_chat_schema_script_runs_after_the_roles():
    scripts = sorted(p.name for p in (ROOT / "deploy" / "pgvector" / "init").glob("*.sql"))
    assert scripts.index("04-init-chat-schema.sql") > scripts.index("01-init-roles.sql")
    assert "CREATE SCHEMA IF NOT EXISTS chat AUTHORIZATION rag_migrator;" in (
        ROOT / "deploy" / "pgvector" / "init" / "04-init-chat-schema.sql"
    ).read_text(encoding="utf-8")
