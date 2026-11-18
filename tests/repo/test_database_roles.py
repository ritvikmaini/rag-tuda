"""Database roles: each service reaches Postgres as its own role through
the files the code already opens, the init scripts
create least-privilege roles from secret files, and no tracked ini or sql file carries a database password."""

import os
import re
import subprocess

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def text(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def load(name):
    return yaml.safe_load(text(name))


def service(name):
    return load("deploy/compose.yaml")["services"][name]


def secret_targets(name):
    """{file name under /run/secrets: compose secret it comes from} for one service."""
    targets = {}
    for entry in service(name).get("secrets") or []:
        if isinstance(entry, dict):
            targets[entry.get("target") or entry["source"]] = entry["source"]
        else:
            targets[entry] = entry
    return targets


def test_the_backend_reads_the_app_role_through_the_files_the_code_opens():
    assert secret_targets("backend") == {
        "pg_vector_password": "rag_app_password",
        "vector_user": "rag_app_user",
        "pgdatabase": "pgdatabase",
        "backend_token": "backend_token",
    }


def test_the_indexer_reads_the_indexer_role_and_no_longer_mounts_the_secrets_folder():
    assert secret_targets("indexer") == {
        "pg_vector_password": "rag_indexer_password",
        "vector_user": "rag_indexer_user",
        "pgdatabase": "pgdatabase",
    }
    assert [v for v in service("indexer")["volumes"] if "secrets" in str(v)] == []


def test_the_migrate_one_shot_reads_the_migrator_role_through_its_own_secret_files():
    # a profile service, so up never starts it; it reaches pgvector directly as rag_migrator
    assert secret_targets("migrate") == {"pgdatabase": "pgdatabase", "rag_migrator_password": "rag_migrator_password"}


def test_pgvector_gets_every_role_password_for_its_init_scripts():
    assert set(secret_targets("pgvector")) == {
        "pg_vector_password",
        "pgdatabase",
        "vector_user",
        "rag_indexer_password",
        "rag_app_password",
        "rag_migrator_password",
    }


def test_the_role_secrets_point_at_generated_password_files_and_tracked_role_names():
    files = {
        name: spec["file"] for name, spec in load("deploy/compose.yaml")["secrets"].items() if name.startswith("rag_")
    }
    assert {k: v for k, v in files.items() if k.startswith("rag_")} == {
        "rag_indexer_password": "../secrets/rag_indexer_password.txt",
        "rag_app_password": "../secrets/rag_app_password.txt",
        "rag_migrator_password": "../secrets/rag_migrator_password.txt",
        "rag_indexer_user": "./pgvector/roles/rag_indexer.txt",
        "rag_app_user": "./pgvector/roles/rag_app.txt",
    }
    assert (text("deploy/pgvector/roles/rag_indexer.txt"), text("deploy/pgvector/roles/rag_app.txt")) == (
        "rag_indexer",
        "rag_app",
    )


def test_no_tracked_ini_or_sql_file_carries_a_database_password():
    tracked = subprocess.run(
        ["git", "ls-files", "*.ini", "*.sql"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    assert tracked, "git ls-files found nothing (not a git checkout?)"
    hits = [
        (path, m.group(0))
        for path in tracked
        for m in re.finditer(r"(?i)\bpassword=\S+|\bPASSWORD\s+'[^']*'", text(path))
        if not m.group(0).startswith(("password=<", "PASSWORD '<"))
    ]
    assert hits == []
    assert "deploy/pgvector/init/01-init-user-and-db.sql" not in tracked


def test_the_init_script_creates_the_roles_from_the_secret_files_without_superuser():
    sql = text("deploy/pgvector/init/01-init-roles.sql")
    for role, short in (("rag_indexer", "indexer"), ("rag_app", "app"), ("rag_migrator", "migrator")):
        assert re.search(r"^\\set pw_%s `cat /run/secrets/%s_password`$" % (short, role), sql, re.M), role
        assert "CREATE ROLE %s LOGIN PASSWORD :'pw_%s';" % (role, short) in sql, role
    assert "SUPERUSER" not in sql and "CREATEDB" not in sql and "CREATEROLE" not in sql
    assert (
        "GRANT USAGE, CREATE ON SCHEMA public TO rag_indexer;" in sql
        and "GRANT USAGE ON SCHEMA public TO rag_app;" in sql
    )
    assert "ALTER DEFAULT PRIVILEGES" not in sql  # the indexer grants the views and the registry itself
    assert "rag_migrator" not in sql.split("GRANT USAGE, CREATE ON SCHEMA public")[1].splitlines()[0]
    assert sql.index("\\set ON_ERROR_STOP on") < sql.index("CREATE ROLE")


def test_the_init_script_names_the_password_file_it_misses():
    sql = text("deploy/pgvector/init/01-init-roles.sql")
    for role in ("rag_indexer", "rag_app", "rag_migrator"):
        assert "RAISE EXCEPTION 'init-roles: /run/secrets/%s_password is missing" % role in sql, role


def test_the_init_script_never_logs_a_password_and_can_run_twice():
    sql = text("deploy/pgvector/init/01-init-roles.sql")
    first = sql.index("CREATE ROLE")
    assert sql.index("SET log_min_error_statement = 'panic';") < first
    assert sql.index("SET log_statement = 'none';") < first
    for role, flag in (("rag_indexer", "new_indexer"), ("rag_app", "new_app"), ("rag_migrator", "new_migrator")):
        assert "NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '%s') AS %s" % (role, flag) in sql, role
        guard = sql.index("\\if :%s" % flag)
        assert guard < sql.index("CREATE ROLE %s " % role) < sql.index("\\endif", guard), role


def test_the_checkpoint_tables_are_created_as_rag_app():
    sql = text("deploy/pgvector/init/03-init-checkpoints.sql")
    grant, set_role = sql.index("GRANT CREATE ON SCHEMA public TO rag_app;"), sql.index("SET ROLE rag_app;")
    assert (
        sql.index("BEGIN;") < grant < set_role < sql.index("CREATE TABLE")
    )  # one transaction: a failure leaves no CREATE behind
    assert sql.rstrip().endswith("RESET ROLE;\nREVOKE CREATE ON SCHEMA public FROM rag_app;\nCOMMIT;")
