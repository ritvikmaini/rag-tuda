"""Database roles on a real pgvector cluster. A fresh container runs the init scripts with fake secret files; the tests
check the roles, SCRAM login over TCP, the checkpoint tables' owner and a second run of the init script. Needs a running
Docker (Colima on the Mac): RUN_DOCKER_TESTS=1."""

import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "docker.io/pgvector/pgvector:pg16"
PASSWORDS = {
    "pg_vector_password": "m" * 64,
    "rag_indexer_password": "i" * 64,
    "rag_app_password": "s" * 64,
    "rag_migrator_password": "g" * 64,
}
VERIFIER = re.compile(r'^"rag_(indexer|app)" "SCRAM-SHA-256\$\d+:[A-Za-z0-9+/=]+\$[A-Za-z0-9+/=]+:[A-Za-z0-9+/=]+"$')
PSQL = ["psql", "-U", "vector_user", "-d", "vector_db", "-X"]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="image tests run only with RUN_DOCKER_TESTS=1 and a running Docker",
)


def docker(*args, check=True, input=None):
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True, input=input)


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    secrets = tmp_path_factory.mktemp("secrets")
    for name, value in {"pgdatabase": "vector_db", "vector_user": "vector_user", **PASSWORDS}.items():
        (secrets / name).write_text(value, encoding="ascii")
        (secrets / name).chmod(0o644)
    secrets.chmod(0o755)  # psql's backticks read the files as the container's postgres user; docker cp keeps the modes
    name = "pgvector-roles-test-" + uuid.uuid4().hex[:8]
    # The secret files are copied in before the start: a bind mount of a temporary folder is empty under Colima, which
    # shares only the home folder with its VM (the scripts below live in the repository, inside the home folder).
    docker(
        "create",
        "--name",
        name,
        "-e",
        "POSTGRES_DB_FILE=/run/secrets/pgdatabase",
        "-e",
        "POSTGRES_USER_FILE=/run/secrets/vector_user",
        "-e",
        "POSTGRES_PASSWORD_FILE=/run/secrets/pg_vector_password",
        "-v",
        "%s:/docker-entrypoint-initdb.d:ro" % (ROOT / "deploy" / "pgvector" / "init"),
        IMAGE,
    )
    docker("cp", str(secrets), name + ":/run/secrets")
    docker("start", name)
    try:
        logs = ""
        for _ in range(120):
            time.sleep(1)
            result = docker("logs", name, check=False)
            logs = result.stdout + result.stderr
            if "PostgreSQL init process complete" in logs:
                break
        else:
            raise RuntimeError("init did not finish:\n" + logs[-3000:])
        for _ in range(30):
            if docker("exec", name, "pg_isready", "-U", "vector_user", "-d", "vector_db", check=False).returncode == 0:
                break
            time.sleep(1)
        yield name
    finally:
        docker("rm", "-f", "-v", name, check=False)


def psql(name, sql, *extra, check=True):
    return docker("exec", "-i", name, *PSQL, "-A", "-t", "-v", "ON_ERROR_STOP=1", *extra, check=check, input=sql)


def rows(name, sql):
    return psql(name, sql).stdout.strip().splitlines()


def test_the_init_scripts_create_three_login_roles_with_scram_secrets_and_no_superuser(cluster):
    assert rows(
        cluster,
        "SELECT rolname || '|' || rolsuper || '|' || rolcanlogin || '|' || left(rolpassword, 13) "
        "FROM pg_authid WHERE rolname LIKE 'rag\\_%' ORDER BY rolname",
    ) == [
        "rag_app|false|true|SCRAM-SHA-256",
        "rag_indexer|false|true|SCRAM-SHA-256",
        "rag_migrator|false|true|SCRAM-SHA-256",
    ]
    assert rows(cluster, "SELECT rolname FROM pg_roles WHERE rolsuper AND rolcanlogin") == ["vector_user"]
    assert rows(cluster, "SELECT extname FROM pg_extension WHERE extname = 'vector'") == ["vector"]


def test_the_checkpoint_tables_belong_to_rag_app_and_no_default_privilege_exists(cluster):
    assert rows(
        cluster,
        "SELECT tablename || '|' || tableowner FROM pg_tables WHERE tablename LIKE 'checkpoint%' "
        'ORDER BY tablename COLLATE "C"',
    ) == [
        "checkpoint_blobs|rag_app",
        "checkpoint_migrations|rag_app",
        "checkpoint_writes|rag_app",
        "checkpoints|rag_app",
    ]
    assert rows(cluster, "SELECT count(*) FROM pg_default_acl") == ["0"]
    assert rows(
        cluster,
        "SELECT 'x' || has_schema_privilege('rag_indexer', 'public', 'CREATE') || "
        "has_schema_privilege('rag_app', 'public', 'CREATE') || "
        "has_schema_privilege('rag_app', 'public', 'USAGE') || "
        "has_schema_privilege('rag_migrator', 'public', 'CREATE')",
    ) == ["xtruefalsetruefalse"]


def test_scram_login_over_tcp_works_for_rag_app_with_the_file_password(cluster):
    ip = docker("inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", cluster).stdout.strip()
    assert rows(cluster, "SELECT auth_method FROM pg_hba_file_rules WHERE type = 'host' AND address = 'all'") == [
        "scram-sha-256"
    ]
    good = docker(
        "exec",
        "-e",
        "PGPASSWORD=" + PASSWORDS["rag_app_password"],
        cluster,
        "psql",
        "-h",
        ip,
        "-U",
        "rag_app",
        "-d",
        "vector_db",
        "-X",
        "-A",
        "-t",
        "-c",
        "SELECT current_user",
    )
    assert good.stdout.strip() == "rag_app"
    bad = docker(
        "exec",
        "-e",
        "PGPASSWORD=wrong",
        cluster,
        "psql",
        "-h",
        ip,
        "-U",
        "rag_app",
        "-d",
        "vector_db",
        "-X",
        "-c",
        "SELECT 1",
        check=False,
    )
    assert bad.returncode != 0 and "password authentication failed" in bad.stderr


def test_the_bootstrap_query_returns_two_scram_verifier_lines(cluster):
    lines = rows(
        cluster,
        "SELECT chr(34) || rolname || chr(34) || chr(32) || chr(34) || rolpassword || chr(34) "
        "FROM pg_authid WHERE rolname IN ('rag_indexer', 'rag_app') ORDER BY rolname",
    )
    assert len(lines) == 2 and all(VERIFIER.match(line) for line in lines)


def test_the_init_roles_script_runs_again_without_error_and_logs_no_password(cluster):
    acl = "SELECT count(*) FROM pg_default_acl d, aclexplode(d.defaclacl) a WHERE a.grantee = 'rag_app'::regrole"
    before = rows(cluster, acl)
    result = docker(
        "exec",
        cluster,
        *PSQL,
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        "/docker-entrypoint-initdb.d/01-init-roles.sql",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert rows(cluster, acl) == before
    assert rows(
        cluster, "SELECT count(*) FROM pg_roles WHERE rolname IN ('rag_indexer', 'rag_app', 'rag_migrator')"
    ) == ["3"]
    failing = docker(
        "exec",
        "-i",
        cluster,
        *PSQL,
        "-v",
        "ON_ERROR_STOP=1",
        check=False,
        input="SET log_min_error_statement = 'panic';\nCREATE ROLE rag_app LOGIN PASSWORD 'zzzz-not-logged';\n",
    )
    assert failing.returncode != 0 and "already exists" in failing.stderr
    logs = docker("logs", cluster, check=False)
    assert "zzzz-not-logged" not in logs.stdout + logs.stderr
    assert PASSWORDS["rag_app_password"] not in logs.stdout + logs.stderr
    control = docker(
        "exec", "-i", cluster, *PSQL, check=False, input="CREATE ROLE rag_app LOGIN PASSWORD 'yyyy-control';\n"
    )
    assert control.returncode == 0 and "already exists" in control.stderr
    logs = docker("logs", cluster, check=False)
    assert "yyyy-control" in logs.stdout + logs.stderr  # without the SET the server logs the failing statement


@pytest.mark.parametrize("role", ["rag_indexer", "rag_app", "rag_migrator"])
def test_a_missing_password_file_stops_the_script_and_is_named(cluster, role):
    path = "/run/secrets/%s_password" % role
    docker("exec", cluster, "mv", path, path + ".away")
    try:
        result = docker(
            "exec",
            cluster,
            *PSQL,
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            "/docker-entrypoint-initdb.d/01-init-roles.sql",
            check=False,
        )
    finally:
        docker("exec", cluster, "mv", path + ".away", path)
    assert result.returncode != 0 and "init-roles: %s is missing" % path in result.stderr
