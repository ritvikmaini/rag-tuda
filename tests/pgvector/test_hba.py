"""pgvector's tracked pg_hba.conf on a real cluster: the superuser logs in over
the socket and on loopback only, the application roles keep SCRAM over the network. A fresh container with production's
postgresql.conf, pg_hba.conf and init scripts; the Mac reaches it through a published port, so its connections arrive from
Docker's gateway, never from loopback. Needs a running Docker (Colima on the Mac): RUN_DOCKER_TESTS=1."""

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import psycopg
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PG = ROOT / "deploy" / "pgvector"
IMAGE = yaml.safe_load((ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8"))["services"]["pgvector"]["image"]
PASSWORDS = {
    "pg_vector_password": "m" * 64,
    "rag_indexer_password": "i" * 64,
    "rag_app_password": "s" * 64,
    "rag_migrator_password": "g" * 64,
}

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="image tests run only with RUN_DOCKER_TESTS=1 and a running Docker",
)


def docker(*args, check=True):
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True)


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    secrets = tmp_path_factory.mktemp("secrets")
    for name, value in {"pgdatabase": "vector_db", "vector_user": "vector_user", **PASSWORDS}.items():
        (secrets / name).write_text(value, encoding="ascii")
        (secrets / name).chmod(0o644)
    secrets.chmod(0o755)
    name = "pgvector-hba-test-" + uuid.uuid4().hex[:8]
    # secrets are copied in before the start (Colima shares only the home folder; the repository files are inside it)
    docker(
        "create",
        "--name",
        name,
        "--shm-size",
        "512m",
        "-p",
        "127.0.0.1::5432",
        "-e",
        "POSTGRES_DB_FILE=/run/secrets/pgdatabase",
        "-e",
        "POSTGRES_USER_FILE=/run/secrets/vector_user",
        "-e",
        "POSTGRES_PASSWORD_FILE=/run/secrets/pg_vector_password",
        "-v",
        "%s:/docker-entrypoint-initdb.d:ro" % (PG / "init"),
        "-v",
        "%s:/etc/postgresql/postgresql.conf:ro" % (PG / "postgresql.conf"),
        "-v",
        "%s:/etc/postgresql/pg_hba.conf:ro" % (PG / "pg_hba.conf"),
        IMAGE,
        "postgres",
        "-c",
        "config_file=/etc/postgresql/postgresql.conf",
        "-c",
        "shared_buffers=256MB",
    )
    docker("cp", str(secrets), name + ":/run/secrets")
    docker("start", name)
    try:
        for _ in range(120):
            time.sleep(1)
            logs = docker("logs", name, check=False)
            if "PostgreSQL init process complete" in logs.stdout + logs.stderr:
                break
        for _ in range(30):
            if (
                docker(
                    "exec", name, "pg_isready", "-h", "localhost", "-U", "vector_user", "-d", "vector_db", check=False
                ).returncode
                == 0
            ):
                break
            time.sleep(1)
        port = int(docker("port", name, "5432/tcp").stdout.splitlines()[0].rsplit(":", 1)[1])
        yield name, port
    finally:
        docker("rm", "-f", name, check=False)


def login(port, user, password):
    return psycopg.connect(
        host="127.0.0.1", port=port, dbname="vector_db", user=user, password=password, connect_timeout=10
    )


def test_the_superuser_is_refused_over_the_network(cluster):
    _, port = cluster
    with pytest.raises(psycopg.OperationalError, match="pg_hba.conf rejects connection"):
        login(port, "vector_user", PASSWORDS["pg_vector_password"])


def test_the_application_roles_log_in_over_the_network(cluster):
    _, port = cluster
    for user, secret in (
        ("rag_app", "rag_app_password"),
        ("rag_indexer", "rag_indexer_password"),
        ("rag_migrator", "rag_migrator_password"),
    ):
        with login(port, user, PASSWORDS[secret]) as conn:
            assert conn.execute("SELECT current_user").fetchone() == (user,)


def test_the_superuser_logs_in_on_the_socket_and_on_loopback(cluster):
    name, _ = cluster
    for extra in ([], ["-h", "127.0.0.1"]):
        out = docker(
            "exec", name, "psql", "-U", "vector_user", "-d", "vector_db", "-X", "-A", "-t", *extra, "-c", "SELECT 1"
        )
        assert out.stdout.strip() == "1", extra
