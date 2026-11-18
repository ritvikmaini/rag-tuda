"""A fresh pgvector cluster per test module: the init scripts and fixed
test passwords; `connect(role)` as rag_indexer, rag_app or the bootstrap superuser vector_user. RUN_DOCKER_TESTS=1 and a
running Docker."""

import subprocess
import time
import uuid
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
ROOT = Path(__file__).resolve().parents[2]
IMAGE = "docker.io/pgvector/pgvector:pg16"
PASSWORDS = {
    "pg_vector_password": "m" * 64,
    "rag_indexer_password": "i" * 64,
    "rag_app_password": "s" * 64,
    "rag_migrator_password": "g" * 64,
}


def docker(*args, check=True):
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True)


@pytest.fixture(scope="module")
def port(tmp_path_factory):
    secrets = tmp_path_factory.mktemp("secrets")
    for name, value in {"pgdatabase": "vector_db", "vector_user": "vector_user", **PASSWORDS}.items():
        (secrets / name).write_text(value, encoding="ascii")
        (secrets / name).chmod(0o644)
    secrets.chmod(0o755)
    name = "pgvector-generations-test-" + uuid.uuid4().hex[:8]
    docker(
        "create",
        "--name",
        name,
        "-p",
        "127.0.0.1::5432",
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
        for _ in range(120):
            time.sleep(1)
            logs = docker("logs", name, check=False)
            if "PostgreSQL init process complete" in logs.stdout + logs.stderr:
                break
        else:
            raise RuntimeError("init did not finish")
        host_port = int(docker("port", name, "5432/tcp").stdout.splitlines()[0].rsplit(":", 1)[1])
        for _ in range(30):
            try:
                psycopg.connect(
                    host="127.0.0.1",
                    port=host_port,
                    dbname="vector_db",
                    user="rag_indexer",
                    password=PASSWORDS["rag_indexer_password"],
                    connect_timeout=3,
                ).close()
                break
            except psycopg.OperationalError:
                time.sleep(1)
        yield host_port
    finally:
        docker("rm", "-f", "-v", name, check=False)


@pytest.fixture
def connect(port):
    opened = []

    def factory(role="rag_indexer", application_name="test", autocommit=True):
        password = PASSWORDS["pg_vector_password" if role == "vector_user" else role + "_password"]
        conn = psycopg.connect(
            host="127.0.0.1",
            port=port,
            dbname="vector_db",
            user=role,
            password=password,
            application_name=application_name,
            autocommit=autocommit,
            connect_timeout=10,
        )
        opened.append(conn)
        return conn

    yield factory
    for conn in opened:
        conn.close()
