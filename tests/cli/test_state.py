import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from rag_tuda import state

posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
COMPOSE = Path(__file__).resolve().parents[2] / "deploy" / "compose.ci.yaml"
# the generated files the compose file's top-level `secrets:` names, read from the file itself
CI_SECRET_FILES = re.findall(r"^  \w+: \{file: \./ci/\.generated/(\S+?)\}$", COMPOSE.read_text(encoding="utf-8"), re.M)
HEX64 = re.compile(r"[0-9a-f]{64}")


def mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_ensure_creates_layout(tmp_path):
    s = state.ensure(tmp_path)
    rag = tmp_path / ".rag"
    assert s.dir == rag
    assert (rag / "token").read_text() == s.token
    assert len(s.token) >= 43  # token_urlsafe(32)
    assert sorted(p.name for p in (rag / "secrets").iterdir()) == ["backend", "indexer", "migrate", "roles"]
    assert sorted(p.name for p in (rag / "secrets" / "roles").iterdir()) == sorted(
        [f"{name}.txt" for name in (*state.PASSWORDS, "pgdatabase", "vector_user")]
    )
    assert set(s.passwords) == set(state.PASSWORDS)
    for name in state.PASSWORDS:
        assert (rag / "secrets" / "roles" / f"{name}.txt").read_text() == s.passwords[name]
    assert len({*s.passwords.values(), s.token}) == 5
    assert (rag / "logs").is_dir() and (rag / "pids").is_dir()
    assert json.loads((rag / "state.json").read_text())["first_run_done"] is False


def read(path):
    return path.read_text(encoding="utf-8")


def test_each_native_process_gets_its_own_role_under_the_names_the_services_open(tmp_path):
    s = state.ensure(tmp_path)
    secrets = tmp_path / ".rag" / "secrets"
    assert sorted(p.name for p in (secrets / "backend").iterdir()) == [
        "backend_token",
        "pg_vector_password",
        "pgdatabase",
        "vector_user",
    ]
    assert sorted(p.name for p in (secrets / "indexer").iterdir()) == [
        "pg_vector_password",
        "pgdatabase",
        "vector_user",
    ]
    assert sorted(p.name for p in (secrets / "migrate").iterdir()) == ["pgdatabase", "rag_migrator_password"]
    assert read(secrets / "backend" / "vector_user") == "rag_app"
    assert read(secrets / "backend" / "pg_vector_password") == s.passwords["rag_app_password"]
    assert read(secrets / "backend" / "backend_token") == s.token
    assert read(secrets / "indexer" / "vector_user") == "rag_indexer"
    assert read(secrets / "indexer" / "pg_vector_password") == s.passwords["rag_indexer_password"]
    assert read(secrets / "migrate" / "rag_migrator_password") == s.passwords["rag_migrator_password"]
    for folder in ("backend", "indexer", "migrate"):
        assert read(secrets / folder / "pgdatabase") == read(secrets / "roles" / "pgdatabase.txt") == "vector_db"
    assert s.secrets_dir("indexer") == secrets / "indexer"


def test_the_role_names_are_the_ones_the_database_scripts_create():
    roles = Path(__file__).resolve().parents[2] / "deploy" / "pgvector" / "roles"
    assert {name: read(roles / f"{role}.txt") for name, role in state.SERVICE_ROLES.items()} == state.SERVICE_ROLES


@posix
def test_secret_file_modes(tmp_path):
    state.ensure(tmp_path)
    rag = tmp_path / ".rag"
    assert mode(rag) == 0o700
    for folder in [p for p in rag.rglob("*") if p.is_dir()]:
        assert mode(folder) == 0o700, folder
    files = [p for p in rag.rglob("*") if p.is_file()]
    assert len(files) >= 16
    roles = rag / "secrets" / "roles"
    for f in files:
        # the role files are the ones compose bind-mounts into non-root containers
        assert mode(f) == (0o644 if f.parent == roles else 0o600), f


@posix
def test_nothing_is_group_or_world_accessible(tmp_path):
    state.ensure(tmp_path)
    roles = tmp_path / ".rag" / "secrets" / "roles"
    for p in (tmp_path / ".rag").rglob("*"):
        if p.parent != roles:
            assert mode(p) & 0o077 == 0, p


@posix
def test_umask_cannot_loosen_modes(tmp_path):
    old = os.umask(0)
    try:
        state.ensure(tmp_path)
    finally:
        os.umask(old)
    assert mode(tmp_path / ".rag" / "token") == 0o600
    assert mode(tmp_path / ".rag") == 0o700


def test_idempotent_never_rotates(tmp_path):
    first = state.ensure(tmp_path)
    (tmp_path / ".rag" / "state.json").write_text(json.dumps({"mode": "native", "first_run_done": True}))
    second = state.ensure(tmp_path)
    assert second.token == first.token
    assert second.passwords == first.passwords
    assert second.data == {"mode": "native", "first_run_done": True}


def test_fills_only_what_is_missing(tmp_path):
    first = state.ensure(tmp_path)
    secrets = tmp_path / ".rag" / "secrets"
    (secrets / "roles" / "rag_app_password.txt").unlink()
    (secrets / "indexer" / "vector_user").unlink()
    second = state.ensure(tmp_path)
    assert second.token == first.token
    assert second.passwords["rag_indexer_password"] == first.passwords["rag_indexer_password"]
    assert second.passwords["rag_app_password"] != first.passwords["rag_app_password"]
    assert len(second.passwords["rag_app_password"]) >= 43
    assert read(secrets / "indexer" / "vector_user") == "rag_indexer"
    assert read(secrets / "backend" / "pg_vector_password") == second.passwords["rag_app_password"]  # the copy follows


def test_a_copy_follows_its_source(tmp_path):
    state.ensure(tmp_path)
    (tmp_path / ".rag" / "secrets" / "backend" / "backend_token").write_text("stale")
    s = state.ensure(tmp_path)
    assert read(tmp_path / ".rag" / "secrets" / "backend" / "backend_token") == s.token


@posix
def test_loose_existing_modes_are_tightened(tmp_path):
    state.ensure(tmp_path)
    (tmp_path / ".rag" / "token").chmod(0o644)
    (tmp_path / ".rag").chmod(0o755)
    state.ensure(tmp_path)
    assert mode(tmp_path / ".rag" / "token") == 0o600
    assert mode(tmp_path / ".rag") == 0o700


def test_empty_secret_file_is_replaced(tmp_path):
    state.ensure(tmp_path)
    (tmp_path / ".rag" / "token").write_text("")
    assert len(state.ensure(tmp_path).token) >= 43


def test_page_url(tmp_path):
    s = state.ensure(tmp_path)
    assert state.page_url(s) == f"http://127.0.0.1:8700/#{s.token}"
    assert state.page_url(s, port=9000) == f"http://127.0.0.1:9000/#{s.token}"


def test_init_command_prints_no_secret(tmp_path, capsys):
    assert state.main(["init", "--root", str(tmp_path)]) == 0
    token = (tmp_path / ".rag" / "token").read_text()
    out = capsys.readouterr()
    assert token not in out.out + out.err


def test_state_init_writes_ci_secrets(tmp_path):
    assert len(CI_SECRET_FILES) == 7, CI_SECRET_FILES  # compose.ci.yaml names seven generated secrets
    assert state.main(["init", "--root", str(tmp_path)]) == 0
    values = {name: (tmp_path / name).read_text(encoding="utf-8") for name in CI_SECRET_FILES}
    assert all(values.values()), values.keys()
    if sys.platform != "win32":
        for name in CI_SECRET_FILES:
            assert mode(tmp_path / name) == 0o644, name
        assert mode(tmp_path / "ci.env") == 0o600
        assert mode(tmp_path / "secrets") == 0o700
    # database name and bootstrap role are fixed (compose and the init scripts name them); every other value is random
    assert values["secrets/pgdatabase.txt"] == "vector_db"
    assert values["secrets/vector_user.txt"] == "vector_user"
    passwords = [values[f"secrets/{n}_password.txt"] for n in ("pg_vector", "rag_indexer", "rag_app", "rag_migrator")]
    assert all(HEX64.fullmatch(p) for p in passwords)
    assert len({*passwords, values["secrets/backend_token.txt"]}) == 5
    env = dict(line.split("=", 1) for line in (tmp_path / "ci.env").read_text(encoding="utf-8").splitlines())
    assert env["BACKEND_REFRESH_TOKEN"] == values["secrets/backend_token.txt"]
    assert list(env) == ["BACKEND_REFRESH_TOKEN"]
    # a second run never rotates what exists
    state.main(["init", "--root", str(tmp_path)])
    assert {name: (tmp_path / name).read_text(encoding="utf-8") for name in CI_SECRET_FILES} == values


def test_the_repo_ignores_every_file_init_writes(tmp_path):
    """A run at the repo root must never leave a secret that `git add -A` would stage."""
    root = Path(__file__).resolve().parents[2]
    work = tmp_path / "repo"
    work.mkdir()
    (work / ".gitignore").write_text((root / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    state.main(["init", "--root", str(work)])
    written = [
        p for p in work.rglob("*") if p.is_file() and ".git" not in p.relative_to(work).parts and p.name != ".gitignore"
    ]
    assert len(written) >= 25
    for path in written:
        rel = path.relative_to(work).as_posix()
        assert subprocess.run(["git", "check-ignore", "-q", rel], cwd=work).returncode == 0, rel
