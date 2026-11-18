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
ROLES = ("rag_owner", "rag_indexer", "rag_app")
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
    assert sorted(p.name for p in (rag / "secrets").iterdir()) == sorted(ROLES)
    assert set(s.passwords) == set(ROLES)
    for role in ROLES:
        assert (rag / "secrets" / role).read_text() == s.passwords[role]
    assert len(set(s.passwords.values()) | {s.token}) == 4
    assert json.loads((rag / "state.json").read_text())["first_run_done"] is False


@posix
def test_modes(tmp_path):
    state.ensure(tmp_path)
    rag = tmp_path / ".rag"
    assert mode(rag) == 0o700
    assert mode(rag / "secrets") == 0o700
    for f in [rag / "token", rag / "state.json", *(rag / "secrets").iterdir()]:
        assert mode(f) == 0o600, f


@posix
def test_nothing_is_group_or_world_accessible(tmp_path):
    state.ensure(tmp_path)
    for p in (tmp_path / ".rag").rglob("*"):
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
    (tmp_path / ".rag" / "secrets" / "rag_app").unlink()
    second = state.ensure(tmp_path)
    assert second.token == first.token
    assert second.passwords["rag_owner"] == first.passwords["rag_owner"]
    assert second.passwords["rag_app"] != first.passwords["rag_app"]
    assert len(second.passwords["rag_app"]) >= 43


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
        for name in [*CI_SECRET_FILES, "ci.env"]:
            assert mode(tmp_path / name) == 0o600, name
        assert mode(tmp_path / "secrets") == 0o700
    # database name and bootstrap role are fixed (compose and the init scripts name them); every other value is random
    assert values["secrets/pgdatabase.txt"] == "vector_db"
    assert values["secrets/vector_user.txt"] == "vector_user"
    passwords = [values[f"secrets/{n}_password.txt"] for n in ("pg_vector", "rag_indexer", "rag_app", "rag_migrator")]
    assert all(HEX64.fullmatch(p) for p in passwords)
    assert len({*passwords, values["backend_token.txt"]}) == 5
    env = dict(line.split("=", 1) for line in (tmp_path / "ci.env").read_text(encoding="utf-8").splitlines())
    assert env["BACKEND_REFRESH_TOKEN"] == values["backend_token.txt"]
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
    assert len(written) >= 12
    for path in written:
        rel = path.relative_to(work).as_posix()
        assert subprocess.run(["git", "check-ignore", "-q", rel], cwd=work).returncode == 0, rel
