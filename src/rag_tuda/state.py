"""The `.rag/` state directory: the API token, the database passwords, the secret files of each native process, the pid files
and logs, and `state.json`; and the secret files of the CI compose stack (`ci_secrets`).

Everything private is created with mode 0600 inside a 0700 directory, except the files compose bind-mounts into
containers that do not run as root (the role passwords, the CI secrets): they are 0644 inside their 0700 directory, since
a 0600 file owned by the host user is unreadable to the container user on Linux. Every writer is idempotent and never
rotates a secret that already exists. The passwords live once, in `secrets/roles/` under the names compose reads (`<name>.txt`); each native
process gets its own folder of copies under the names the service opens (RAG_SECRETS_DIR), the way compose maps a service's
role onto them."""

import argparse
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

PASSWORDS = ("pg_vector_password", "rag_indexer_password", "rag_app_password", "rag_migrator_password")
DATABASE = "vector_db"  # the database and the bootstrap role are fixed names; the init scripts and compose name them
BOOTSTRAP_ROLE = "vector_user"
SERVICE_ROLES = {"backend": "rag_app", "indexer": "rag_indexer"}  # deploy/pgvector/roles/*.txt
DEFAULT_PORT = 8700
DEFAULT_STATE = {
    "mode": None,
    "card": None,
    "profile": None,
    "model": None,
    "llm_url": None,
    "ports": {"app": 8700, "embedding": 8701, "reranker": 8702, "postgres": 55432},
    "first_run_done": False,
}


@dataclass(frozen=True)
class State:
    dir: Path
    token: str
    passwords: dict[str, str]
    data: dict

    def secrets_dir(self, name: str) -> Path:
        """`roles` (the compose files), or one native process: backend, indexer or migrate."""
        return self.dir / "secrets" / name


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)  # the mkdir mode is cut by the umask and ignored for an existing directory


MOUNTED = 0o644  # the mode of a file that a container reads; its folder is 0700


def _write_new(path: Path, text: str, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    path.chmod(mode)


def write_private(path: Path, text: str) -> None:
    """The file with this content and mode 0600, in a 0700 folder; the content is replaced."""
    _private_dir(path.parent)
    _write_new(path, text)


def _secret(path: Path, make=lambda: secrets.token_urlsafe(32), mode: int = 0o600) -> str:
    """The file's content, or `make()` written to it when it is missing or empty."""
    value = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    if not value:
        value = make()
        _write_new(path, value, mode)
    path.chmod(mode)
    return value


def _copy(path: Path, value: str) -> None:
    """A derived file: written when it is missing or differs from its source."""
    if not path.is_file() or path.read_text(encoding="utf-8") != value:
        _write_new(path, value)
    path.chmod(0o600)


def ensure(root: str | Path) -> State:
    rag = Path(root) / ".rag"
    for folder in ("", "secrets", "logs", "pids"):
        _private_dir(rag / folder)
    token = _secret(rag / "token")
    roles = rag / "secrets" / "roles"
    _private_dir(roles)
    passwords = {name: _secret(roles / f"{name}.txt", mode=MOUNTED) for name in PASSWORDS}
    _secret(roles / "pgdatabase.txt", lambda: DATABASE, MOUNTED)
    _secret(roles / "vector_user.txt", lambda: BOOTSTRAP_ROLE, MOUNTED)
    # one folder per native process: backend and indexer open the same file names, each holding its own role
    for service, role in SERVICE_ROLES.items():
        folder = rag / "secrets" / service
        _private_dir(folder)
        _copy(folder / "pgdatabase", DATABASE)
        _copy(folder / "vector_user", role)
        _copy(folder / "pg_vector_password", passwords[f"{role}_password"])
    _copy(rag / "secrets" / "backend" / "backend_token", token)
    migrate = rag / "secrets" / "migrate"
    _private_dir(migrate)
    _copy(migrate / "pgdatabase", DATABASE)
    _copy(migrate / "rag_migrator_password", passwords["rag_migrator_password"])
    state_file = rag / "state.json"
    if not state_file.is_file():
        _write_new(state_file, json.dumps(DEFAULT_STATE, indent=2) + "\n")
    state_file.chmod(0o600)
    return State(rag, token, passwords, json.loads(state_file.read_text(encoding="utf-8")))


def ci_secrets(root: str | Path) -> None:
    """The files deploy/compose.ci.yaml reads, under `root`: `secrets/*.txt` (the compose secrets, 0644 in the 0700
    folder, the backend token among them) and `ci.env` (the backend token, 0600). The database name and the bootstrap role are fixed names; every other value is random."""
    root = Path(root)
    _private_dir(root / "secrets")

    def hex64() -> str:
        return secrets.token_hex(32)

    for name, value in {"pgdatabase": "vector_db", "vector_user": "vector_user"}.items():
        _secret(root / "secrets" / f"{name}.txt", lambda value=value: value, MOUNTED)
    for name in ("pg_vector_password", "rag_indexer_password", "rag_app_password", "rag_migrator_password"):
        _secret(root / "secrets" / f"{name}.txt", hex64, MOUNTED)
    backend = _secret(root / "secrets" / "backend_token.txt", hex64, MOUNTED)
    env = root / "ci.env"
    if not env.is_file() or not env.read_text(encoding="utf-8").strip():
        _write_new(env, f"BACKEND_REFRESH_TOKEN={backend}\n")
    env.chmod(0o600)


def page_url(state: State, port: int = DEFAULT_PORT) -> str:
    return f"http://127.0.0.1:{port}/#{state.token}"


def main(argv: list[str] | None = None) -> int:
    """`python -m rag_tuda.state init --root DIR` creates the state directory and the CI secret files (for scripts and the
    harness)."""
    parser = argparse.ArgumentParser(prog="python -m rag_tuda.state")
    parser.add_argument("command", choices=["init"])
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    ensure(args.root)
    ci_secrets(args.root)
    print(f"state ready in {Path(args.root) / '.rag'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
