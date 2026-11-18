"""The `.rag/` state directory: the API token, the three database role passwords and `state.json`; and the secret files of
the CI compose stack (`ci_secrets`).

Everything private is created with mode 0600 inside a 0700 directory; every writer is idempotent and never rotates a secret
that already exists."""

import argparse
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

ROLES = ("rag_owner", "rag_indexer", "rag_app")
DEFAULT_PORT = 8700
DEFAULT_STATE = {
    "mode": None,
    "class": None,
    "model": None,
    "ports": {"app": 8700, "embedding": 8701, "reranker": 8702, "postgres": 55432},
    "first_run_done": False,
}


@dataclass(frozen=True)
class State:
    dir: Path
    token: str
    passwords: dict[str, str]
    data: dict


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)  # the mkdir mode is cut by the umask and ignored for an existing directory


def _write_new(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    path.chmod(0o600)


def _secret(path: Path, make=lambda: secrets.token_urlsafe(32)) -> str:
    """The file's content, or `make()` written to it when it is missing or empty."""
    value = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    if not value:
        value = make()
        _write_new(path, value)
    path.chmod(0o600)
    return value


def ensure(root: str | Path) -> State:
    rag = Path(root) / ".rag"
    _private_dir(rag)
    _private_dir(rag / "secrets")
    token = _secret(rag / "token")
    passwords = {role: _secret(rag / "secrets" / role) for role in ROLES}
    state_file = rag / "state.json"
    if not state_file.is_file():
        _write_new(state_file, json.dumps(DEFAULT_STATE, indent=2) + "\n")
    state_file.chmod(0o600)
    return State(rag, token, passwords, json.loads(state_file.read_text(encoding="utf-8")))


def ci_secrets(root: str | Path) -> None:
    """The files deploy/compose.ci.yaml reads, under `root`: `secrets/*.txt`, `backend_token.txt` and `ci.env` (the
    backend token). The database name and the bootstrap role are fixed names; every other value is random."""
    root = Path(root)
    _private_dir(root / "secrets")

    def hex64() -> str:
        return secrets.token_hex(32)

    for name, value in {"pgdatabase": "vector_db", "vector_user": "vector_user"}.items():
        _secret(root / "secrets" / f"{name}.txt", lambda value=value: value)
    for name in ("pg_vector_password", "rag_indexer_password", "rag_app_password", "rag_migrator_password"):
        _secret(root / "secrets" / f"{name}.txt", hex64)
    backend = _secret(root / "backend_token.txt", hex64)
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
