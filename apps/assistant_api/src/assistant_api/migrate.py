"""python -m assistant_api.migrate: the assistant's Alembic migrations to head. A one-off container of the image runs it
(compose service migrate; the deploy's migrate step, the harness) as rag_migrator directly on Postgres: the DDL and
the version table want one session. The schema chat exists before
(deploy/pgvector/init/04-init-chat-schema.sql, the maintenance role).

`--pending` changes nothing: it prints `[Migrate] pending: <revisions>` and exits 3, or `[Migrate] nothing pending; at <heads>` and
exits 0 (the deploy backs the database up before a pending revision; any other exit is a failed check). A database at a revision
this image does not know (a rollback kept a newer schema) prints `[Migrate] the database is at <rev>, which this image does not
know` and exits 4: the deploy is refused."""

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import URL

MIGRATIONS = Path(__file__).with_name("migrations")
ROLE = "rag_migrator"
PENDING_EXIT = 3
UNKNOWN_EXIT = 4


def migrator_url(environ: Mapping[str, str], secrets_dir: str) -> URL:
    def secret(name: str) -> str:
        return Path(secrets_dir, name).read_text(encoding="utf-8").strip()

    return URL.create(
        "postgresql+psycopg",
        username=ROLE,
        password=secret("rag_migrator_password"),
        host=environ.get("RAG_MIGRATE_HOST", "pgvector"),
        port=int(environ.get("RAG_MIGRATE_PORT", "5432")),
        database=secret("pgdatabase"),
    )


def alembic_config(url: URL | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.attributes["url"] = url
    return cfg


def current(url: URL) -> tuple[str, ...]:
    """The revisions the database is at (empty before the first migration; the version table lives in chat)."""
    engine = create_engine(url, poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection, opts={"version_table_schema": "chat"}).get_current_heads()
    finally:
        engine.dispose()


def pending(cfg: Config, current: tuple[str, ...]) -> list[str]:
    """The revisions after the current ones up to head, oldest first (a linear chain; () = a fresh database)."""
    chain = [s.revision for s in reversed(list(ScriptDirectory.from_config(cfg).walk_revisions()))]
    return chain[max((chain.index(c) for c in current), default=-1) + 1 :]


def main(environ: Mapping[str, str] = os.environ, secrets_dir: str = "/run/secrets", argv: Sequence[str] = ()) -> int:
    url = migrator_url(environ, secrets_dir)
    cfg = alembic_config(url)
    if "--pending" in argv:
        at = current(url)
        known = {s.revision for s in ScriptDirectory.from_config(cfg).walk_revisions()}
        if unknown := [c for c in at if c not in known]:
            print("[Migrate] the database is at " + ", ".join(unknown) + ", which this image does not know")
            return UNKNOWN_EXIT
        todo = pending(cfg, at)
        if todo:
            print("[Migrate] pending: " + ", ".join(todo))
            return PENDING_EXIT
        print("[Migrate] nothing pending; at " + ", ".join(at))
        return 0
    command.upgrade(cfg, "head")
    print("[Migrate] the assistant's tables are at head " + ", ".join(ScriptDirectory.from_config(cfg).get_heads()))
    return 0


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv[1:]))
