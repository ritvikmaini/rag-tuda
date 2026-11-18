"""A direct Postgres connection as the container's role: the build lock (a session advisory lock) and the DDL need one session."""

from typing import Any

import psycopg
from pydantic import SecretStr

from rag_core.settings import SettingsError
from rag_core.settings.sections import DatabaseSection


def _value(secret: SecretStr | None, key: str) -> str:
    if secret is None:
        raise SettingsError(f"database.{key}: not set (secret file or environment)")
    return secret.get_secret_value().strip()


def connect_direct(db: DatabaseSection, application_name: str) -> psycopg.Connection[Any]:
    return psycopg.connect(  # pyright: ignore[reportUnknownMemberType]
        host=db.host,
        port=db.port,
        dbname=_value(db.name, "name"),
        user=_value(db.user, "user"),
        password=_value(db.password, "password"),
        application_name=application_name,
        autocommit=True,
        connect_timeout=10,
    )


def credentials(db: DatabaseSection) -> tuple[str, str, str]:
    """(user, password, database name) from the section's secret files: the assistant's read pools (the index reads and the
    turn store) build their connection strings from them."""
    return _value(db.user, "user"), _value(db.password, "password"), _value(db.name, "name")
