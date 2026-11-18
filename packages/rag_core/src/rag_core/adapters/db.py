"""A direct Postgres connection as the container's role: the build lock (a session advisory lock) and the DDL need one session."""

from typing import Any

import psycopg

from rag_core.settings import SettingsError, secret_file_name
from rag_core.settings.sections import DatabaseSection


def _value(db: DatabaseSection, key: str) -> str:
    secret = getattr(db, key)
    if secret is None:
        raise SettingsError(
            f"database.{key}: not set (the secret file {secret_file_name(DatabaseSection, key)} in the secrets folder, "
            f"RAG_SECRETS_DIR or /run/secrets, or RAG__DATABASE__{key.upper()})"
        )
    return secret.get_secret_value().strip()


def connect_direct(db: DatabaseSection, application_name: str) -> psycopg.Connection[Any]:
    return psycopg.connect(  # pyright: ignore[reportUnknownMemberType]
        host=db.host,
        port=db.port,
        dbname=_value(db, "name"),
        user=_value(db, "user"),
        password=_value(db, "password"),
        application_name=application_name,
        autocommit=True,
        connect_timeout=10,
    )


def credentials(db: DatabaseSection) -> tuple[str, str, str]:
    """(user, password, database name) from the section's secret files: the assistant's read pools (the index reads and the
    turn store) build their connection strings from them."""
    return _value(db, "user"), _value(db, "password"), _value(db, "name")
