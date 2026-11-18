"""rag_core.adapters.db: a direct connection as the container's role, built from the settings."""

from typing import Any

import psycopg
import pytest
from pydantic import SecretStr

from rag_core.adapters.db import connect_direct, credentials
from rag_core.settings import SettingsError
from rag_core.settings.sections import DatabaseSection


def section(**kw):
    values: dict[str, Any] = {
        "name": SecretStr("example\n"),
        "user": SecretStr("rag_indexer"),
        "password": SecretStr("pw\n"),
    }
    values.update(kw)
    return DatabaseSection(host="pgvector", port=5432, **values)


def test_connect_direct_reads_the_role_from_the_settings_and_connects_directly(monkeypatch):
    calls = []
    monkeypatch.setattr(psycopg, "connect", lambda **kw: calls.append(kw) or "conn")
    assert connect_direct(section(), "indexer") == "conn"
    assert calls == [
        {
            "host": "pgvector",
            "port": 5432,
            "dbname": "example",
            "user": "rag_indexer",
            "password": "pw",
            "application_name": "indexer",
            "autocommit": True,
            "connect_timeout": 10,
        }
    ]


def test_a_missing_secret_is_named():
    with pytest.raises(SettingsError, match=r"database\.password: not set"):
        connect_direct(section(password=None), "indexer")


def test_credentials_are_the_settings_secrets_stripped():
    assert credentials(section()) == ("rag_indexer", "pw", "example")


def test_credentials_name_the_missing_key():
    with pytest.raises(SettingsError, match="database.password"):
        credentials(section(password=None))
