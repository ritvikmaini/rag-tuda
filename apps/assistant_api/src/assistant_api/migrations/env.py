"""Alembic's environment for the assistant's tables: online only, one connection as rag_migrator
(assistant_api.migrate passes the URL), the version table in the schema chat (rag_migrator may create nothing in public)."""

from alembic import context
from sqlalchemy import create_engine, pool

SCHEMA = "chat"

if context.is_offline_mode():
    raise SystemExit("assistant_api.migrations: the offline (SQL script) mode is not used")
engine = create_engine(context.config.attributes["url"], poolclass=pool.NullPool)
with engine.connect() as connection:
    context.configure(connection=connection, version_table_schema=SCHEMA)
    with context.begin_transaction():
        context.run_migrations()
