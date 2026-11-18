-- The assistant's own schema. Owned by rag_migrator, which creates the
-- tables with Alembic (python -m assistant_api.migrate); rag_app gets USAGE and its table privileges from migration 0001.
-- Runs on an empty volume as the bootstrap superuser after 01-init-roles.sql (which created rag_migrator), and on an existing
-- cluster by psql -f over the container socket as the maintenance role. Idempotent.
\set ON_ERROR_STOP on
CREATE SCHEMA IF NOT EXISTS chat AUTHORIZATION rag_migrator;
