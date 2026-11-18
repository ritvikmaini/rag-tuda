-- Database roles. Runs on an empty volume as the bootstrap superuser (POSTGRES_USER_FILE creates it and the database
-- before the init scripts), and can run once on an existing cluster through psql -f. The passwords come from the
-- secret files mounted at /run/secrets; psql reads them with backticks, so no value is written here, and the session
-- logs no statement, not even a failing one (log_min_error_statement), so none reaches the server log. Each role is
-- created only when missing, so a second run (a retry after a failed hand-over) does not fail on an existing role.
--   rag_indexer   indexer: owns the index tables; grants rag_app SELECT on each collection's two views and on
--                 index_generations itself: no default privilege
--   rag_app     backend: reads only what it is granted; owns the LangGraph checkpoint tables (03 creates them)
--   rag_migrator  the deploy step's migrations (the chat schema, created by 04)
\set ON_ERROR_STOP on
\set pw_indexer `cat /run/secrets/rag_indexer_password`
\set pw_app `cat /run/secrets/rag_app_password`
\set pw_migrator `cat /run/secrets/rag_migrator_password`
SELECT length(:'pw_indexer') >= 32 AND length(:'pw_app') >= 32 AND length(:'pw_migrator') >= 32 AS passwords_present \gset
\if :passwords_present
\else
DO $$ BEGIN RAISE EXCEPTION 'init-roles: a role password file under /run/secrets is missing or shorter than 32 characters'; END $$;
\endif
SET log_min_error_statement = 'panic';
SET log_statement = 'none';
SELECT NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_indexer') AS new_indexer,
       NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') AS new_app,
       NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_migrator') AS new_migrator \gset
\if :new_indexer
CREATE ROLE rag_indexer LOGIN PASSWORD :'pw_indexer';
\endif
\if :new_app
CREATE ROLE rag_app LOGIN PASSWORD :'pw_app';
\endif
\if :new_migrator
CREATE ROLE rag_migrator LOGIN PASSWORD :'pw_migrator';
\endif
GRANT CONNECT ON DATABASE :"DBNAME" TO rag_indexer, rag_app, rag_migrator;
GRANT USAGE, CREATE ON SCHEMA public TO rag_indexer;
GRANT USAGE ON SCHEMA public TO rag_app;
