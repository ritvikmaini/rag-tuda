"""The turn tables: the turns, their answer versions and their context, in the schema chat (owned by rag_migrator;
deploy/pgvector/init/04-init-chat-schema.sql). rag_app, the assistant's role, gets USAGE and SELECT, INSERT, UPDATE,
DELETE through default privileges set before the tables exist; alembic_version (created before this runs) gets nothing.
chat.turns.error holds `<code>` or `<code>:<ExceptionClass>` of a failed turn, never a message. Plain statements without
parameters, colons or percent signs (op.execute reads a colon as a bind)."""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

UPGRADE = (
    "GRANT USAGE ON SCHEMA chat TO rag_app",
    "ALTER DEFAULT PRIVILEGES FOR ROLE rag_migrator IN SCHEMA chat GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag_app",
    """CREATE TABLE chat.turns (
  turn_id uuid PRIMARY KEY,
  conversation_id text NOT NULL,
  principal text NOT NULL DEFAULT 'anonymous',
  question text NOT NULL,
  standalone_question text,
  intent text,
  answer text,
  answer_shape text,
  retrieved_section_ids text[],
  cited_section_ids text[],
  prompts jsonb,
  collection text,
  index_generation int,
  config_hash text,
  stage_ms jsonb,
  usage jsonb,
  outcome text NOT NULL,
  error text,
  superseded_by uuid REFERENCES chat.turns (turn_id),
  created_at timestamptz NOT NULL DEFAULT now()
)""",
    "CREATE INDEX turns_conversation ON chat.turns (conversation_id, created_at)",
    "COMMENT ON COLUMN chat.turns.conversation_id IS 'any client id, p-<uuid> from the v2 page, c-<uuid> from older tabs, the "
    "evals runner''s probe ids'",
    "COMMENT ON COLUMN chat.turns.answer IS 'the original answer; the versions are in turn_versions'",
    """CREATE TABLE chat.turn_versions (
  turn_id uuid REFERENCES chat.turns,
  name text,
  text text NOT NULL,
  shape text,
  active boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (turn_id, name)
)""",
    "CREATE TABLE chat.turn_context (turn_id uuid PRIMARY KEY REFERENCES chat.turns, sections jsonb NOT NULL)",
)

DOWNGRADE = (
    "DROP TABLE chat.turn_context, chat.turn_versions, chat.turns",
    "ALTER DEFAULT PRIVILEGES FOR ROLE rag_migrator IN SCHEMA chat REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM rag_app",
    "REVOKE USAGE ON SCHEMA chat FROM rag_app",
)


def upgrade() -> None:
    for statement in UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
