"""The hard filter a turn ran with: chat.turns.applied_filter holds Filter.as_json() of the question's applied filter, SQL NULL
for a turn that ran without filtering (the switch off, a profile without roles). The next question starts from it, and a
variant runs on the filter of the turn it changes. The column is nullable without a default, so a row of an older image
stays valid. Plain statements without parameters, colons or percent signs (op.execute reads a colon as a bind)."""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

UPGRADE = (
    "ALTER TABLE chat.turns ADD COLUMN applied_filter jsonb",
    "COMMENT ON COLUMN chat.turns.applied_filter IS 'the question''s applied hard filter as JSON, NULL when the turn ran without filtering'",
)

DOWNGRADE = ("ALTER TABLE chat.turns DROP COLUMN applied_filter",)


def upgrade() -> None:
    for statement in UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
