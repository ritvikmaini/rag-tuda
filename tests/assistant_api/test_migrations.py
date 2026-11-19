"""The assistant's own tables: one Alembic head, three tables in the schema chat, rag_app's privileges set before the tables
exist, the version table in chat. The Docker-gated test reads the harness database after its migrate one-shot ran."""

import importlib.util
import os
import re
from pathlib import Path

import pytest
from alembic.script import ScriptDirectory
from sqlalchemy.engine import URL

from assistant_api import migrate

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "deploy" / "ci" / ".generated"


def first_migration():
    path = migrate.MIGRATIONS / "versions" / "0001_initial.py"
    spec = importlib.util.spec_from_file_location("m0001", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def second_migration():
    path = migrate.MIGRATIONS / "versions" / "0002_applied_filter.py"
    spec = importlib.util.spec_from_file_location("m0002", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migrations_chain_and_the_turn_tables_schema() -> None:
    """A linear chain of two files with one head; the first creates the turn tables (the nullable error column is part of
    its CREATE), the second adds the applied filter."""
    assert sorted(p.name for p in (migrate.MIGRATIONS / "versions").glob("*.py")) == [
        "0001_initial.py",
        "0002_applied_filter.py",
    ]
    assert ScriptDirectory.from_config(migrate.alembic_config()).get_heads() == ["0002"]
    m = first_migration()
    assert (m.revision, m.down_revision) == ("0001", None)
    turns = next(" ".join(s.split()) for s in m.UPGRADE if s.startswith("CREATE TABLE chat.turns "))
    assert "outcome text NOT NULL, error text, superseded_by" in turns  # nullable, no default
    assert m.DOWNGRADE[0] == "DROP TABLE chat.turn_context, chat.turn_versions, chat.turns"


def test_the_second_migration_adds_a_nullable_applied_filter_column_and_drops_it_again() -> None:
    m = second_migration()
    assert (m.revision, m.down_revision) == ("0002", "0001")
    assert [" ".join(s.split()) for s in m.UPGRADE][0] == "ALTER TABLE chat.turns ADD COLUMN applied_filter jsonb"
    assert m.DOWNGRADE == ("ALTER TABLE chat.turns DROP COLUMN applied_filter",)
    sql = "\n".join(m.UPGRADE + m.DOWNGRADE)
    assert ":" not in sql.replace("::", "") and "%" not in sql  # op.execute reads a colon as a bind


def test_pending_lists_the_revisions_after_the_current_one() -> None:
    cfg = migrate.alembic_config()
    assert migrate.pending(cfg, ()) == ["0001", "0002"]
    assert migrate.pending(cfg, ("0001",)) == ["0002"]
    assert migrate.pending(cfg, ("0002",)) == []


def test_main_pending_exit_codes(monkeypatch, tmp_path: Path, capsys) -> None:
    (tmp_path / "rag_migrator_password").write_text("p\n", encoding="utf-8")
    (tmp_path / "pgdatabase").write_text("vector_db\n", encoding="utf-8")

    def main(*argv):
        return migrate.main({}, str(tmp_path), argv)

    monkeypatch.setattr(migrate, "current", lambda url: ())
    assert main("--pending") == 3
    assert capsys.readouterr().out == "[Migrate] pending: 0001, 0002\n"
    monkeypatch.setattr(migrate, "current", lambda url: ("0001",))
    assert main("--pending") == 3
    assert capsys.readouterr().out == "[Migrate] pending: 0002\n"
    monkeypatch.setattr(migrate, "current", lambda url: ("0002",))
    assert main("--pending") == 0
    assert capsys.readouterr().out == "[Migrate] nothing pending; at 0002\n"
    monkeypatch.setattr(
        migrate, "current", lambda url: ("0003",)
    )  # a rollback left a newer schema than this image knows
    assert main("--pending") == 4
    assert capsys.readouterr().out == "[Migrate] the database is at 0003, which this image does not know\n"


def test_the_first_migration_creates_spec_6_6_and_grants_before_it_creates() -> None:
    sql = "\n".join(first_migration().UPGRADE)
    for table in ("chat.turns", "chat.turn_versions", "chat.turn_context"):
        assert "CREATE TABLE " + table in sql
    for column in (
        "turn_id uuid PRIMARY KEY",
        "conversation_id text NOT NULL",
        "principal text NOT NULL DEFAULT 'anonymous'",
        "retrieved_section_ids text[]",
        "cited_section_ids text[]",
        "prompts jsonb",
        "index_generation int",
        "stage_ms jsonb",
        "usage jsonb",
        "outcome text NOT NULL",
        "superseded_by uuid REFERENCES chat.turns (turn_id)",
        "created_at timestamptz NOT NULL DEFAULT now()",
        "PRIMARY KEY (turn_id, name)",
        "sections jsonb NOT NULL",
        "CREATE INDEX turns_conversation ON chat.turns (conversation_id, created_at)",
    ):
        assert column in " ".join(sql.split()), column
    grants = sql.index(
        "ALTER DEFAULT PRIVILEGES FOR ROLE rag_migrator IN SCHEMA chat GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag_app"
    )
    assert grants < sql.index("CREATE TABLE") and "GRANT USAGE ON SCHEMA chat TO rag_app" in sql
    assert "public." not in sql and "PUBLIC" not in sql and "%" not in sql


def test_the_url_is_rag_migrator_direct_to_pgvector(tmp_path: Path) -> None:
    (tmp_path / "rag_migrator_password").write_text("p@ss:w%rd\n", encoding="utf-8")
    (tmp_path / "pgdatabase").write_text("vector_db\n", encoding="utf-8")
    url = migrate.migrator_url({}, str(tmp_path))
    assert (url.drivername, url.username, url.password, url.host, url.port, url.database) == (
        "postgresql+psycopg",
        "rag_migrator",
        "p@ss:w%rd",
        "pgvector",
        5432,
        "vector_db",
    )
    assert (
        migrate.migrator_url({"RAG_MIGRATE_HOST": "127.0.0.1", "RAG_MIGRATE_PORT": "25432"}, str(tmp_path)).port
        == 25432
    )


@pytest.mark.skipif(not (os.environ.get("RUN_DOCKER_TESTS") and (GEN / "ci.env").exists()), reason="needs the harness")
def test_the_harness_database_has_the_tables_with_least_privilege() -> None:
    import psycopg
    from rag_testing import (
        harness,
    )  # DB_PORT: 25432, or the worktree's Docker-picked port

    def secret(name: str) -> str:
        return (GEN / "secrets" / (name + ".txt")).read_text(encoding="utf-8").strip()

    common = {"host": "127.0.0.1", "port": harness.DB_PORT, "dbname": secret("pgdatabase"), "autocommit": True}
    with psycopg.connect(user="rag_migrator", password=secret("rag_migrator_password"), **common) as conn:
        assert conn.execute("SELECT version_num FROM chat.alembic_version").fetchall() == [("0002",)]
        owners = dict(conn.execute("SELECT tablename, tableowner FROM pg_tables WHERE schemaname = 'chat'").fetchall())
        assert owners == {t: "rag_migrator" for t in ("alembic_version", "turns", "turn_versions", "turn_context")}
        grants = set(
            conn.execute(
                "SELECT table_name, grantee, privilege_type FROM information_schema.role_table_grants "
                "WHERE table_schema = 'chat' AND grantee <> 'rag_migrator'"
            ).fetchall()
        )
        assert conn.execute(
            "SELECT data_type, is_nullable, column_default FROM information_schema.columns "
            "WHERE table_schema = 'chat' AND table_name = 'turns' AND column_name = 'error'"
        ).fetchall() == [("text", "YES", None)]
        assert conn.execute(
            "SELECT data_type, is_nullable, column_default FROM information_schema.columns "
            "WHERE table_schema = 'chat' AND table_name = 'turns' AND column_name = 'applied_filter'"
        ).fetchall() == [("jsonb", "YES", None)]
        assert grants == {
            (t, "rag_app", p)
            for t in ("turns", "turn_versions", "turn_context")
            for p in ("SELECT", "INSERT", "UPDATE", "DELETE")
        }
    with psycopg.connect(user="rag_app", password=secret("rag_app_password"), **common) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("SELECT 1 FROM chat.alembic_version")


# --- additive only: a migration runs before the previous backend stops, so every upgrade
# only adds. Test code: the migration's `upgrade()` runs against a recorder in place of alembic's `op`.

ADDITIVE_OPS = {
    "execute",
    "add_column",
    "create_table",
    "create_index",
    "create_foreign_key",
    "create_primary_key",
}
NOT_ADDITIVE_SQL = (
    re.compile(r"\bDROP\b(?!\s+(NOT\s+NULL|DEFAULT)\b)", re.IGNORECASE),
    re.compile(r"\bRENAME\b", re.IGNORECASE),
    re.compile(r"\bALTER\s+(COLUMN\s+)?(\w+|\"[^\"]+\")\s+(SET\s+DATA\s+)?TYPE\b", re.IGNORECASE),
    # the running backend keeps inserting while the new schema is live: no new NOT NULL, CHECK or UNIQUE on an existing column
    re.compile(r"\bSET\s+NOT\s+NULL\b", re.IGNORECASE),
    re.compile(r"\bADD\s+(CONSTRAINT\s+(\w+|\"[^\"]+\")\s+)?(CHECK|UNIQUE)\b", re.IGNORECASE),
)
ADD_COLUMN = re.compile(r"\bADD\s+(?:COLUMN\s+)?(?:IF\s+NOT\s+EXISTS\s+)?([^,;]*)", re.IGNORECASE)
# known limit: a CHECK or UNIQUE written inside an ADD COLUMN clause, and a NOT NULL added by a helper, are not seen


def upgrade_calls(path: Path) -> list[tuple[str, tuple, dict]]:
    """What upgrade() asks of alembic's op, in order: (name, args, kwargs)."""
    calls: list[tuple[str, tuple, dict]] = []

    class Recorder:
        def __getattr__(self, name):
            return lambda *args, **kwargs: calls.append((name, args, kwargs))

    spec = importlib.util.spec_from_file_location("migration_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.op = Recorder()
    module.upgrade()
    return calls


def not_additive(calls: list[tuple[str, tuple, dict]]) -> list[str]:
    """One finding per call that is not an addition; an op not listed is refused by name (unknown = not checked = no)."""
    found = []
    sql = [" ".join(str(args[0]).split()) for name, args, _ in calls if name == "execute"]
    created = {
        m.group(1).lower()
        for st in sql
        for m in re.finditer(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.\"]+)", st, re.I)
    }
    for name, args, _ in calls:
        if name not in ADDITIVE_OPS:
            found.append("op." + name)
        elif name == "execute":
            statement = " ".join(str(args[0]).split())
            unique_index = re.search(r"\bCREATE\s+UNIQUE\s+INDEX\b.*?\bON\s+([\w.\"]+)", statement, re.I)
            if (
                any(rule.search(statement) for rule in NOT_ADDITIVE_SQL)
                or any(
                    re.search(r"\bNOT\s+NULL\b", c.group(1), re.I) and not re.search(r"\bDEFAULT\b", c.group(1), re.I)
                    for c in ADD_COLUMN.finditer(statement)
                    if not c.group(1).upper().startswith(("CONSTRAINT", "CHECK", "UNIQUE"))
                )
                or (unique_index and unique_index.group(1).lower() not in created)
            ):
                found.append(statement[:80])
    return found


def write_migration(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "0001_sample.py"
    path.write_text("from alembic import op\n\n\ndef upgrade():\n    " + body + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "body, finding",
    [
        ('op.execute("ALTER TABLE chat.turns DROP COLUMN intent")', "ALTER TABLE chat.turns DROP COLUMN intent"),
        ('op.execute("ALTER TABLE chat.turns DROP intent")', "ALTER TABLE chat.turns DROP intent"),
        ('op.execute("DROP TABLE chat.turn_context")', "DROP TABLE chat.turn_context"),
        (
            'op.execute("ALTER TABLE chat.turns RENAME COLUMN intent TO kind")',
            "ALTER TABLE chat.turns RENAME COLUMN intent TO kind",
        ),
        (
            'op.execute("ALTER TABLE chat.turns ALTER COLUMN intent TYPE int USING 0")',
            "ALTER TABLE chat.turns ALTER COLUMN intent TYPE int USING 0",
        ),
        (
            'op.execute("ALTER TABLE chat.turns ALTER COLUMN \\"intent\\" TYPE int")',
            'ALTER TABLE chat.turns ALTER COLUMN "intent" TYPE int',
        ),
        (
            'op.execute("ALTER TABLE chat.turns ALTER COLUMN intent SET NOT NULL")',
            "ALTER TABLE chat.turns ALTER COLUMN intent SET NOT NULL",
        ),
        (
            'op.execute("ALTER TABLE chat.turns ADD COLUMN error text NOT NULL")',
            "ALTER TABLE chat.turns ADD COLUMN error text NOT NULL",
        ),
        (
            'op.execute("ALTER TABLE chat.turns ADD CONSTRAINT c CHECK (length(intent) < 9)")',
            "ALTER TABLE chat.turns ADD CONSTRAINT c CHECK (length(intent) < 9)",
        ),
        (
            'op.execute("ALTER TABLE chat.turns ADD UNIQUE (conversation_id)")',
            "ALTER TABLE chat.turns ADD UNIQUE (conversation_id)",
        ),
        (
            'op.execute("CREATE UNIQUE INDEX t_u ON chat.turns (conversation_id)")',
            "CREATE UNIQUE INDEX t_u ON chat.turns (conversation_id)",
        ),
        ('op.create_unique_constraint("u", "turns", ["x"])', "op.create_unique_constraint"),
        ('op.create_check_constraint("c", "turns", "x > 0")', "op.create_check_constraint"),
        ('op.drop_column("turns", "intent")', "op.drop_column"),
        ('op.alter_column("turns", "intent", new_column_name="kind")', "op.alter_column"),
        ('op.rename_table("turns", "t")', "op.rename_table"),
    ],
)
def test_the_additive_check_refuses_drops_renames_and_retypes(tmp_path: Path, body: str, finding: str) -> None:
    assert not_additive(upgrade_calls(write_migration(tmp_path, body))) == [finding]


def test_the_additive_check_accepts_an_added_nullable_column(tmp_path: Path) -> None:
    body = (
        'op.execute("ALTER TABLE chat.turns ADD COLUMN error text")\n'
        '    op.execute("ALTER TABLE chat.turns ALTER COLUMN intent DROP NOT NULL")\n'
        '    op.add_column("turns", None)'
    )
    assert not_additive(upgrade_calls(write_migration(tmp_path, body))) == []


def test_the_additive_check_accepts_new_tables_defaults_and_plain_indexes(tmp_path: Path) -> None:
    body = (
        'op.execute("CREATE TABLE chat.notes (id int NOT NULL, body text NOT NULL, UNIQUE (id))")\n'
        '    op.execute("CREATE UNIQUE INDEX notes_id ON chat.notes (id)")\n'
        '    op.execute("ALTER TABLE chat.turns ADD COLUMN IF NOT EXISTS flag boolean NOT NULL DEFAULT false")\n'
        "    op.execute(\"ALTER TABLE chat.turns ADD COLUMN kind text DEFAULT 'a' NOT NULL, ADD COLUMN note text\")\n"
        '    op.execute("CREATE INDEX turns_kind ON chat.turns (kind)")'
    )
    assert not_additive(upgrade_calls(write_migration(tmp_path, body))) == []


def test_every_upgrade_only_adds() -> None:
    versions = sorted((migrate.MIGRATIONS / "versions").glob("*.py"))
    assert versions
    for path in versions:
        assert not_additive(upgrade_calls(path)) == [], path.name


@pytest.mark.skipif(not (os.environ.get("RUN_DOCKER_TESTS") and (GEN / "ci.env").exists()), reason="needs the harness")
def test_the_migrations_go_down_and_up_and_pending_sees_them() -> None:
    import psycopg
    from alembic import command
    from rag_testing import harness

    def secret(name: str) -> str:
        return (GEN / "secrets" / (name + ".txt")).read_text(encoding="utf-8").strip()

    common = {"host": "127.0.0.1", "port": harness.DB_PORT, "dbname": secret("pgdatabase")}
    password = secret("rag_migrator_password")
    url = URL.create(
        "postgresql+psycopg",
        username=migrate.ROLE,
        password=password,
        host=common["host"],
        port=common["port"],
        database=common["dbname"],
    )
    cfg = migrate.alembic_config(url)
    turns = "SELECT to_regclass('chat.turns')"
    with psycopg.connect(user=migrate.ROLE, password=password, autocommit=True, **common) as conn:
        try:
            command.downgrade(cfg, "base")
            assert conn.execute(turns).fetchone() == (None,)
            assert migrate.current(url) == () and migrate.pending(cfg, migrate.current(url)) == ["0001", "0002"]
        finally:
            command.upgrade(cfg, "head")
        assert conn.execute(turns).fetchone() == ("chat.turns",)
        assert migrate.current(url) == ("0002",)
        assert conn.execute(
            "SELECT to_regclass('chat.turns'), count(*) FROM information_schema.columns WHERE table_schema = 'chat' AND table_name = 'turns' AND column_name = 'applied_filter'"
        ).fetchone() == ("chat.turns", 1)


def test_the_secrets_folder_defaults_to_the_environments(tmp_path: Path) -> None:
    (tmp_path / "rag_migrator_password").write_text("p\n", encoding="utf-8")
    (tmp_path / "pgdatabase").write_text("vector_db\n", encoding="utf-8")
    url = migrate.migrator_url({"RAG_SECRETS_DIR": str(tmp_path)})
    assert (url.password, url.database) == ("p", "vector_db")


def test_a_missing_password_file_is_named_and_the_step_exits_1(tmp_path: Path, capsys) -> None:
    (tmp_path / "pgdatabase").write_text("vector_db\n", encoding="utf-8")
    assert migrate.main({}, tmp_path) == 1
    out = capsys.readouterr().out
    assert out.startswith("[Migrate] ") and str(tmp_path / "rag_migrator_password") in out
