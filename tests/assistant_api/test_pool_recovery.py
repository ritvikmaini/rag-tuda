"""Pool recovery after a Postgres restart: every idle connection of a pool is dead, psycopg_pool's getconn checks them one by
one and backs off between failed checks (0, 1, 2, 4 s), so four dead connections outlast the 5 s timeout and the first request
fails with PoolTimeout. The check callback sweeps the pool on the first failed check (pool.check() discards every dead idle
connection at once), so the next round finds a fresh one. Helper tested with fakes; the wiring in app.py at source level; the
behaviour on a real database restart is tests/integration/test_database_restart.py (the harness)."""

import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
from rag_core.domain import resilience as rs  # noqa: E402

BACKEND = (ROOT / "apps" / "assistant_api" / "src" / "assistant_api" / "app.py").read_text(encoding="utf-8")


class FakePool:
    def __init__(self, broken):
        self.broken = broken
        self.swept = 0

    async def check_connection(self, conn):
        if conn in self.broken:
            raise OSError("terminating connection due to administrator command")

    async def check(self):
        self.swept += 1


def test_a_working_connection_passes_without_a_sweep():
    pool = FakePool(broken=set())
    asyncio.run(rs.check_or_sweep(pool, "a"))
    assert pool.swept == 0


def test_a_dead_connection_sweeps_the_pool_and_still_fails_its_check():
    pool = FakePool(broken={"a"})
    with pytest.raises(OSError, match="administrator command"):
        asyncio.run(rs.check_or_sweep(pool, "a"))
    assert pool.swept == 1


def test_the_lifespan_opens_the_turn_pool_through_read_pool():
    # the turn store's pool is read_pool's (check_or_sweep inside; its own test is
    # test_the_pool_prepares_nothing_and_checks_connections), at most 4 connections
    lifespan = BACKEND[BACKEND.index("async def lifespan") :]
    assert (
        'read_pool(store_conninfo(db), max_size=4, name="turns")' in BACKEND and "AsyncConnectionPool(" not in BACKEND
    )
    assert "TurnStore(pool)" in lifespan and "make_checkpointer" not in BACKEND
    assert "check=AsyncConnectionPool.check_connection" not in BACKEND
