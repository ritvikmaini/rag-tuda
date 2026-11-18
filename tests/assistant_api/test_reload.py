"""The backend's reload keeps the engines it built and disposes the previous ones after the swap; the shutdown uses the same
helper. dispose_engines is tested with fakes; the wiring is checked at source level (runtime.py, app.py)."""

import asyncio
import os
import re

from rag_core.domain import resilience as rs  # noqa: E402

PACKAGE = os.path.join(os.path.dirname(__file__), "..", "..", "apps", "assistant_api", "src", "assistant_api")


def source(name):
    with open(os.path.join(PACKAGE, name), encoding="utf-8") as fh:
        return fh.read()


RUNTIME, APP = source("runtime.py"), source("app.py")


class Closable:
    closed = False

    async def close(self):
        self.closed = True


class Disposable:
    disposed = False

    async def dispose(self):
        self.disposed = True


class Broken:
    async def close(self):
        raise RuntimeError("pool gone")


def test_engines_are_closed_or_disposed_and_none_is_skipped():
    a, b = Closable(), Disposable()
    assert asyncio.run(rs.dispose_engines(a, None, b)) == 2
    assert a.closed and b.disposed


def test_a_failing_engine_is_logged_not_raised():
    assert asyncio.run(rs.dispose_engines(Broken(), object())) == 0


def test_the_reload_swaps_then_retires_the_old_engines():
    reload = RUNTIME[RUNTIME.index("async def reload(") : RUNTIME.index("async def try_reload")]
    assert "old = rt.index" in reload
    assert reload.count("rt.index = index") == 1
    assert reload.index("rt.chain = await asyncio.to_thread(build_graph") < reload.index("retire(rt, old.engines")
    assert reload.index("LangGraph chain has been re-created with new data.") < reload.index("retire(rt, old.engines")
    # the previous pool closes only after the request deadline: psycopg_pool's close() fails every later getconn
    retire = RUNTIME[RUNTIME.index("def retire(") : RUNTIME.index("async def try_reload")]
    assert retire.index("await asyncio.sleep(grace_s)") < retire.index("await dispose_engines(*engines)")


def test_overlapping_reload_attempts_run_one_at_a_time():
    # two reloads at once would dispose the same old engines and leak the first reload's new ones
    attempt = RUNTIME[RUNTIME.index("async def try_reload") : RUNTIME.index("def _logged_in")]
    assert "async with rt.reload_lock:" in attempt and "await reload(rt)" in attempt


def test_the_lifespan_clears_and_disposes_through_the_helper():
    lifespan = APP[APP.index("async def lifespan") :]
    assert "rt.index = None" in lifespan
    call = "await dispose_engines(*(rt.index.engines if rt.index is not None else ()), *stop_retiring(rt))"
    assert re.sub(r"\s+", "", call) in re.sub(r"\s+", "", lifespan)  # ruff format may wrap the call
    assert "register_all_prompts" not in APP
