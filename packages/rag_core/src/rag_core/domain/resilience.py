"""
Bounded parallel calls: each awaitable gets its own time limit,
and a slow or failing one never cancels or sinks the others. Pure asyncio.
"""

import asyncio
from collections.abc import Awaitable, Iterable
from typing import Any


async def gather_bounded(
    awaitables: Iterable[Awaitable[Any]], timeout_s: float
) -> tuple[list[Any], list[tuple[int, str]]]:
    """Run the awaitables concurrently, each limited to timeout_s seconds.

    Returns (results, errors): results in input order with None for every call that
    failed or timed out; errors as (index, reason) pairs. Cancelling the caller
    still cancels everything."""
    awaitables = list(awaitables)
    if not awaitables:
        return [], []
    outcomes = await asyncio.gather(
        *(asyncio.wait_for(a, timeout=timeout_s) for a in awaitables), return_exceptions=True
    )
    results: list[Any] = []
    errors: list[tuple[int, str]] = []
    for index, outcome in enumerate(outcomes):
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, asyncio.TimeoutError):
            results.append(None)
            errors.append((index, f"TimeoutError after {timeout_s:g} s"))
        elif isinstance(outcome, BaseException):
            results.append(None)
            errors.append((index, f"{type(outcome).__name__}: {outcome}"))
        else:
            results.append(outcome)
    return results, errors


async def dispose_engines(*engines: Any, log: Any = None) -> int:
    """Closes database engines the reload or the shutdown no longer needs: an engine has
    close() or dispose(); checked-out connections finish their request and close on return.
    Returns how many engines were closed; a failing or unknown one is logged, never raised."""
    import logging

    log = log or logging
    closed = 0
    for engine in engines:
        if engine is None:
            continue
        method = getattr(engine, "close", None) or getattr(engine, "dispose", None)
        if method is None:
            log.warning(f"[Reload] engine {type(engine).__name__} has neither close() nor dispose()")
            continue
        try:
            result = method()
            if asyncio.iscoroutine(result):
                await result
            closed += 1
        except Exception as exc:
            log.warning(f"[Reload] disposing {type(engine).__name__} failed: {exc}")
    return closed


async def check_or_sweep(pool: Any, conn: Any) -> None:
    """The read pools' check (read_pool: the index reads and the turn store). After a Postgres restart every idle connection is dead; psycopg_pool's getconn checks them one at a time
    with a growing pause (0, 1, 2, 4 s), so four dead ones outlast the 5 s timeout and the first request fails. On the
    first failed check, pool.check() discards every dead idle connection at once and opens replacements, so getconn's
    next round gets a fresh one. Raises like the check."""
    try:
        await pool.check_connection(conn)
    except Exception:
        await pool.check()
        raise
