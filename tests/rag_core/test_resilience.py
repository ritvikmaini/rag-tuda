"""Bounded parallel calls: one slow or failing search never sinks the others."""

import asyncio

from rag_core.domain import resilience as rs  # noqa: E402


async def ok(value, delay=0.0):
    await asyncio.sleep(delay)
    return value


async def boom():
    raise ConnectionError("database down")


def test_results_keep_their_order_and_failures_become_none():
    results, errors = asyncio.run(rs.gather_bounded([ok("a"), ok("b", 5), boom()], timeout_s=0.1))
    assert results == ["a", None, None]
    assert [i for i, _ in errors] == [1, 2]
    assert errors[0][1].startswith("TimeoutError") and errors[1][1] == "ConnectionError"


def test_all_succeed_without_errors():
    assert asyncio.run(rs.gather_bounded([ok(1), ok(2)], timeout_s=1)) == ([1, 2], [])


def test_nothing_to_do():
    assert asyncio.run(rs.gather_bounded([], timeout_s=1)) == ([], [])
