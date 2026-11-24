"""The thinking-time normalisation of the golden and terminated-frames comparisons (no browser): a shared shift of
startedAt and endedAt is ignored, a changed duration or any other difference is not."""

import json
from types import ModuleType
from typing import Any

S, E = 1791619200800, 1791619200800 + 400


def stored(started: int = S, ended: int | None = E, text: str = "answer", other: int = 1) -> list[dict[str, Any]]:
    thinking = {"startedAt": started, "endedAt": ended}
    return [{"storage": {"turns": [{"thinking": thinking, "answer": text, "n": other}]}}]


def shot(**kw: Any) -> dict[str, Any]:
    return {
        "label": "a",
        "storage": json.dumps(stored(**kw)[0]["storage"], separators=(",", ":")),
        "captured": "{}",
        "title": "t",
        "pageErrors": [],
        "raw": "<p>x</p>",
    }


def same(e2e: ModuleType, a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, bool]:
    """Equal under comparable() (raw storage strings), and under the golden form (parsed JSON, dumped alike)."""
    dump = lambda v: json.dumps(e2e.relative_times(v), ensure_ascii=False, indent=1)  # noqa: E731
    return (
        e2e.comparable([shot(**a)]) == e2e.comparable([shot(**b)]),
        dump(stored(**a)) == dump(stored(**b)),
    )


def test_a_shared_shift_of_start_and_end_compares_equal(e2e: ModuleType) -> None:
    assert same(e2e, {}, {"started": S + 200, "ended": E + 200}) == (True, True)


def test_a_changed_duration_or_any_other_difference_still_fails(e2e: ModuleType) -> None:
    for other in ({"ended": E + 200}, {"started": S + 200}, {"ended": None}, {"text": "other"}, {"other": 2}):
        assert same(e2e, {}, other) == (False, False), other
