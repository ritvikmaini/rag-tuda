"""Parallel conversations for the load test."""

import threading
import time

from evals import runner as rp


def fake_ask(answer="Die Kernarbeitszeit endet um 15:30 Uhr [1]."):
    calls = []

    def ask_fn(question, user, opts=None):
        calls.append((question, user))
        return (
            answer,
            [{"metadata": {"url": "https://docs.example.org/personal/x"}}],
            [],
            0.1,
            0.2,
            ["chunk", "docs"],
        )

    return ask_fn, calls


def test_run_conversation_plays_turns_in_order_and_emits_each_row():
    conv = {
        "id": "C9",
        "set": "memory",
        "turns": [{"q": "Erste?"}, {"q": "Zweite?", "expect": {"contains_any": ["15:30"]}}],
    }
    ask_fn, calls = fake_ask()
    emitted = []
    rows = rp.run_conversation(conv, ask_fn, 0.0, "t", "20260925-120000", emitted.append)
    assert [q for q, _ in calls] == ["Erste?", "Zweite?"]
    assert {u for _, u in calls} == {"probe-t-20260925-120000-C9"}
    assert [r["turn"] for r in rows] == [1, 2] and emitted == rows
    assert rows[1]["checks"]["auto_pass"] and rows[1]["id"] == "C9"


def test_run_all_limits_parallel_conversations_and_keeps_file_order():
    active, peak, lock = [0], [0], threading.Lock()

    def run_one(conv):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return [{"id": conv["id"]}]

    convs = [{"id": "C%d" % i} for i in range(7)]
    rows = rp.run_all(convs, 3, run_one)
    assert [r["id"] for r in rows] == ["C%d" % i for i in range(7)]
    assert 2 <= peak[0] <= 3


def test_run_all_sequential_by_default():
    order = []
    rows = rp.run_all([{"id": "A"}, {"id": "B"}], 1, lambda c: order.append(c["id"]) or [{"id": c["id"]}])
    assert order == ["A", "B"] and [r["id"] for r in rows] == ["A", "B"]
