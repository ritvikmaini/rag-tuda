"""Source-level checks that the graph nodes and the request handler use the history policy."""

import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
NODES = os.path.join(ROOT, "apps", "assistant_api", "src", "assistant_api", "graph", "nodes")


SERVICE = open(
    os.path.join(ROOT, "apps", "assistant_api", "src", "assistant_api", "chat", "service.py"), encoding="utf-8"
).read()


def test_the_cut_is_applied_again_once_the_pipeline_slot_is_held():
    # an earlier run of the same conversation may still store a turn until the slot is free: the
    # store's begin cuts before the early exits and its history cuts again once the slot is held
    source = SERVICE
    assert source.index("rt.turns.begin(") < source.index("max_question_chars") < source.index("rt.slot")
    assert source.index("rt.slot") < source.index("rt.turns.history(")


def test_both_nodes_take_the_history_policy_from_the_profile():
    for name in ("analyse.py", "generate.py"):
        node = open(os.path.join(NODES, name), encoding="utf-8").read()
        assert "context_turns(" in node and "history.turns" in node and "history.max_age_hours" in node, name
    assert "select_turns(" in open(os.path.join(NODES, "generate.py"), encoding="utf-8").read()
