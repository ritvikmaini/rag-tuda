"""Hypothesis invariants of the gate's arithmetic: the retrieval overlap, the pair comparison and the verdict on a run compared
with itself."""

from pathlib import Path

from evals import gate, stats
from hypothesis import given, settings
from hypothesis import strategies as st

CFG = gate.load_config(Path(__file__).parent / "profile")

sections = st.frozensets(st.sampled_from(["s%d" % i for i in range(20)]), max_size=14)


@st.composite
def run(draw, min_size=0):
    """One run: first and later turns of some conversations, each with sources, a verdict and a time."""
    keys = draw(
        st.lists(
            st.tuples(st.sampled_from(["G01", "G02", "G03", "S01", "S02"]), st.integers(1, 3)),
            unique=True,
            min_size=min_size,
            max_size=8,
        )
    )
    return [
        {
            "id": cid,
            "turn": turn,
            "first_token_s": draw(st.floats(0.1, 30)),
            "total_s": draw(st.floats(0.1, 125)),
            "sources": [{"url": "u", "title": "", "internal": False, "section": s} for s in sorted(draw(sections))],
            "checks": {"auto_pass": draw(st.booleans()), "errors": [], "failed": []},
        }
        for cid, turn in keys
    ]


@settings(max_examples=200, deadline=None)
@given(a=sections, b=sections)
def test_overlap_is_symmetric_bounded_and_one_only_for_equal_sets(a, b):
    score = stats.overlap(a, b)
    assert score == stats.overlap(b, a)
    assert (score is None) == (not a and not b)
    if score is not None:
        assert 0 <= score <= 1
        assert (score == 1) == (a == b)


@settings(max_examples=200, deadline=None)
@given(rows=run())
def test_a_run_compared_with_itself_agrees_fully(rows):
    pair = stats.compare_pair(rows, rows)
    assert all(score == 1.0 for score in pair["scores"])
    assert pair["differing"] == [] and pair["flips"] == [] and pair["missing"] == []


@settings(max_examples=200, deadline=None)
@given(a=run(), b=run())
def test_compare_pair_is_symmetric(a, b):
    ab, ba = stats.compare_pair(a, b), stats.compare_pair(b, a)
    assert sorted(ab["scores"]) == sorted(ba["scores"])
    assert ab["flips"] == ba["flips"]
    assert [d["id"] for d in ab["differing"]] == [d["id"] for d in ba["differing"]]


@settings(max_examples=200, deadline=None)
@given(rows=run())
def test_identical_runs_pass_the_gate(rows):
    assert gate.evaluate({"gold": rows}, {"gold": rows}, {}, CFG, "c")[0]["exit"] == 0


def exit_of(cand, ref):
    return gate.evaluate({"gold": cand}, {"gold": ref}, {}, CFG, "c")[0]["exit"]


@settings(max_examples=200, deadline=None)
@given(data=st.data(), cand=run(min_size=1), ref=run(min_size=1))
def test_two_recorded_generations_always_void(data, cand, ref):
    """Whatever else the pair holds, rows of two index generations (the gate's void rule) end in exit 4."""
    cand[0]["generation"], ref[0]["generation"] = 41, 42
    for r in cand[1:] + ref[1:]:
        generation = data.draw(st.sampled_from([41, 42, None]))
        if generation is not None:
            r["generation"] = generation
    assert exit_of(cand, ref) == 4


@settings(max_examples=200, deadline=None)
@given(cand=run(), ref=run())
def test_unrecorded_runs_never_void(cand, ref):
    assert exit_of(cand, ref) != 4
