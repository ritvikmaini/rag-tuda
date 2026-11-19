"""Fusion of the two legs of one query and the cut over all queries: pure functions over hits."""

import pytest

from rag_core.domain.fusion import cut_across_queries, fuse
from rag_core.domain.retrieval import Hit


def semantic(text: str, distance: float) -> Hit:
    return Hit(page_content=text, metadata={"retriever": "semantic", "child_chunk_semantic_search_score": distance})


def lexical(text: str, rank: float) -> Hit:
    return Hit(page_content=text, metadata={"retriever": "fts", "fts_rank": rank})


def fused(text: str, score: float) -> Hit:
    return Hit(page_content=text, metadata={"fusion_score": score})


def scores(hits: list[Hit]) -> dict[str, float]:
    return {h.page_content: h.metadata["fusion_score"] for h in hits}


def test_convex_fusion_minmax() -> None:
    sem = [semantic("a", 0.2), semantic("b", 0.4), semantic("c", 0.6)]  # a distance: the nearest is the best
    lex = [lexical("b", 3.0), lexical("d", 1.0)]
    hits = fuse(sem, lex, 0.7)
    # semantic min-max: a 1.0, b 0.5, c 0.0; lexical: b 1.0, d 0.0; a hit of one leg is 0 in the other
    assert scores(hits) == pytest.approx({"a": 0.7, "b": 0.7 * 0.5 + 0.3, "c": 0.0, "d": 0.0})
    assert [h.page_content for h in hits] == ["a", "b", "c", "d"]  # best fused first, a tie in the legs' order
    assert fuse(sem, lex, 1.0)[1].page_content == "b" and scores(fuse(sem, lex, 0.0))["b"] == 1.0
    both = next(h for h in hits if h.page_content == "b")
    assert both.metadata["fts_rank"] == 3.0 and both.metadata["child_chunk_semantic_search_score"] == 0.4


def test_a_leg_without_spread_scores_one_per_hit() -> None:
    assert scores(fuse([semantic("a", 0.3)], [], 0.5)) == {"a": 0.5}  # one hit: 1.0 weighted, no division by zero
    assert scores(fuse([], [lexical("a", 2.0), lexical("b", 2.0)], 0.25)) == {"a": 0.75, "b": 0.75}  # all equal
    assert scores(fuse([Hit("a", {}), Hit("b", {})], [], 0.5)) == {"a": 0.5, "b": 0.5}  # no score at all
    assert fuse([], [], 0.5) == []


def test_a_semantic_hit_without_a_score_counts_as_the_worst_distance() -> None:
    sem = [semantic("a", 0.2), Hit("unscored", {"retriever": "semantic"}), semantic("b", 0.6)]
    hits = fuse(sem, [], 1.0)
    assert scores(hits) == pytest.approx({"a": 1.0, "unscored": 0.0, "b": 0.0})
    assert hits[0].page_content == "a"
    zero = fuse([semantic("a", 0.0), semantic("b", 0.5)], [], 1.0)  # a distance of 0.0 is a real score, the best
    assert scores(zero) == {"a": 1.0, "b": 0.0}


def test_fuse_leaves_its_input_untouched() -> None:
    sem = [semantic("a", 0.2)]
    fuse(sem, [lexical("a", 1.0)], 0.5)
    assert "fusion_score" not in sem[0].metadata and "fts_rank" not in sem[0].metadata


def test_cut_is_total_across_queries() -> None:
    per_query = [
        [fused("a", 0.9), fused("b", 0.8), fused("c", 0.1)],
        [fused("b", 0.95), fused("d", 0.7), fused("e", 0.6)],
        [],
    ]
    cut = cut_across_queries(per_query, 3)
    assert [h.page_content for h in cut] == ["b", "a", "d"]  # the best of each text, three in total, not per query
    assert cut[0].metadata["fusion_score"] == 0.95
    assert [h.page_content for h in cut_across_queries(per_query, 99)] == ["b", "a", "d", "e", "c"]
    assert cut_across_queries(per_query, 0) == [] and cut_across_queries([], 3) == []
