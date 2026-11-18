"""The parent merge keeps the order in which the search ranked the matched parts and passes rerank_failed on. After a
reranker failure every score is 0.0, so this order is all the cut to VECTOR_COUNT and the ranking have to go by."""

from rag_core.domain import doc_scores  # noqa: E402
from rag_core.domain.retrieval import Hit, Section, parents_of


def sections(*section_ids):
    """The sections in the database's own order (sorted by id), not in the order asked."""
    return {
        section_id: Section(section_id, "Parent " + section_id, {"url": "u/" + section_id})
        for section_id in sorted(section_ids)
    }


def child(section_id, text, score=None):
    meta = {"section_id": section_id}
    if score is not None:
        meta["cross_encoder_score"] = score
    return Hit(page_content=text, metadata=meta)


def test_parents_keep_the_retrieval_order_and_the_rerank_failed_flag():
    children = doc_scores.mark_unscored(
        [child("p9", "a"), child("p3", "b"), child("p7", "c"), child("p3", "d"), child("p1", "e")]
    )
    parents = parents_of(children, sections("p9", "p3", "p7", "p1"))
    assert [p.metadata["section_id"] for p in parents] == ["p9", "p3", "p7", "p1"]
    assert all(p.metadata.get("rerank_failed") for p in parents)
    assert all(p.metadata["cross_encoder_score"] == 0.0 for p in parents)


def test_a_scored_parent_carries_its_best_child():
    parents = parents_of([child("p2", "x", 0.4), child("p2", "y", 0.9)], sections("p2"))
    assert parents[0].metadata["cross_encoder_score"] == 0.9
    assert parents[0].metadata["matched_child_text"] == "y"
    assert "rerank_failed" not in parents[0].metadata
