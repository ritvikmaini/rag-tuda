"""A reranker outage must never crash the ordering of documents."""

from types import SimpleNamespace

from rag_core.domain import doc_scores as ds  # noqa: E402


def test_sort_score_reads_the_cross_encoder_score():
    assert ds.sort_score({"cross_encoder_score": 0.4}) == 0.4
    assert ds.sort_score({"recency_boosted_score": 0.7, "cross_encoder_score": 0.2}) == 0.2


def test_missing_or_empty_scores_count_as_zero():
    for meta in ({}, None, {"cross_encoder_score": None}, {"cross_encoder_score": "x"}, {"cross_encoder_score": True}):
        assert ds.sort_score(meta) == 0.0


def test_unscored_documents_sort_without_error_and_keep_their_order():
    docs = [SimpleNamespace(metadata={"cross_encoder_score": None, "n": i}) for i in range(5)]
    docs.sort(key=lambda d: ds.sort_score(d.metadata), reverse=True)
    assert [d.metadata["n"] for d in docs] == [0, 1, 2, 3, 4]


def test_mark_unscored_after_a_reranker_failure():
    docs = [SimpleNamespace(metadata={"url": "a"}), SimpleNamespace(metadata={"cross_encoder_score": None})]
    out = ds.mark_unscored(docs)
    assert out is docs
    assert all(d.metadata["cross_encoder_score"] == 0.0 and d.metadata["rerank_failed"] for d in docs)
