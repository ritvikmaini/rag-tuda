"""The reranker scores each child and its parent section, a child keeps the higher score."""

from langchain_core.documents import Document

from rag_core.domain.parent_rerank import parent_texts, rerank_texts, score_children


def child(text, section_id):
    return Document(page_content=text, metadata={"section_id": section_id})


def test_parent_texts_title_prefix_without_page_marks():
    parents = {
        "a": {"text": "[Page 3]\nDie Kernzeit beginnt.\n\n[Page 4]\nSie endet.", "metadata": {"title": "DA Zeit"}},
        "b": {"text": "Ohne Titel.", "metadata": {}},
        "c": {"text": "", "metadata": {"title": "Leer"}},
    }
    out = parent_texts(parents)
    assert out["a"].startswith("DA Zeit: ") and "[Page" not in out["a"] and "Sie endet." in out["a"]
    assert out["b"] == "Ohne Titel." and "c" not in out
    assert parent_texts(None) == {}


def test_rerank_texts_children_first_then_distinct_parents():
    docs = [child("k1", "a"), child("k2", "a"), child("k3", "b"), child("k4", None)]
    assert rerank_texts(docs, {"a": "P-A", "b": "P-B"}) == ["k1", "k2", "k3", "k4", "P-A", "P-B"]
    assert rerank_texts(docs, {"a": "k1"}) == [
        "k1",
        "k2",
        "k3",
        "k4",
    ]  # a parent equal to a child text is not sent twice


def test_score_children_takes_the_max_and_sorts():
    docs = [child("k1", "a"), child("k2", "b"), child("k3", "c"), child("k4", "a")]
    scores = {"k1": 0.2, "k2": 0.6, "k3": 0.3, "P-A": 0.9, "P-B": 0.1}
    out = score_children(docs, scores, {"a": "P-A", "b": "P-B"})
    assert [d.page_content for d in out] == ["k1", "k2", "k3"]  # k4 was not scored: left out
    assert [d.metadata["cross_encoder_score"] for d in out] == [
        0.9,
        0.6,
        0.3,
    ]  # k1 takes its parent's 0.9; k3 has no parent text


def test_tied_children_of_a_winning_parent_keep_their_own_order():
    """When the parent's score beats its children, they tie; the child with the best own score must lead, because the
    first child of a parent becomes matched_child_text (citation quote, neighbour anchor)."""
    docs = [child("schwach", "a"), child("stark", "a"), child("anders", "b")]
    out = score_children(docs, {"schwach": 0.1, "stark": 0.4, "anders": 0.5, "P-A": 0.9}, {"a": "P-A"})
    assert [d.page_content for d in out] == ["stark", "schwach", "anders"]
    assert out[0].metadata["cross_encoder_score"] == out[1].metadata["cross_encoder_score"] == 0.9
    assert out[0].metadata["child_score"] == 0.4
