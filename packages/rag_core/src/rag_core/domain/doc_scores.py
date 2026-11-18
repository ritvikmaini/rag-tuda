"""
Scores on retrieved documents. A missing or empty score counts as 0.0, so a
reranker outage can never crash the ordering before the cut to VECTOR_COUNT
(the fallback documents once carried cross_encoder_score None and the sort
raised TypeError). Pure functions.
"""

from collections.abc import Mapping
from typing import Any


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def sort_score(metadata: Mapping[str, Any] | None) -> float:
    """Score used to order documents: the cross-encoder score, else 0.0."""
    value = _number((metadata or {}).get("cross_encoder_score"))
    return value if value is not None else 0.0


def mark_unscored(documents: list[Any]) -> list[Any]:
    """After a reranker failure: every document gets cross_encoder_score 0.0 and
    rerank_failed True; the retrieval order is kept (a stable sort leaves ties)."""
    for doc in documents:
        doc.metadata["cross_encoder_score"] = 0.0
        doc.metadata["rerank_failed"] = True
    return documents
