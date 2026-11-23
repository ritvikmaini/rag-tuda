"""Recency: the bonus goes only to the document types the profile lists."""

from datetime import date

from rag_core.domain.ranking import RankingPolicy, rank

RECENCY = dict(recency_bonus_per_year=0.01, recency_max_bonus=0.05, today=date(2026, 10, 2))
METAS = [
    {"cross_encoder_score": 0.5, "doc_type": "Mitteilungsblatt", "doc_year": 2026},
    {"cross_encoder_score": 0.5, "doc_type": "Webseite", "doc_year": 2026},
]


def adjusted(**policy):
    _, scores, _ = rank(METAS, "q", RankingPolicy(**RECENCY, **policy))
    return [round(scores[i], 3) for i in range(len(METAS))]


def test_recency_bonus_only_for_profile_types():
    assert adjusted(recency_types=("Mitteilungsblatt",)) == [0.55, 0.5]
    assert adjusted(recency_types=("Webseite",)) == [0.5, 0.55]


def test_empty_recency_types_gives_no_bonus():
    assert RankingPolicy().recency_types == ()
    assert adjusted() == [0.5, 0.5]
