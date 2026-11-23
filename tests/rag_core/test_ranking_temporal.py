"""rank(): time rules. Every switch is off by default."""

from datetime import date

from rag_core.domain import ranking as rk  # noqa: E402

TODAY = date(2026, 10, 2)


def scores(metas, question, **policy):
    _, adjusted, _ = rk.rank(metas, question, rk.RankingPolicy(today=TODAY, **policy))
    return [round(adjusted[i], 3) for i in range(len(metas))]


def test_a_year_only_source_inside_the_period_gets_the_in_range_bonus():
    metas = [
        {"cross_encoder_score": 0.5, "doc_type": "Mitteilungsblatt", "doc_year": 2023},
        {"cross_encoder_score": 0.5, "doc_type": "Mitteilungsblatt", "doc_year": 2024},
        {"cross_encoder_score": 0.5, "doc_type": "Pressemitteilung", "doc_date": "2023-05-02", "doc_year": 2023},
    ]
    period = dict(date_from="2023-01-01", date_to="2023-12-31", in_range_bonus=0.2)
    assert scores(metas, "q", **period) == [0.5, 0.5, 0.7]  # default: exact dates only
    assert scores(metas, "q", in_range_by_year=True, **period) == [0.7, 0.5, 0.7]


def test_the_years_a_question_names_give_the_period_when_the_analysis_has_none():
    assert rk.question_period("Wie hoch war der Hebesatz im Jahr 2024?") == ("2024-01-01", "2024-12-31")
    assert rk.question_period("Schulen im Schuljahr 2022/2023?") == ("2022-01-01", "2023-12-31")
    assert rk.question_period("Wie viele Studierende hat die Universität Beispiel?") == (None, None)
    assert rk.question_period("Studierende am 31.12.2012 und Ende 2015?") == ("2012-01-01", "2015-12-31")
    assert rk.question_period("Kosten 1.500 Euro, Tel. 0000 556677, 2.019,50 €") == (None, None)
    metas = [
        {"cross_encoder_score": 0.5, "doc_type": "Haushalt", "doc_year": 2024},
        {"cross_encoder_score": 0.5, "doc_type": "Haushalt", "doc_year": 2026},
    ]
    on = dict(in_range_bonus=0.2, in_range_by_year=True)
    assert scores(metas, "Hebesatz im Jahr 2024?", **on) == [0.5, 0.5]  # the analysis gave no dates
    assert scores(metas, "Hebesatz im Jahr 2024?", question_years=True, **on) == [0.7, 0.5]
    assert scores(metas, "Hebesatz 2024?", question_years=True, date_from="2026-01-01", date_to="2026-12-31", **on) == [
        0.5,
        0.7,
    ]  # the analysis's dates win


def test_no_recency_bonus_inside_a_named_period():
    metas = [{"cross_encoder_score": 0.5, "doc_type": "Mitteilungsblatt", "doc_year": 2026}]
    recency = dict(recency_types=("Mitteilungsblatt",), recency_bonus_per_year=0.01, recency_max_bonus=0.05)
    assert scores(metas, "q", **recency) == [0.55]
    assert scores(metas, "q", date_from="2015-01-01", date_to="2015-12-31", **recency) == [0.55]  # default
    assert scores(metas, "q", date_from="2015-01-01", date_to="2015-12-31", no_recency_in_range=True, **recency) == [
        0.5
    ]
    assert scores(metas, "q", no_recency_in_range=True, **recency) == [0.55]  # no period: recency stays
