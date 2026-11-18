"""Source description, source selection and ranking policy."""

from datetime import date
from typing import Any

from rag_core.domain import source_meta as sm
from rag_core.domain import ranking as rk

EXAMPLE = sm.SourceLabels(  # the literal of tests/rag_core/domain/test_source_labels.py
    source_types={"map": "Karte"},
    origins={"archive": "Archiv", "website": "example.org"},
    hosts={"docs.example.org": "Archiv", "example.org": "example.org", "www.example.org": "example.org"},
)


def test_describe_source_uses_the_live_metadata():
    url = "https://docs.example.org/archiv/berichte/jahresbericht-2024/?layout=file"
    d = sm.describe_source(
        {
            "title": "Jahresbericht 2024",
            "doc_type": "Haushalt",
            "origin": "archive",
            "doc_year": 2024,
            "download_url": url,
        },
        EXAMPLE,
    )
    assert d == {
        "title": "Jahresbericht 2024",
        "origin": "Archiv",
        "type": "Haushalt",
        "date": "2024",
        "url": url,
    }
    # the legacy title keys are no longer read (every parent of the live index has a title)
    assert (
        sm.describe_source({"download_title": "X", "doc_type": "Haushalt", "download_url": url}, EXAMPLE)["title"]
        == "Dokument"
    )


def test_describe_source_news_and_page():
    assert (
        sm.describe_source(
            {
                "title": "100 Tage im MVV",
                "doc_type": "Pressemitteilung",
                "doc_date": "2026-04-24",
                "url": "https://example.org/news/x",
            },
            EXAMPLE,
        )["date"]
        == "24.04.2026"
    )
    page = sm.describe_source(
        {"title": "Bürgerbüro", "doc_type": "Webseite", "url": "https://example.org/leben/buergerbuero"}, EXAMPLE
    )
    assert page["type"] == "Webseite" and page["origin"] == "example.org"
    assert (
        sm.describe_source({"title": "Wertstoffhof", "origin": "website", "source_type": "map"}, EXAMPLE)["origin"]
        == "Karte"
    )
    assert sm.describe_source({"title": "DV mobile Arbeit", "origin": "archive"}, EXAMPLE)["origin"] == "Archiv"


def test_strip_source_text():
    s = sm.strip_source_text("[Page 3]\nText <!-- 🖼️❌ Image not available. Please use `PdfPipelineOptions(x)` --> mehr")
    assert s == "Text  mehr"


def test_rank_floor_abstains():
    kept, _, rep = rk.rank(
        [{"cross_encoder_score": 0.2}, {"cross_encoder_score": 0.1}], "Gedicht", rk.RankingPolicy(relevance_floor=0.35)
    )
    assert kept == [] and rep["abstain"]


def test_rank_per_document_cap():
    metas = [
        {"cross_encoder_score": 0.8, "doc_type": "Haushalt", "doc_key": "hh"},
        {"cross_encoder_score": 0.7, "doc_type": "Haushalt", "doc_key": "hh"},
        {"cross_encoder_score": 0.6, "doc_type": "Webseite", "doc_key": "web"},
        {"cross_encoder_score": 0.55, "doc_type": "Haushalt", "doc_key": "hh"},
    ]
    kept, _, rep = rk.rank(metas, "Wie viele Schulen hat die Stadt?", rk.RankingPolicy(max_per_document=1))
    assert kept == [0, 2] and rep["capped"] == 2


def test_rank_recency():
    # The older source starts slightly ahead, so only the recency bonus puts the newer one first.
    metas = [
        {"cross_encoder_score": 0.52, "doc_type": "Satzung/Verordnung", "doc_year": 2016},
        {"cross_encoder_score": 0.5, "doc_type": "Satzung/Verordnung", "doc_year": 2025},
    ]
    types = ("Satzung/Verordnung", "Dienstanweisung/Dienstvereinbarung")
    p = rk.RankingPolicy(
        recency_types=types, recency_bonus_per_year=0.01, recency_max_bonus=0.05, today=date(2026, 9, 24)
    )
    kept, _, _ = rk.rank(metas, "Streupflicht", p)
    assert kept == [1, 0]
    dv = [
        {"cross_encoder_score": 0.51, "doc_type": "Dienstanweisung/Dienstvereinbarung", "doc_date": "2022-03-01"},
        {"cross_encoder_score": 0.5, "doc_type": "Dienstanweisung/Dienstvereinbarung", "doc_date": "2024-06-01"},
    ]
    kept2, _, _ = rk.rank(dv, "Welche Dienstvereinbarung zur mobilen Arbeit ist die aktuellste?", p)
    assert kept2 == [1, 0]  # a listed document type gets the recency bonus


def test_rank_date_range_bonus():
    metas = [
        {"cross_encoder_score": 0.6, "doc_type": "Pressemitteilung", "doc_date": "2026-08-01"},
        {"cross_encoder_score": 0.5, "doc_type": "Pressemitteilung", "doc_date": "2026-09-22"},
    ]
    p = rk.RankingPolicy(date_from="2026-09-21", date_to="2026-09-27", in_range_bonus=0.2)
    kept, _, _ = rk.rank(metas, "Was gibt es Neues diese Woche?", p)
    assert kept == [1, 0]


def test_select_sources_for_display():
    src = [{"n": 1, "title": "a"}, {"n": 2, "title": "b"}, {"n": 3, "title": "c"}]
    assert [s["n"] for s in sm.select_sources_for_display(src, "Antwort [2] und [3].", "cited")] == [2, 3]
    assert len(sm.select_sources_for_display(src, "Antwort ohne Beleg.", "cited")) == 3
    assert len(sm.select_sources_for_display(src, "Antwort [1].", "seen")) == 3


def test_rank_keeps_documents_when_the_reranker_failed():
    # after a reranker failure every score is 0.0 and rerank_failed is set;
    # the floor and the minimum score must not turn that into an abstention
    metas = [{"cross_encoder_score": 0.0, "rerank_failed": True}, {"cross_encoder_score": 0.0, "rerank_failed": True}]
    kept, _, rep = rk.rank(metas, "x", rk.RankingPolicy(relevance_floor=0.3, source_min_score=0.2))
    assert kept == [0, 1]  # retrieval order kept, not reversed
    assert rep["rerank_failed"] and not rep["abstain"]


def test_rank_keeps_retrieval_order_for_equal_scores():
    # fp16 reranker scores tie often; with the defaults the order must equal a stable sort
    metas = [{"cross_encoder_score": 0.5}, {"cross_encoder_score": 0.994140625}, {"cross_encoder_score": 0.994140625}]
    kept, _, _ = rk.rank(metas, "x", rk.RankingPolicy())
    assert kept == [1, 2, 0]


LIVE: dict[str, Any] = dict(
    recency_types=("Webseite", "Rundschreiben", "Haushalt"),
    recency_bonus_per_year=0.01,
    recency_max_bonus=0.05,
    in_range_bonus=0.2,
    date_from="2026-09-21",
    date_to="2026-09-27",
    today=date(2026, 9, 29),
)


def test_rank_reranker_failure_keeps_retrieval_order_with_live_bonuses():
    # after a reranker failure every score is 0.0; the live
    # bonuses must not reorder the candidates by date (the retrieval order is kept)
    metas = [
        {"cross_encoder_score": 0.0, "rerank_failed": True, "doc_type": "Webseite"},
        {"cross_encoder_score": 0.0, "rerank_failed": True, "doc_type": "Haushalt", "doc_year": 2020},
        {"cross_encoder_score": 0.0, "rerank_failed": True, "doc_type": "Rundschreiben", "doc_year": 2026},
        {"cross_encoder_score": 0.0, "rerank_failed": True, "doc_type": "Pressemitteilung", "doc_date": "2026-09-22"},
    ]
    kept, _, rep = rk.rank(metas, "Wer ist zuständig?", rk.RankingPolicy(**LIVE))
    assert kept == [0, 1, 2, 3]
    assert rep["rerank_failed"] and not rep["abstain"]


def test_years_ignore_encoded_digits_and_future_years():
    # "%20" + two digits is not a year; years after next year are unknown
    web = "https://www.example.org/leben/%s"
    assert (
        sm.describe_source({"title": "Schulung", "url": web % "schulung/#Schulung%20%C2%A7%2045%20a"}, EXAMPLE)["date"]
        == ""
    )
    assert sm.describe_source({"title": "Stadtradeln", "url": web % "Stadtradeln%2021/"}, EXAMPLE)["date"] == ""
    assert sm.describe_source({"title": "STEP 2040", "url": web % "step"}, EXAMPLE)["date"] == ""
    assert sm.describe_source({"title": "Sprengel", "url": web % "sprengel-2019/"}, EXAMPLE)["date"] == "2019"
    # a decoded fragment keeps its real date; an index doc_year after next year is not shown
    assert (
        sm.describe_source({"title": "Radeln", "url": web % "radeln/#Stadtradeln%2011.%20Juli%202026"}, EXAMPLE)["date"]
        == "2026"
    )
    assert sm.describe_source({"title": "Plan", "doc_year": 2040, "url": web % "plan"}, EXAMPLE)["date"] == ""
    metas = [
        {"cross_encoder_score": 0.5, "doc_type": "Webseite", "title": "Agenda 2030", "url": web % "agenda"},
        {"cross_encoder_score": 0.5, "doc_type": "Webseite", "title": "Schulung", "url": web % "x/#a%2045"},
        {"cross_encoder_score": 0.5, "doc_type": "Webseite", "title": "Sprengel 2026/2027", "url": web % "sprengel"},
        {"cross_encoder_score": 0.5, "doc_type": "Webseite", "doc_year": "zwanzig"},
        {
            "cross_encoder_score": 0.5,
            "doc_type": "Webseite",
            "title": "Radeln",
            "url": web % "radeln/#Stadtradeln%2011.%20Juli%202026",
        },
    ]
    _, adj, _ = rk.rank(metas, "Was gilt?", rk.RankingPolicy(**LIVE))
    assert adj[0] == 0.5 and adj[1] == 0.5 and adj[3] == 0.5  # no recency bonus for unknown years
    assert round(adj[2], 4) == 0.55  # next year still counts (full bonus)
    assert round(adj[4], 4) == 0.55  # the decoded fragment's 2026, not "%2011"
