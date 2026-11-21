"""The retrieval sequence as a service over fakes of its ports (the SQL adapters are the harness's)."""

import asyncio
import logging
from datetime import date
from pathlib import Path

import httpx
import pytest

from rag_core.adapters.reranker_http import RerankerHttp
from rag_core.domain.filters import Filter
from rag_core.domain.retrieval import Hit, Section
from rag_core.ports.retrieval import RerankError
from rag_core.services.retrieval import RetrievalService
from rag_core.settings import load_profile
from rag_core.settings.profile import Profile
from rag_core.settings.sections import Timeouts

PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "example" / "profile.yaml"
TIMEOUTS = Timeouts(
    retrieval_s=10,
    parent_lookup_s=5,
    reranker_s=15,
)


class Embedder:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.asked.append(text)
        return [0.0]


class Vector:
    def __init__(self, texts: dict[str, str], fail: bool = False) -> None:
        self.texts, self.fail, self.filters = texts, fail, []

    async def search(self, vector, k, *, max_distance=None, filter=None):
        self.filters.append(filter)
        if self.fail:
            raise RuntimeError("db down")
        return [Hit(page_content=t, metadata={"section_id": i, "retriever": "semantic"}) for t, i in self.texts.items()]


class FullText(Vector):
    async def search(self, query, k, filter=None):  # type: ignore[override]
        return await Vector.search(self, [0.0], k, filter=filter)


class Sections:
    def __init__(self, sections: dict[str, Section]) -> None:
        self.sections, self.filters = sections, []

    async def get_many(self, section_ids, filter=None):
        self.filters.append(filter)
        return {i: self.sections[i] for i in section_ids if i in self.sections}

    async def sections_by_ref(self, regulation, paragraph, programme, po_version, filter=None):
        return []

    async def ping(self) -> None:
        return None


class Deny:
    def __init__(self, section_id: str) -> None:
        self.section_id = section_id

    def allows(self, principal, section) -> bool:
        return section.section_id != self.section_id


class Reranker:
    def __init__(self, scores: dict[str, float] | None) -> None:
        self.scores = scores

    async def rerank(self, queries, texts, timeout_s):
        if self.scores is None:
            raise RerankError("HTTPStatusError boom")
        return sorted(((self.scores.get(t, 0.0), t) for t in set(texts)), reverse=True)


def make_service(
    vector=None, full_text=None, sections=None, reranker=None, access=None, final_k=8, pre_rerank_k=20, rerank_queries=1,
    fusion_cut=True, whole_parents=True, relevance_floor=0.0, chars_per_token=3.3,
) -> RetrievalService:  # fmt: skip
    profile = load_profile(Profile, PROFILE)
    switches = profile.retrieval.switches.model_copy(update={"fusion_cut": fusion_cut, "whole_parents": whole_parents})
    retrieval = profile.retrieval.model_copy(update={"switches": switches})
    sections = sections or Sections(
        {f"p{i}": Section(f"p{i}", f"Parent {i}", {"filename": f"f{i}.pdf"}) for i in range(1, 6)}
    )
    return RetrievalService(
        Embedder(),
        vector or Vector(CHILDREN),
        full_text or FullText({}),
        sections,
        reranker or Reranker({}),
        retrieval,
        profile.ranking.model_copy(update={"relevance_floor": relevance_floor}),
        TIMEOUTS,
        final_k,
        pre_rerank_k,
        rerank_queries,
        access,
        chars_per_token=chars_per_token,
    )


CHILDREN = {"child one": "p1", "child two": "p2", "child three": "p3", "child four": "p4"}


def test_children_are_reranked_cut_and_lifted_to_their_parents() -> None:
    scores = {"Parent 1": 0.9, "Parent 2": 0.8, "Parent 3": 0.1, "Parent 4": 0.05}  # the parent text scores too
    docs = asyncio.run(make_service(reranker=Reranker(scores)).retrieve(["q"], question="q"))
    assert [d.page_content for d in docs] == ["Parent 1", "Parent 2", "Parent 3"]  # threshold 0.50, min_docs 3
    assert all("rank_score" in d.metadata for d in docs)


def test_the_filter_reaches_both_legs_and_every_section_lookup() -> None:
    vector, full_text, sections = Vector(CHILDREN), FullText(CHILDREN), Sections({"p1": Section("p1", "Parent 1", {})})
    flt = Filter(po_version=(2023,))
    asyncio.run(
        make_service(vector, full_text, sections, Reranker({"child one": 0.9})).retrieve(["q"], flt, question="q")
    )
    assert vector.filters == full_text.filters == [flt]
    assert sections.filters == [
        flt,
        flt,
    ]  # the parent texts for the reranker, then the parents of the reranked children
    asyncio.run(make_service(vector, full_text, sections, Reranker({})).retrieve(["q"], question="q"))
    assert vector.filters[-1] is None and sections.filters[-1] is None  # no filter given: none applied


def test_the_result_is_cut_to_the_cards_final_k() -> None:
    scores = {f"Parent {i}": 0.9 for i in range(1, 5)}
    docs = asyncio.run(make_service(reranker=Reranker(scores), final_k=2).retrieve(["q"], question="q"))
    assert len(docs) == 2


def test_a_reranker_failure_keeps_the_search_order_unscored(caplog) -> None:
    caplog.set_level(logging.ERROR)
    docs = asyncio.run(make_service(reranker=Reranker(None)).retrieve(["q"], question="q"))
    assert docs and all(d.metadata.get("rerank_failed") and d.metadata["cross_encoder_score"] == 0.0 for d in docs)
    assert (
        "Error calling reranker multi-query service: HTTPStatusError boom. Falling back to original documents."
        in caplog.text
    )


def test_the_adapters_failure_is_logged_in_the_backends_words(caplog) -> None:
    # the log line reads f"...service: {type(e).__name__} {e}. Falling back..." for the exception httpx raised
    caplog.set_level(logging.ERROR)
    request = httpx.Request("POST", "http://reranker_service:8003/v2/rerank")
    with pytest.raises(httpx.HTTPStatusError) as raised:
        httpx.Response(503, request=request).raise_for_status()
    old = raised.value
    expected = (
        f"Error calling reranker multi-query service: {type(old).__name__} {old}. Falling back to original documents."
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    asyncio.run(
        make_service(reranker=RerankerHttp("http://reranker_service:8003", client)).retrieve(["q"], question="q")
    )
    assert expected in caplog.text


def test_a_failed_query_is_logged_by_its_class_only(caplog) -> None:
    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        asyncio.run(
            make_service(vector=Vector(CHILDREN, fail=True), full_text=FullText({}, fail=True)).retrieve(
                ["q1"], question="q"
            )
        )
    assert "retrieval query 1/1 failed: RuntimeError" in caplog.text and "db down" not in caplog.text


def test_every_query_failing_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="retrieval unavailable: every query failed"):
        asyncio.run(
            make_service(vector=Vector(CHILDREN, fail=True), full_text=FullText({}, fail=True)).retrieve(
                ["q1", "q2"], question="q"
            )
        )


def test_without_parents_the_chunks_are_the_context(caplog) -> None:
    caplog.set_level(logging.WARNING)
    docs = asyncio.run(
        make_service(
            reranker=Reranker({"child one": 0.9}), sections=Sections({"none": Section("none", "", {})})
        ).retrieve(["q"], question="q")
    )
    assert docs and docs[0].page_content == "child one"
    assert "no parents for the matched chunks; answering from the chunks" in caplog.text


class FailingSections(Sections):
    def __init__(self, sections: dict[str, Section], batch: bool = False, lookup: bool = False) -> None:
        super().__init__(sections)
        self.batch, self.lookup = batch, lookup

    async def get_many(self, section_ids, filter=None):
        if self.batch:  # the first call (the reranker's parent texts) fails, the parents lookup after it works
            self.batch = False
            raise RuntimeError("db down")
        if self.lookup:
            raise asyncio.TimeoutError
        return await super().get_many(section_ids, filter)


class Verbatim(Reranker):
    """Returns exactly the scores it was given, also for texts that were not sent."""

    async def rerank(self, queries, texts, timeout_s):
        return sorted(((score, text) for text, score in (self.scores or {}).items()), reverse=True)


def retrieve(sections, scores, vector=None, reranker=Reranker, **changes):
    profile = load_profile(Profile, PROFILE)
    rerank = profile.retrieval.rerank.model_copy(update=changes)
    retrieval = profile.retrieval.model_copy(update={"rerank": rerank})
    svc = RetrievalService(
        Embedder(),
        vector or Vector(CHILDREN),
        FullText({}),
        sections,
        reranker(scores),
        retrieval,
        profile.ranking,
        TIMEOUTS,
        8,
        20,
        1,
        chars_per_token=3.3,
    )
    return asyncio.run(svc.retrieve(["q"], question="q"))


PARENTS = {f"p{i}": Section(f"p{i}", f"Parent {i}", {"filename": f"f{i}.pdf"}) for i in range(1, 6)}


def test_a_failing_parent_text_lookup_scores_the_children_alone(caplog) -> None:
    caplog.set_level(logging.ERROR)
    docs = retrieve(FailingSections(PARENTS, batch=True), {"child one": 0.9, "child two": 0.8, "child three": 0.7})
    assert [d.page_content for d in docs][:3] == ["Parent 1", "Parent 2", "Parent 3"]
    assert "parent texts for reranking unavailable: RuntimeError" in caplog.text and "db down" not in caplog.text


def test_a_timed_out_parent_lookup_answers_from_the_chunks(caplog) -> None:
    caplog.set_level(logging.WARNING)
    docs = retrieve(FailingSections(PARENTS, lookup=True), {"child one": 0.9}, parent_text=False)
    assert docs and docs[0].page_content == "child one"
    assert "parent lookup timed out" in caplog.text and "answering from the chunks" in caplog.text


def test_without_parent_text_a_reranked_text_that_is_no_document_is_skipped() -> None:
    scores = {"child one": 0.9, "child two": 0.8, "child three": 0.7, "a text nobody searched": 0.99}
    docs = retrieve(Sections(PARENTS), scores, reranker=Verbatim, parent_text=False)
    assert [d.page_content for d in docs] == ["Parent 1", "Parent 2", "Parent 3"]


def test_every_child_above_the_threshold_needs_no_min_doc_guarantee() -> None:
    scores = {"Parent 1": 0.9, "Parent 2": 0.8, "Parent 3": 0.7, "Parent 4": 0.6}
    assert len(retrieve(Sections(PARENTS), scores)) == 4


def test_a_search_that_finds_nothing_returns_nothing() -> None:
    assert retrieve(Sections(PARENTS), {}, vector=Vector({})) == []


def test_a_query_searches_both_ways_and_one_failure_does_not_sink_the_other() -> None:
    service = make_service(vector=Vector({"v": "a"}), full_text=FullText({}, fail=True))
    hits = asyncio.run(service._search("q", None))
    assert [h.page_content for h in hits] == ["v"] and service.embedder.asked == ["q"]
    both = make_service(vector=Vector({}, fail=True), full_text=FullText({}, fail=True))
    with pytest.raises(RuntimeError, match="every search failed") as raised:
        asyncio.run(both._search("Personalnummer 4711 Beihilfe", None))
    assert "4711" not in str(raised.value)  # no question text in an error or a log


def test_a_failing_search_logs_the_kind_and_the_reason_never_the_query(caplog) -> None:
    caplog.set_level(logging.ERROR)
    service = make_service(vector=Vector({"v": "a"}), full_text=FullText({}, fail=True))
    asyncio.run(service._search("Personalnummer 4711 Beihilfe", None))
    assert "[HybridRetriever] keyword search failed: " in caplog.text
    assert "4711" not in caplog.text and "Beihilfe" not in caplog.text


def test_a_denied_section_is_not_a_parent() -> None:
    scores = {"Parent 1": 0.9, "Parent 2": 0.8, "Parent 3": 0.1, "Parent 4": 0.05}
    allowed_hits = asyncio.run(make_service(reranker=Reranker(scores)).retrieve(["q"], question="q"))
    hits = asyncio.run(make_service(reranker=Reranker(scores), access=Deny("p2")).retrieve(["q"], question="q"))
    assert "p2" in {h.metadata.get("section_id") for h in allowed_hits}
    assert "p2" not in {h.metadata.get("section_id") for h in hits}


def test_a_denied_sections_children_never_answer() -> (
    None
):  # not through the parent text, not through the rerank fallback
    service = make_service(
        vector=Vector({"child two": "p2"}), reranker=Reranker({"child two": 0.9, "Parent 2": 0.9}), access=Deny("p2")
    )
    assert asyncio.run(service.retrieve(["q"], question="q")) == []


def test_the_ranking_policy_is_the_profiles_ranking_with_the_request_dates(monkeypatch) -> None:
    from rag_core.domain.ranking import RankingPolicy, rank
    from rag_core.services import retrieval as retrieval_service

    seen: list[RankingPolicy] = []

    def spy(metas, question, policy):
        seen.append(policy)
        return rank(metas, question, policy)

    monkeypatch.setattr(retrieval_service, "rank", spy)
    asyncio.run(
        make_service(reranker=Reranker({"Parent 1": 0.9})).retrieve(
            ["q"], question="q", date_from="2024-01-01", date_to="2024-12-31"
        )
    )
    assert seen == [
        RankingPolicy(
            relevance_floor=0.0,
            source_min_score=0.0,
            max_per_document=0,
            recency_types=(),
            recency_bonus_per_year=0.01,
            recency_max_bonus=0.05,
            date_from="2024-01-01",
            date_to="2024-12-31",
            in_range_bonus=0.2,
            in_range_by_year=True,
            question_years=True,
            no_recency_in_range=True,
        )
    ]


def test_retrieve_passes_today_to_the_ranking(monkeypatch) -> None:
    """The offline harness pins the ranking's clock; production passes nothing: the day of the request."""
    import rag_core.services.retrieval as module

    seen = []

    def capture(metas, question, policy):
        seen.append(policy.today)
        return list(range(len(metas))), {}, ""

    monkeypatch.setattr(module, "rank", capture)
    service = make_service()
    asyncio.run(service.retrieve(["q"], question="q", today=date(2026, 9, 29)))
    asyncio.run(service.retrieve(["q"], question="q"))
    assert seen == [date(2026, 9, 29), None]


class Legs:
    """Both legs over ten children whose semantic distance grows and whose keyword rank falls with the number."""

    def __init__(self, keyword: bool) -> None:
        self.keyword = keyword

    async def search(self, *args, **kwargs):
        key, retriever = ("fts_rank", "fts") if self.keyword else ("child_chunk_semantic_search_score", "semantic")
        return [
            Hit(
                page_content=f"child {i}",
                metadata={"section_id": f"p{i}", "retriever": retriever, key: (10 - i) if self.keyword else i / 10},
            )
            for i in range(1, 11)
        ]


class RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    async def rerank(self, queries, texts, timeout_s):
        self.calls.append((list(queries), list(texts)))
        return sorted(((0.9, t) for t in set(texts)), reverse=True)


TEN_PARENTS = Sections({f"p{i}": Section(f"p{i}", f"Parent {i}", {"filename": f"f{i}.pdf"}) for i in range(1, 11)})


def test_rerank_pairs_bounded() -> None:
    pre_rerank_k, rerank_queries = 4, 2
    reranker = RecordingReranker()
    service = make_service(
        Legs(False), Legs(True), TEN_PARENTS, reranker, pre_rerank_k=pre_rerank_k, rerank_queries=rerank_queries
    )
    asyncio.run(service.retrieve(["standalone", "analysis", "other"], question="q"))
    ((queries, texts),) = reranker.calls
    assert queries == ["standalone", "analysis"]  # the standalone question, then the first analysis query
    assert len(texts) == 2 * pre_rerank_k  # the cut children and their parents
    assert len(texts) * len(queries) <= 2 * pre_rerank_k * rerank_queries


def test_the_cut_keeps_the_best_fused_children_over_all_queries() -> None:
    reranker = RecordingReranker()
    service = make_service(Legs(False), Legs(True), TEN_PARENTS, reranker, pre_rerank_k=3, rerank_queries=1)
    asyncio.run(service.retrieve(["q1", "q2"], question="q"))
    # child 1 is the nearest and the best keyword match; the semantic and keyword order agree, so 1, 2 and 3 stay
    assert {t for t in reranker.calls[0][1] if t.startswith("child")} == {"child 1", "child 2", "child 3"}


def test_without_the_fusion_cut_every_child_and_query_reaches_the_reranker() -> None:
    reranker = RecordingReranker()
    service = make_service(
        Legs(False), Legs(True), TEN_PARENTS, reranker, pre_rerank_k=3, rerank_queries=1, fusion_cut=False
    )
    asyncio.run(service.retrieve(["q1", "q2"], question="q"))
    queries, texts = reranker.calls[0]
    assert queries == ["q1", "q2"] and len([t for t in texts if t.startswith("child")]) == 10


BELOW_THRESHOLD = {"Parent 1": 0.4, "Parent 2": 0.35, "Parent 3": 0.1, "Parent 4": 0.05}  # the threshold is 0.50


def test_min_docs_respects_relevance_floor() -> None:
    def kept(floor: float) -> list[str]:
        service = make_service(reranker=Reranker(BELOW_THRESHOLD), relevance_floor=floor)
        return [d.page_content for d in asyncio.run(service.retrieve(["q"], question="q"))]

    assert kept(0.0) == ["Parent 1", "Parent 2", "Parent 3"]  # min_docs 3 keeps three below the threshold
    assert kept(0.3) == ["Parent 1", "Parent 2"]  # a parent below the floor is not kept to reach min_docs


def test_no_docs_prompt_when_none_pass() -> None:
    service = make_service(reranker=Reranker(BELOW_THRESHOLD), relevance_floor=0.5)
    # nothing passes the threshold or the floor: no documents, so the answer node runs the no-docs prompt (with the
    # applied-filter line when a filter was set)
    assert asyncio.run(service.retrieve(["q"], filter=Filter(programme=("msc-informatik",)), question="q")) == []


def test_rerank_texts_not_cut_at_512() -> None:
    long_parent = "x" * 2000
    sections = Sections({"p1": Section("p1", long_parent, {"filename": "f1.pdf"})})

    def sent(whole_parents: bool, chars_per_token: float = 3.3) -> list[str]:
        reranker = RecordingReranker()
        service = make_service(
            Vector({"child one": "p1"}),
            None,
            sections,
            reranker,
            whole_parents=whole_parents,
            chars_per_token=chars_per_token,
        )
        asyncio.run(service.retrieve(["q"], question="q"))
        return reranker.calls[0][1]

    assert long_parent in sent(True)  # the model server's MAX_LENGTH decides the length, not the client
    # the switch off: the former 512-token cut, estimated at the card's chars_per_token (here 2.0)
    assert [len(t) for t in sent(False, 2.0) if t.startswith("x")] == [1024]


def test_min_docs_only_tops_up_above_the_floor() -> None:
    scores = {"Parent 1": 0.9, "Parent 2": 0.55, "Parent 3": 0.4, "Parent 4": 0.2}  # two pass the threshold of 0.50
    service = make_service(reranker=Reranker(scores), relevance_floor=0.6)  # a floor above the threshold
    kept = [d.page_content for d in asyncio.run(service.retrieve(["q"], question="q"))]
    assert kept == ["Parent 1", "Parent 2"]  # the threshold-passers stay; nothing below the floor is added
    service = make_service(reranker=Reranker(scores), relevance_floor=0.3)
    kept = [d.page_content for d in asyncio.run(service.retrieve(["q"], question="q"))]
    assert kept == ["Parent 1", "Parent 2", "Parent 3"]  # topped up to min_docs 3 with the one at or above the floor
