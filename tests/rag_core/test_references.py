"""The reference hop: the regex reading "§ n <regulation>" references, and the retrieval service adding the referenced
sections over fakes of its ports."""

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from rag_core.domain.filters import Filter
from rag_core.domain.references import Reference, find_references
from rag_core.domain.retrieval import Hit, Section
from rag_core.ports.retrieval import RerankError
from rag_core.services.retrieval import MAX_HOP_SECTIONS, RetrievalService
from rag_core.settings import load_profile
from rag_core.settings.profile import Profile
from rag_core.settings.sections import Timeouts

PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "example" / "profile.yaml"
TIMEOUTS = Timeouts(retrieval_s=10, parent_lookup_s=5, reranker_s=15)
PO = {"regulation": "po", "programme": ["bsc-informatik"], "po_version": [2023]}


def test_parses_paragraph_and_regulation() -> None:
    text = (
        "Die Anerkennung regelt § 22 APB. Zur Frist siehe gemäß § 5 Abs. 2 Satz 1 der Ausführungsbestimmungen, "
        "zur Meldung § 12 MHB sowie § 16b AufenthG und § 3 der Einschreibeordnung. Kein Bezug: § 7 Absatz 1 gilt."
    )
    assert find_references(text, PO) == [
        Reference("apb", "§ 22", None, None),
        Reference("po", "§ 5", ("bsc-informatik",), (2023,)),
        Reference("mhb", "§ 12", ("bsc-informatik",), (2023,)),
        Reference("residence", "§ 16b", None, None),
        Reference("eso", "§ 3", None, None),
    ]


def test_a_list_names_every_paragraph_and_a_year_the_version() -> None:
    assert find_references("Siehe § 5 und § 6 APB sowie § 7, § 8 Abs. 2 der ESO.", PO) == [
        Reference("apb", "§ 5", None, None),
        Reference("apb", "§ 6", None, None),
        Reference("eso", "§ 7", None, None),
        Reference("eso", "§ 8", None, None),
    ]
    assert find_references("Es gilt § 10 PO 2015 und § 3 MHB 2020.", PO) == [
        Reference("po", "§ 10", ("bsc-informatik",), (2015,)),
        Reference("mhb", "§ 3", ("bsc-informatik",), (2020,)),
    ]


def test_a_reference_sits_on_one_line() -> None:
    contents = "§ 1 Geltungsbereich\n§ 2 Prüfungsordnung\n§ 3\nAPB\n§ 4 Studienordnung"
    assert find_references(contents, PO) == []  # table-of-contents lines, and a word on the line after the paragraph
    assert find_references("Das folgt aus\n  § 2 Prüfungsordnung\ngenau.", PO) == []  # a line of the reference alone
    assert find_references("Das folgt aus § 2 Prüfungsordnung\ngenau.", PO) == [
        Reference("po", "§ 2", ("bsc-informatik",), (2023,))
    ]


def test_dieser_ordnung_is_same_document() -> None:
    own = {"regulation": "mhb", "programme": ["msc-informatik"], "po_version": [2020]}
    assert find_references("Ausnahmen nach § 5 dieser Ordnung bleiben unberührt, ebenso § 5 dieser Ordnung.", own) == [
        Reference("mhb", "§ 5", ("msc-informatik",), (2020,))
    ]
    apb = {"regulation": "apb"}  # a university-wide regulation has no programme or version
    assert find_references("Das regelt § 9 dieser Ordnung.", apb) == [Reference("apb", "§ 9", None, None)]
    assert find_references("Das regelt § 9 dieser Ordnung.", {}) == []  # a parent of no known regulation has no "own"


class Embedder:
    async def aembed_query(self, text: str) -> list[float]:
        return [0.0]


class Vector:
    async def search(self, vector, k, *, max_distance=None, filter=None):
        return [Hit(page_content="child one", metadata={"section_id": "p1", "retriever": "semantic"})]


class NoKeywords:
    async def search(self, query, k, filter=None):
        return []


class Store:
    """The section store: `parents` by id, and `by_ref` mapping (regulation, paragraph) to the sections found."""

    def __init__(self, parents: dict[str, Section], by_ref: dict[tuple[str, str], list[Section]], fail=False) -> None:
        self.parents, self.by_ref, self.fail, self.calls = parents, by_ref, fail, []

    async def get_many(self, section_ids, filter=None):
        return {i: self.parents[i] for i in section_ids if i in self.parents}

    async def sections_by_ref(self, regulation, paragraph, programme, po_version, filter=None):
        self.calls.append((regulation, paragraph, programme, po_version, filter))
        if self.fail:
            raise RuntimeError("db down")
        return self.by_ref.get((regulation, paragraph), [])

    async def ping(self) -> None:
        return None


class Scores:
    """The reranker: a fixed score per text (0.0 for others); `hop_fails` makes a call for the hop's texts raise."""

    def __init__(self, scores: dict[str, float], hop_fails: bool = False) -> None:
        self.scores, self.hop_fails, self.calls = scores, hop_fails, []

    async def rerank(self, queries, texts, timeout_s):
        self.calls.append((list(queries), list(texts)))
        if self.hop_fails and "child one" not in texts:
            raise RerankError("HTTPStatusError boom")
        return sorted(((self.scores.get(t, 0.0), t) for t in set(texts)), reverse=True)


def hop(
    by_ref, scores, text, *, switch=True, fail=False, hop_fails=False, filter=None, queries=("standalone question",)
) -> tuple[list[Hit], Store, Scores]:
    """Retrieve with one kept parent (a PO of the Bachelor Informatik 2023) that reads `text`."""
    profile = load_profile(Profile, PROFILE)
    switches = profile.retrieval.switches.model_copy(update={"reference_hop": switch})
    retrieval = profile.retrieval.model_copy(update={"switches": switches})
    store = Store({"p1": Section("p1", text, {"filename": "po.pdf", **PO})}, by_ref, fail)
    reranker = Scores({"child one": 0.9, text: 0.9, **scores}, hop_fails)
    service = RetrievalService(
        Embedder(), Vector(), NoKeywords(), store, reranker, retrieval, profile.ranking, TIMEOUTS, 8, 20, 1,
        chars_per_token=3.3,
    )  # fmt: skip
    docs = asyncio.run(service.retrieve(list(queries), filter, question="standalone question"))
    return docs, store, reranker


def section(n: str, text: str) -> Section:
    return Section(f"s{n}", text, {"filename": f"apb{n}.pdf", "regulation": "apb"})


def ids(docs: list[Hit]) -> list[Any]:
    return [d.metadata["section_id"] for d in docs]


def test_hop_sections_need_threshold() -> None:
    by_ref = {("apb", "§ 22"): [section("22", "Anerkennung")], ("po", "§ 5"): [section("5", "Fristen")]}
    scores = {"Anerkennung": 0.8, "Fristen": 0.2}  # the profile's threshold is 0.50
    docs, store, reranker = hop(by_ref, scores, "Siehe § 22 APB und § 5 der Ausführungsbestimmungen.")
    assert ids(docs) == ["p1", "s22"]
    assert docs[1].metadata["cross_encoder_score"] == 0.8 and docs[1].page_content == "Anerkennung"
    assert [c[:4] for c in store.calls] == [("apb", "§ 22", None, None), ("po", "§ 5", ("bsc-informatik",), (2023,))]
    assert reranker.calls[-1] == (
        ["standalone question"],
        ["Anerkennung", "Fristen"],
    )  # against the standalone question


def test_at_most_three_added() -> None:
    numbers = ["1", "2", "3", "4", "5"]
    by_ref = {("apb", f"§ {n}"): [section(n, f"Text {n}")] for n in numbers}
    scores = {f"Text {n}": 0.6 + int(n) / 100 for n in numbers}  # all above the threshold, 5 the best
    docs, _, _ = hop(by_ref, scores, "Siehe " + ", ".join(f"§ {n} APB" for n in numbers) + ".")
    assert MAX_HOP_SECTIONS == 3 and sorted(ids(docs)) == ["p1", "s3", "s4", "s5"]  # the three best scores


def test_the_switch_turns_the_hop_off_and_a_kept_section_is_not_added_again() -> None:
    text = "Siehe § 22 APB."
    by_ref = {("apb", "§ 22"): [section("22", "Anerkennung")]}
    docs, store, _ = hop(by_ref, {"Anerkennung": 0.9}, text, switch=False)
    assert ids(docs) == ["p1"] and store.calls == []
    docs, _, _ = hop(
        {("apb", "§ 22"): [Section("p1", text, {})]}, {}, text
    )  # the lookup finds the citing parent itself
    assert ids(docs) == ["p1"]


def test_a_failing_lookup_or_reranker_adds_nothing(caplog) -> None:
    caplog.set_level(logging.ERROR)
    by_ref = {("apb", "§ 22"): [section("22", "Anerkennung")]}
    docs, _, _ = hop(by_ref, {"Anerkennung": 0.9}, "Siehe § 22 APB.", fail=True)
    assert ids(docs) == ["p1"] and "reference lookup failed" in caplog.text
    caplog.clear()
    docs, _, _ = hop(by_ref, {"Anerkennung": 0.9}, "Siehe § 22 APB.", hop_fails=True)
    assert ids(docs) == ["p1"] and "reference rerank failed: RerankError" in caplog.text
    assert "boom" not in caplog.text  # the class only, never the reranker's message


def test_the_hop_is_scored_against_the_standalone_question_not_the_first_search_query() -> None:
    by_ref = {("apb", "§ 22"): [section("22", "Anerkennung")]}
    _, _, reranker = hop(by_ref, {"Anerkennung": 0.8}, "Siehe § 22 APB.", queries=("search rewrite", "other"))
    assert reranker.calls[-1] == (["standalone question"], ["Anerkennung"])


def test_the_question_filter_reaches_the_lookup_without_its_regulation() -> None:
    by_ref = {("apb", "§ 22"): [section("22", "Anerkennung")]}
    flt = Filter(degree=("bachelor",), programme=("bsc-informatik",), po_version=(2023,), regulation=("mhb",))
    docs, store, _ = hop(by_ref, {"Anerkennung": 0.8}, "Siehe § 22 APB.", filter=flt)
    assert ids(docs) == ["p1", "s22"]
    assert store.calls[0][4] == replace(flt, regulation=None)  # degree, programme and version still hold
