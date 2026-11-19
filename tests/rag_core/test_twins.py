"""Twins: sections that share a twin_key are the same text in two languages, and only one language is in the answer. The
German text wins for the regulations (it binds), any other pair follows the question's language."""

import asyncio
from pathlib import Path

from rag_core.domain.retrieval import Hit, Section
from rag_core.services.retrieval import RetrievalService, collapse_twins
from rag_core.settings import load_profile
from rag_core.settings.profile import Profile
from rag_core.settings.sections import Timeouts

PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "example" / "profile.yaml"


def hit(name: str, lang: str, regulation: str | None, twin_key: str | None = "t", score: float = 0.5) -> Hit:
    meta = {
        "filename": name,
        "lang": lang,
        "regulation": regulation,
        "twin_key": twin_key,
        "cross_encoder_score": score,
    }
    return Hit(page_content=name, metadata=meta)


def names(hits: list[Hit]) -> list[str]:
    return [h.page_content for h in hits]


def test_collapse_keeps_the_german_twin_of_a_regulation_whatever_the_question_language() -> None:
    for regulation in ("po", "mhb", "spp", "apb", "eso", "promotion"):
        twins = [hit("en", "en", regulation, score=0.9), hit("de", "de", regulation, score=0.2)]
        assert names(collapse_twins(twins, "en")) == ["de"], regulation


def test_collapse_follows_the_question_language_for_other_documents() -> None:
    twins = [hit("de", "de", "info", score=0.9), hit("en", "en", "info", score=0.2)]
    assert names(collapse_twins(twins, "en")) == ["en"]
    assert names(collapse_twins(twins, "de")) == ["de"]
    assert names(collapse_twins([hit("de", "de", None), hit("en", "en", None)], "en")) == ["en"]  # no regulation


def test_collapse_leaves_alone_what_has_no_twin_and_keeps_a_lone_twin() -> None:
    plain = [hit("a", "de", "po", twin_key=None), hit("b", "en", "po", twin_key=None), hit("c", "en", "info")]
    assert names(collapse_twins(plain, "de")) == ["a", "b", "c"]  # no key: no twin; the English one stays when alone


def test_collapse_keeps_every_section_of_the_wanted_language_since_a_pdf_has_one_key_for_all_its_sections() -> None:
    pdf = [hit(f"de{i}", "de", "info", "pdf", 0.1 * i) for i in range(3)] + [hit("en0", "en", "info", "pdf")]
    assert names(collapse_twins(pdf, "de")) == ["de0", "de1", "de2"]
    assert names(collapse_twins(pdf, "en")) == ["en0"]
    other = [hit("x", "de", "info", "k1"), *pdf]
    assert names(collapse_twins(other, "de")) == ["x", "de0", "de1", "de2"]  # the hits' order, other keys untouched


def test_a_key_is_binding_when_any_twin_is_labelled_a_regulation() -> None:
    # the twins' labels are derived apart: the English PDF's link text may miss the regulation patterns
    mixed = [hit("de", "de", "po", "k"), hit("en", "en", "info", "k"), hit("en-none", "en", None, "k")]
    for language in ("en", "de"):
        assert names(collapse_twins(mixed, language)) == ["de"], language
    assert names(collapse_twins(mixed[1:], "en")) == [
        "en",
        "en-none",
    ]  # the German twin was not found: the key stays whole
    unlabelled = [hit("de", "de", "info", "u"), hit("en", "en", None, "u")]
    assert names(collapse_twins(unlabelled, "en")) == ["en"]  # no label binds: the question's language


def test_collapse_keeps_the_twins_there_are_when_the_wanted_language_is_not_among_them() -> None:
    english = [hit("en1", "en", "apb", "k", 0.1), hit("en2", "en", "apb", "k", 0.6)]
    assert names(collapse_twins(english, "de")) == [
        "en1",
        "en2",
    ]  # the German text was not found: the English one stays
    assert names(collapse_twins(english[:1], "en")) == ["en1"]


class Index:
    """Both legs and the reranker over two twin sections and one plain section."""

    async def search(self, *args, **kwargs):
        return [Hit(page_content=f"child {i}", metadata={"section_id": f"s{i}"}) for i in ("de", "en", "info")]

    async def get_many(self, section_ids, filter=None):
        meta = {"regulation": "apb", "twin_key": "apb-twin"}
        sections = {
            "sde": Section("sde", "Parent de", {**meta, "lang": "de", "filename": "apb.de.pdf"}),
            "sen": Section("sen", "Parent en", {**meta, "lang": "en", "filename": "apb.en.pdf"}),
            "sinfo": Section("sinfo", "Parent info", {"lang": "en", "filename": "info.en.jsp"}),
        }
        return {i: sections[i] for i in section_ids}

    async def sections_by_ref(self, regulation, paragraph, programme, po_version, filter=None):
        return []

    async def ping(self) -> None:
        return None

    async def rerank(self, queries, texts, timeout_s):
        return [(0.9, t) for t in sorted(set(texts))]


def retrieve(question: str) -> list[Hit]:
    profile = load_profile(Profile, PROFILE)
    index = Index()
    timeouts = Timeouts(retrieval_s=10, parent_lookup_s=5, reranker_s=15)
    service = RetrievalService(
        None,
        index,
        index,
        index,
        index,
        profile.retrieval,
        profile.ranking,
        timeouts,
        8,
        20,
        1,  # type: ignore[arg-type]
        chars_per_token=3.3,
    )
    return asyncio.run(service.retrieve([question], question=question))


def test_twins_collapse_german_wins_for_regulations() -> None:
    for question in ("What does the APB say about repeating an exam?", "Wie oft darf ich eine Prüfung wiederholen?"):
        docs = retrieve(question)
        assert sorted(d.page_content for d in docs) == ["Parent de", "Parent info"], question
