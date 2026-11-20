"""The retrieval-only eval: the gold's rank after fusion and after reranking, hit@k and MRR, the analysis asked once per
question, and the ablation rows; never a question or a document in the output."""

import asyncio
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from assistant_api import retrieval_eval as ev
from assistant_api.graph.state import GraphDeps
from rag_core.domain.retrieval import Hit, Section
from rag_core.services.prompts import PromptService, load_profile_prompts
from rag_core.services.retrieval import RetrievalService
from rag_core.settings import load_profile
from rag_core.settings.card import load_card
from rag_core.settings.profile import Profile
from rag_core.settings.sections import Timeouts

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "profiles" / "example" / "profile.yaml"
GOLD = {"url": "https://example.test/gesetz", "heading": "§ 3 Dauer", "quote": "Der Urlaub"}


class Analysis:
    """The analysis model: counts its calls, answers with a standalone text and queries that repeat the question."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        return json.dumps({"intent": "lookup", "standalone_question": "Standalone", "search_queries": ["Suchwort"]})


class Embedder:
    async def aembed_query(self, text):
        return [0.0]


class Vector:
    """Three children of three sections, the gold's second."""

    async def search(self, vector, k, *, max_distance=None, filter=None):
        return [Hit(page_content=f"child {i}", metadata={"section_id": f"s{i}"}) for i in (1, 2, 3)]


class FullText:
    async def search(self, query, k, filter=None):
        return []


class Sections:
    """s2 is the gold (url and heading); s1 and s3 are other sections of the same document and of another one."""

    META = {
        "s1": {"url": GOLD["url"], "section_path": "Gesetz > § 1 Anspruch"},
        "s2": {"url": GOLD["url"] + "#page=2", "section_path": "Gesetz > § 3 Dauer des Urlaubs", "page_number": 2},
        "s3": {"url": "https://example.test/anderes", "section_path": "Anderes > § 3 Dauer"},
    }

    async def get_many(self, section_ids, filter=None):
        return {i: Section(i, f"Parent {i}", self.META[i]) for i in section_ids if i in self.META}

    async def sections_by_ref(self, *args, **kwargs):
        return []


class Reranker:
    """Scores the parent texts: the gold's section best, the children low."""

    SCORES = {"Parent s1": 0.6, "Parent s2": 0.95, "Parent s3": 0.1}

    async def rerank(self, queries, texts, timeout_s):
        return sorted(((self.SCORES.get(t, 0.0), t) for t in set(texts)), reverse=True)


def service(profile: Profile) -> RetrievalService:
    return RetrievalService(
        Embedder(), Vector(), FullText(), Sections(), Reranker(), profile.retrieval, profile.ranking,
        Timeouts(retrieval_s=10, parent_lookup_s=5, reranker_s=15), 8, 20, 1, chars_per_token=3.3,
    )  # fmt: skip


def make_deps() -> GraphDeps:
    profile = load_profile(Profile, PROFILE)
    return GraphDeps(
        profile=profile,
        card=load_card(ROOT / "deploy" / "env" / "ci-fake-card.toml"),
        messages=SimpleNamespace(),  # type: ignore[arg-type]
        prompts=PromptService(load_profile_prompts(PROFILE.parent / "prompts")),
        analysis=Analysis(),  # type: ignore[arg-type]
        answer_llm=None,
        counter=None,  # type: ignore[arg-type]
        retrieval=None,  # type: ignore[arg-type]
    )


class Rows:
    """open_row over the fake service: remembers the switches each row was opened with."""

    def __init__(self, profile: Profile) -> None:
        self.profile, self.opened = profile, []

    def __call__(self, overrides):
        self.opened.append(dict(overrides))
        return self.open(overrides)

    @asynccontextmanager
    async def open(self, overrides):
        yield ev.Traced(service(ev.with_switches(self.profile, overrides)))


def probe(cid, question, gold=GOLD):
    turn = {"q": question, **({"gold": gold} if gold else {})}
    return {"id": cid, "turns": [turn, {"q": "Zweite Frage Pinguin"}]}


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    (tmp_path / "probes").mkdir()
    convs = [
        probe("A1", "Wie viel Urlaub Kanarienvogel?"),
        probe("A2", "Wie viel Urlaub Kanarienvogel?"),  # the same question as A1
        probe("A3", "Wer ist Tintenfisch?", gold=None),
        probe("A4", "Anderes Zaunkoenig?", gold={**GOLD, "url": "https://example.test/fehlt"}),
    ]
    (tmp_path / "probes" / "a.json").write_text(json.dumps({"conversations": convs}), encoding="utf-8")
    (tmp_path / "gate.yaml").write_text(
        yaml.safe_dump({"sets": {"set_a": {"file": "probes/a.json"}}}), encoding="utf-8"
    )
    return tmp_path


def run(profile_dir: Path, rows, deps=None, open_row=None, limit=None):
    deps = deps or make_deps()
    open_row = open_row or Rows(deps.profile)
    out = io.StringIO()
    code = asyncio.run(ev.evaluate(deps, open_row, profile_dir, None, limit, rows, out))
    return code, out.getvalue(), deps, open_row


def cells(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines()]


def test_hit_at_k_and_mrr(profile_dir: Path) -> None:
    ranks = [1, 2, None, 7]
    assert ev.summarise(ranks) == {
        "hit@1": 0.25, "hit@3": 0.5, "hit@5": 0.5, "hit@10": 0.75, "mrr": (1 + 1 / 2 + 1 / 7) / 4,
    }  # fmt: skip
    assert ev.summarise([])["mrr"] == 0.0
    # the gold: url without its fragment, then the heading in the section path or the page
    assert ev.is_gold(GOLD, {"url": GOLD["url"] + "#page=3", "section_path": "G > § 3 Dauer des Urlaubs"})
    assert not ev.is_gold(GOLD, {"url": GOLD["url"], "section_path": "G > § 13 Dauer"})
    assert not ev.is_gold(GOLD, {"url": "https://example.test/x", "section_path": "§ 3 Dauer"})
    page = {"url": "u", "page": 4}
    assert ev.is_gold(page, {"url": "u", "page_number": 4}) and not ev.is_gold(page, {"url": "u", "page_number": 5})
    assert ev.is_gold({"url": "u"}, {"url": "u"})  # no heading, no page: the url decides
    # end to end: A1 and A2 hold the gold second among the fused sections and first after reranking; A4's is never found
    code, text, _, _ = run(profile_dir, ev.PLAIN)
    assert code == 0
    assert cells(text) == [
        ["1", "conversations", "without", "a", "gold", "left", "out"],
        ["stage", "n", "failed", "hit@1", "hit@3", "hit@5", "hit@10", "MRR"],
        ["after", "fusion", "3", "0", "0.00", "0.67", "0.67", "0.67", "0.333"],
        ["after", "reranking", "3", "0", "0.67", "0.67", "0.67", "0.67", "0.667"],
    ]


def test_analysis_cached_per_run(profile_dir: Path) -> None:
    code, text, deps, opened = run(profile_dir, ev.ABLATION)
    assert code == 0 and len(opened.opened) == 6
    # A1 and A2 share a question, A4's differs: two analysis calls for six rows of three conversations
    assert deps.analysis.calls == 2  # type: ignore[attr-defined]
    for word in ("Kanarienvogel", "Tintenfisch", "Zaunkoenig", "Suchwort", "Standalone", "Parent", "Pinguin"):
        assert word not in text, word  # no question, query or document text in the table


def test_ablation_turns_switches_off_one_by_one(profile_dir: Path) -> None:
    assert [label for label, _ in ev.ABLATION] == [
        "no filters", "+ filters", "+ acronyms", "+ fusion cut", "+ whole parents", "+ reference hop",
    ]  # fmt: skip
    assert all(list(switches) == list(ev.ORDER) for _, switches in ev.ABLATION)
    assert not any(ev.ABLATION[0][1].values()) and all(ev.ABLATION[-1][1].values())
    for (_, before), (_, after) in zip(ev.ABLATION, ev.ABLATION[1:]):
        assert [k for k in ev.ORDER if before[k] != after[k]] == [ev.ORDER[sum(before.values())]]  # one more, in order
        assert not any(before[k] and not after[k] for k in ev.ORDER)  # never one off again
    # the rows reach the service and the profile
    _, text, _, rows = run(profile_dir, ev.ABLATION)
    assert rows.opened == [switches for _, switches in ev.ABLATION]
    profile = ev.with_switches(make_deps().profile, ev.ABLATION[1][1])
    assert profile.retrieval.switches.filters and not profile.retrieval.switches.fusion_cut
    assert [line[0] for line in cells(text)[2:]] == ["no", "+", "+", "+", "+", "+"]
    assert text.count("\n") == 8  # the note, the head and six rows


def test_a_conversation_without_a_gold_or_a_failing_one_does_not_count_as_a_miss(profile_dir: Path) -> None:
    deps = make_deps()

    class Down(Rows):
        @asynccontextmanager
        async def open(self, overrides):
            traced = ev.Traced(service(ev.with_switches(self.profile, overrides)))

            async def fail(*args, **kwargs):
                raise ConnectionError("down: Kanarienvogel")

            traced.service.retrieve = fail  # type: ignore[method-assign]
            yield traced

    code, text, _, _ = run(profile_dir, ev.PLAIN, deps=deps, open_row=Down(deps.profile))
    assert code == 1 and "Kanarienvogel" not in text
    assert cells(text)[3][:5] == ["after", "reranking", "0", "3", "0.00"]  # all three failed, none counted a miss


def test_the_example_golds_are_in_their_documents() -> None:
    probes = json.loads((ROOT / "profiles" / "example" / "probes" / "smoke.json").read_text(encoding="utf-8"))
    base = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))["ingest"]["sources"]["gesetze"]["base_url"]
    golds = {c["id"]: g for c in probes["conversations"] if (g := ev.gold_of(c))}
    assert sorted(golds) == ["E01", "E02", "E04", "E05"]  # the joke has none
    for gold in golds.values():
        text = (ROOT / "profiles" / "example" / "documents" / (gold["url"].removeprefix(base) + ".md")).read_text(
            encoding="utf-8"
        )
        assert f"## {gold['heading']}\n" in text and gold["quote"] in text and len(gold["quote"]) <= 200
