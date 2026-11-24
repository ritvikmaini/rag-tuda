"""The retrieval-only eval: the gold's rank after fusion and after reranking, hit@k and MRR, the analysis asked once per
question, and the ablation rows; never a question or a document in the output."""

import argparse
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
from rag_core.domain.filters import Chips
from rag_core.domain.retrieval import Hit, Section
from rag_core.services.prompts import PromptService, load_profile_prompts
from rag_core.ports.retrieval import RerankError
from rag_core.services.retrieval import RetrievalService
from rag_core.settings import load_profile
from rag_core.settings.card import load_card
from rag_core.settings.profile import Profile
from rag_core.settings.sections import Timeouts

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "profiles" / "example" / "profile.yaml"
GOLD = {"url": "https://example.test/gesetz", "heading": "§ 3 Dauer des Urlaubs", "quote": "Der Urlaub beträgt"}


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
    """s2 is the gold: its url, and its text holds the quote (across a line break) and the heading line while its section path is
    only the document title, as a structure-chunked section of several paragraphs is. s3 holds the same quote under another url."""

    META = {
        "s1": {"url": GOLD["url"], "section_path": "Gesetz"},
        "s2": {"url": GOLD["url"] + "#page=2", "section_path": "Gesetz", "page_number": 2},
        "s3": {"url": "https://example.test/anderes", "section_path": "Anderes"},
    }
    TEXTS = {
        "s1": "Parent s1 ## § 1 Anspruch",
        "s2": "Parent s2\n## § 3 Dauer des Urlaubs\nDer\nUrlaub beträgt 24 Werktage.",
        "s3": "Parent s3 Der Urlaub beträgt 24 Werktage.",
    }

    async def get_many(self, section_ids, filter=None):
        return {i: Section(i, self.TEXTS[i], self.META[i]) for i in section_ids if i in self.META}

    async def sections_by_ref(self, *args, **kwargs):
        return []


class Reranker:
    """Scores the parent texts: the gold's section best, the children low."""

    SCORES = {"Parent s1": 0.6, "Parent s2": 0.95, "Parent s3": 0.1}

    def __init__(self, scores=None) -> None:
        self.scores = scores or self.SCORES

    async def rerank(self, queries, texts, timeout_s):
        return sorted(
            ((next((v for k, v in self.scores.items() if t.startswith(k)), 0.0), t) for t in set(texts)), reverse=True
        )


class DeadReranker:
    async def rerank(self, queries, texts, timeout_s):
        raise RerankError("down")


def service(profile: Profile, reranker=None) -> RetrievalService:
    return RetrievalService(
        Embedder(), Vector(), FullText(), Sections(), reranker or Reranker(), profile.retrieval, profile.ranking,
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

    def __init__(self, profile: Profile, reranker=None) -> None:
        self.profile, self.reranker, self.opened = profile, reranker, []

    def __call__(self, overrides):
        self.opened.append(dict(overrides))
        return self.open(overrides)

    @asynccontextmanager
    async def open(self, overrides):
        yield ev.Traced(service(ev.with_switches(self.profile, overrides), self.reranker))


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
        probe(
            "A5", "Wie viel Urlaub Kanarienvogel?", gold={"url": GOLD["url"], "heading": GOLD["heading"]}
        ),  # no quote
        probe("A6", "Ohne Adresse Seeigel?", gold={"quote": "Der Urlaub"}),  # no url: a bad gold
    ]
    (tmp_path / "probes" / "a.json").write_text(json.dumps({"conversations": convs}), encoding="utf-8")
    (tmp_path / "gate.yaml").write_text(
        yaml.safe_dump({"sets": {"set_a": {"file": "probes/a.json"}}}), encoding="utf-8"
    )
    return tmp_path


def run(profile_dir: Path, rows, deps=None, open_row=None, limit=None, header=""):
    deps = deps or make_deps()
    open_row = open_row or Rows(deps.profile)
    out = io.StringIO()
    code = asyncio.run(ev.evaluate(deps, open_row, profile_dir, None, limit, rows, out, header))
    return code, out.getvalue(), deps, open_row


def cells(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines()]


def table_of(text: str) -> str:
    """The text before the probe blocks: the notes and the table."""
    return text.partition("\nprobes (")[0]


def probes_of(text: str, label: str = "profile") -> dict[str, list[str]]:
    """The probe block of a row: probe name -> its other cells."""
    block = text.partition(f"\nprobes ({label})\n")[2].partition("\n\n")[0]
    return {line.split()[0]: line.split()[1:] for line in block.splitlines()[1:]}


class Clock:
    """A perf_counter whose k-th interval lasts the k-th of these seconds."""

    def __init__(self, *seconds: float) -> None:
        self.seconds, self.now, self.started = list(seconds), 0.0, False

    def __call__(self) -> float:
        self.started = not self.started
        if not self.started:
            self.now += self.seconds.pop(0) if self.seconds else 0.0  # the no-answer probes are timed too, unused
        return self.now


def test_hit_at_k_and_mrr(profile_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ranks = [1, 2, None, 7]
    assert ev.summarise(ranks) == {
        "hit@1": 0.25, "hit@3": 0.5, "hit@5": 0.5, "hit@10": 0.75, "mrr": (1 + 1 / 2 + 1 / 7) / 4,
    }  # fmt: skip
    assert ev.summarise([])["mrr"] == 0.0

    # the gold: the url without its fragment, then the section's text
    def section(url=GOLD["url"], text="", **metadata):
        return Section("s", text, {"url": url, **metadata})

    quote = "Der Urlaub beträgt"
    assert ev.is_gold(GOLD, section(GOLD["url"] + "#page=3", f"Davor. {quote} 24 Werktage."))
    assert ev.is_gold(GOLD, section(text="Der \n  Urlaub\tbeträgt")) and ev.is_gold(
        {**GOLD, "quote": "Der\nUrlaub  beträgt"}, section(text=quote)
    )
    assert not ev.is_gold(GOLD, section("https://example.test/x", quote))  # the same text under another url
    assert not ev.is_gold(
        GOLD, section(text="Ein anderer Absatz", section_path="G > § 3 Dauer des Urlaubs")
    )  # the quote decides
    # without a quote: the heading in the path or as a heading line of the text (a section path of the document title only)
    heading = {"url": GOLD["url"], "heading": "§ 3 Dauer des Urlaubs"}
    assert ev.is_gold(heading, section(section_path="G > § 3 Dauer des Urlaubs"))
    assert ev.is_gold(heading, section(text="Davor\n## § 3 Dauer des Urlaubs\nText", section_path="Gesetz"))
    assert not ev.is_gold(heading, section(text="## § 13 Dauer des Urlaubs\nText", section_path="Gesetz"))
    page = {"url": "u", "page": 4}
    assert ev.is_gold(page, section("u", page_number=4)) and not ev.is_gold(page, section("u", page_number=5))
    assert ev.is_gold({"url": "u"}, section("u"))  # no quote, heading or page: the url decides
    assert not ev.valid_gold({"quote": "x"}) and not ev.valid_gold("x") and ev.valid_gold({"url": "u"})
    # end to end: A1, A2 and A5 (a gold with no quote) hold the gold second among the fused sections and first after reranking; A4's is never found, A6's gold has no url
    monkeypatch.setattr(ev, "perf_counter", Clock(1.0, 2.0, 3.0, 4.5))  # the four retrievals that work
    code, text, _, _ = run(profile_dir, ev.PLAIN)
    assert code == 0
    assert cells(table_of(text)) == [
        ["1", "conversations", "without", "a", "gold", "left", "out"],
        ["1", "conversations", "with", "a", "bad", "gold", "left", "out"],
        ["stage", "n", "failed", "hit@1", "hit@3", "hit@5", "hit@10", "MRR", "median", "s", "max", "s"],
        ["after", "fusion", "4", "0", "0.00", "0.75", "0.75", "0.75", "0.375", "-", "-"],
        ["after", "reranking", "4", "0", "0.75", "0.75", "0.75", "0.75", "0.750", "2.50", "4.50"],
    ]  # the latency is of the whole retrieval, so only the last stage carries it


def test_a_gold_with_several_sources_ranks_by_its_rule() -> None:
    other = {"url": "https://example.test/anderes", "quote": "Der Urlaub beträgt"}
    sections = [
        Section(i, t, {"url": u})
        for i, t, u in (
            ("s3", "Der Urlaub beträgt", other["url"]),
            ("s0", "x", "y"),
            ("s1", "Der Urlaub beträgt 24", GOLD["url"]),
        )
    ]
    assert ev.rank_of(GOLD, sections) == 3  # one source, as before
    assert ev.rank_of({**GOLD, "also": [other]}, sections) == 1  # any: the best
    assert ev.rank_of({**GOLD, "also": [other], "rule": "all"}, sections) == 3  # all: the worst
    assert ev.rank_of({**GOLD, "also": [{"url": "https://example.test/fehlt"}], "rule": "all"}, sections) is None
    assert ev.valid_gold({**GOLD, "also": [other], "rule": "all"})
    assert not ev.valid_gold({**GOLD, "also": [{"quote": "x"}]}) and not ev.valid_gold({**GOLD, "rule": "some"})


def test_the_score_of_a_gold_with_several_sources_follows_its_rule() -> None:
    other = {"url": "https://example.test/anderes", "quote": "Der Urlaub beträgt"}
    unscored = {"url": "https://example.test/fehlt"}
    sections = [
        Section("s1", "Der Urlaub beträgt 24", {"url": GOLD["url"]}),
        Section("s2", "Der Urlaub beträgt", {"url": other["url"]}),
        Section("s3", "Der Urlaub beträgt 30", {"url": GOLD["url"]}),
        Section("s4", "x", {"url": unscored["url"]}),
    ]
    scores = {"s1": 0.2, "s2": 0.9, "s3": 0.6}  # s4 was not scored
    assert ev.gold_score(GOLD, sections, scores) == 0.6  # one source: the best of its sections
    assert ev.gold_score({**GOLD, "also": [other]}, sections, scores) == 0.9  # any: the best source
    assert ev.gold_score({**GOLD, "also": [other], "rule": "all"}, sections, scores) == 0.6  # all: the worst
    assert ev.gold_score({**GOLD, "also": [unscored]}, sections, scores) == 0.6  # any: the scored one
    assert ev.gold_score({**GOLD, "also": [unscored], "rule": "all"}, sections, scores) is None  # all: one unscored
    assert ev.gold_score(unscored, sections, scores) is None


def test_a_source_whose_quote_is_in_no_section_of_its_url_is_stale() -> None:
    texts = {
        GOLD["url"]: ["Davor.", "Der\nUrlaub  beträgt 24 Werktage."],
        "https://example.test/geaendert": ["Neuer Text"],
    }
    assert ev.without_stale(GOLD, texts) is GOLD  # the quote is there, across a line break
    gone = {"url": "https://example.test/geaendert", "quote": "Alter Text"}
    assert ev.without_stale({**GOLD, "url": gone["url"]}, texts) is None
    assert ev.without_stale({**GOLD, "url": "https://example.test/fehlt#a"}, texts) is None  # no section at all
    assert ev.without_stale({"url": gone["url"]}, texts) == {"url": gone["url"]}  # no quote: nothing to be stale
    # any: the stale source is dropped, the gold stays; all: one stale source makes the whole gold stale
    assert ev.without_stale({**GOLD, "also": [gone]}, texts) == GOLD
    assert ev.without_stale({**gone, "also": [GOLD]}, texts) == GOLD
    assert ev.without_stale({**GOLD, "also": [gone], "rule": "all"}, texts) is None


def test_stale_golds_are_counted_and_left_out_of_the_scores(profile_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    async def texts(urls):
        seen.append(list(urls))
        return {GOLD["url"]: ["Der Urlaub beträgt 24 Werktage."]}  # A4's url has no section

    deps = make_deps()
    out = io.StringIO()
    monkeypatch.setattr(ev, "perf_counter", Clock(1.0, 2.0, 3.0))
    asyncio.run(ev.evaluate(deps, Rows(deps.profile), profile_dir, None, None, ev.PLAIN, out, "", texts))
    assert seen == [["https://example.test/fehlt", GOLD["url"]]]  # one lookup, the urls once each, no fragment
    lines = cells(table_of(out.getvalue()))
    assert lines[2][:3] == ["1", "conversations", "with"] and "stale" in lines[2]
    assert lines[-1][:3] == ["after", "reranking", "3"]  # A1, A2 and A5; A4 is not scored as a miss


def test_the_chips_of_a_probe_reach_the_analysis_and_key_its_cache(
    profile_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chipped = probe("A7", "Wie viel Urlaub Kanarienvogel?")
    chipped["turns"][0]["chips"] = {"degree": "bachelor", "po": 2023}
    path = profile_dir / "probes" / "a.json"
    path.write_text(
        json.dumps({"conversations": [probe("A1", "Wie viel Urlaub Kanarienvogel?"), chipped]}), encoding="utf-8"
    )
    asked = []
    real = ev.Analyses.of

    async def of(self, question, chips=None):
        asked.append(chips)
        return await real(self, question, chips)

    monkeypatch.setattr(ev.Analyses, "of", of)
    deps = make_deps()
    run(profile_dir, ev.PLAIN, deps=deps)
    assert asked == [None, Chips(degree="bachelor", po=2023)]
    assert deps.analysis.calls == 2  # the same question with other chips is asked again  # type: ignore[attr-defined]


def test_analysis_cached_per_run(profile_dir: Path) -> None:
    code, text, deps, opened = run(profile_dir, ev.ABLATION)
    assert code == 0 and len(opened.opened) == 6
    # A1, A2 and A5 share a question, A4's and the no-answer probe A3's differ, A6 has no usable gold: three analysis calls
    assert deps.analysis.calls == 3  # type: ignore[attr-defined]
    for word in (
        "Kanarienvogel",
        "Tintenfisch",
        "Zaunkoenig",
        "Suchwort",
        "Standalone",
        "Parent",
        "Pinguin",
        "Seeigel",
    ):
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
    assert [line[0] for line in cells(table_of(text))[3:]] == ["no", "+", "+", "+", "+", "+"]
    assert table_of(text).count("\n") == 9  # the two notes, the head and six rows


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
    assert cells(table_of(text))[4][:5] == [
        "after",
        "reranking",
        "0",
        "4",
        "0.00",
    ]  # all four failed, none counted a miss
    assert {p[-1] for p in probes_of(text).values()} == {"error"} and len(
        probes_of(text)
    ) == 5  # the no-answer probe too


def test_the_table_comes_after_the_header_line(profile_dir: Path) -> None:
    deps = make_deps()
    retrieval = deps.profile.retrieval
    rt = SimpleNamespace(
        collection="example",
        settings=SimpleNamespace(profile=deps.profile, card=SimpleNamespace(name="s-card")),
    )
    line = ev.header_line(rt, 7)  # type: ignore[arg-type]
    assert line == (
        f"index example generation 7; embedder {retrieval.embedding.model}@{retrieval.embedding.revision}; "
        f"reranker {retrieval.rerank.model}@{retrieval.rerank.revision}; card s-card\n"
    )
    assert "generation unknown" in ev.header_line(rt, None)  # type: ignore[arg-type]
    _, text, _, _ = run(profile_dir, ev.PLAIN, header=line)
    assert text.startswith(line) and "Kanarienvogel" not in line


def test_only_models_that_are_the_pins_are_served() -> None:
    ok = {"embedding": "ok", "reranker": "ok", "llm": "ok"}
    assert ev.not_served(ok) == []
    assert ev.not_served({**ok, "llm": "remote, not probed", "analysis_llm": "ok"}) == []
    assert ev.not_served({**ok, "reranker": "mismatch: other/rerank@abc"}) == ["reranker: mismatch: other/rerank@abc"]
    assert ev.not_served({**ok, "embedding": "unavailable: HTTP 503"}) == ["embedding: unavailable: HTTP 503"]
    # the analysis falls back to the bare question when its model is down: the table would measure that
    assert ev.not_served({**ok, "llm": "unavailable: ConnectError"}) == ["llm: unavailable: ConnectError"]
    assert ev.not_served({**ok, "llm": "remote, not probed", "analysis_llm": "unavailable: HTTP 502"}) == [
        "analysis_llm: unavailable: HTTP 502"
    ]


def test_a_run_against_other_models_than_the_pins_stops_with_exit_1(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def reports(rt):
        return {"embedding": "ok", "reranker": "mismatch: other/rerank@abc", "llm": "ok"}

    # nothing past the check is reached: no index is opened, no table written
    monkeypatch.setattr(ev, "load_settings", lambda environ, secrets_dir=None: SimpleNamespace())
    monkeypatch.setattr(ev, "build_runtime", lambda *args: SimpleNamespace())
    monkeypatch.setattr(ev, "model_reports", reports)
    code = asyncio.run(ev.run(argparse.Namespace(sets=None, limit=None, ablation=False), {}, None))
    assert code == 1 and "reranker: mismatch: other/rerank@abc" in caplog.text


def test_the_generation_is_read_through_a_pool_read_pool_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def reports(rt):
        return {"embedding": "ok", "reranker": "ok", "llm": "ok"}

    built = []

    class Ctx:
        async def __aenter__(self):
            return "pool"

        async def __aexit__(self, *exc):
            return None

    def pool(*args, **kwargs):
        built.append(ev_read_pool(*args, **kwargs))  # the real constructor refuses a max_size below its minimum
        return Ctx()

    class Store:
        def __init__(self, pool):
            pass

        async def live_generation(self, collection):
            return 3

    async def no_index(*args):
        raise RuntimeError("past the generation")

    ev_read_pool = ev.read_pool
    settings = SimpleNamespace(overlay=SimpleNamespace(database=SimpleNamespace()), profile=None, acronyms=None)
    monkeypatch.setattr(ev, "load_settings", lambda environ, secrets_dir=None: settings)
    monkeypatch.setattr(ev, "build_runtime", lambda *args: SimpleNamespace(collection="example"))
    monkeypatch.setattr(ev, "model_reports", reports)
    monkeypatch.setattr(ev, "store_conninfo", lambda db: "host=h port=1 dbname=d user=u password=p")
    monkeypatch.setattr(ev, "read_pool", pool)
    monkeypatch.setattr(ev, "TurnStore", Store)
    monkeypatch.setattr(ev, "open_index", no_index)
    with pytest.raises(RuntimeError, match="past the generation"):
        asyncio.run(ev.run(argparse.Namespace(sets=None, limit=None, ablation=False), {}, None))
    assert len(built) == 1


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


def run_probes(tmp_path: Path, reranker, gold=GOLD, floor: float | None = None) -> dict[str, list[str]]:
    """The probe block of a run over one probe with a gold (G) and one without (N); a floor above the scores makes the
    retrieval decline."""
    (tmp_path / "probes").mkdir(parents=True)
    convs = [probe("G", "Frage Kanarienvogel?", gold), probe("N", "Frage Tintenfisch?", gold=None)]
    (tmp_path / "probes" / "a.json").write_text(json.dumps({"conversations": convs}), encoding="utf-8")
    (tmp_path / "gate.yaml").write_text(
        yaml.safe_dump({"sets": {"set_a": {"file": "probes/a.json"}}}), encoding="utf-8"
    )
    deps = make_deps()
    profile = deps.profile
    if floor is not None:
        profile = profile.model_copy(update={"ranking": profile.ranking.model_copy(update={"relevance_floor": floor})})
    out = io.StringIO()
    code = asyncio.run(ev.evaluate(deps, Rows(profile, reranker), tmp_path, None, None, ev.PLAIN, out))
    assert code == 0 and not {"Kanarienvogel", "Tintenfisch"} & set(out.getvalue().split())
    return probes_of(out.getvalue())


def test_a_no_answer_probe_with_an_empty_final_list_is_a_decline(tmp_path: Path) -> None:
    rows = run_probes(tmp_path, Reranker({"Parent s1": 0.1, "Parent s2": 0.3, "Parent s3": 0.2}), floor=0.4)
    assert rows["set_a/N"] == ["none", "0", "-", "-", "-", "0.300", "ok"]  # nothing above threshold or floor
    assert rows["set_a/G"][:3] == ["gold", "0", "2"]  # the gold is the second fused section and is not returned


def test_a_no_answer_probe_with_a_non_empty_final_list_shows_its_top_score(tmp_path: Path) -> None:
    rows = run_probes(tmp_path, Reranker())
    assert rows["set_a/N"] == [
        "none",
        "3",
        "-",
        "-",
        "-",
        "0.950",
        "ok",
    ]  # s1 and s2 pass the threshold, min_docs adds s3


def test_the_gold_above_and_below_another_document(tmp_path: Path) -> None:
    above = run_probes(tmp_path / "above", Reranker())
    assert above["set_a/G"] == ["gold", "3", "2", "1", "0.950", "0.950", "ok"]  # first after reranking, the best score
    below = run_probes(tmp_path / "below", Reranker({"Parent s1": 0.7, "Parent s2": 0.3, "Parent s3": 0.1}))
    assert below["set_a/G"] == ["gold", "3", "2", "2", "0.300", "0.700", "ok"]  # under the threshold, kept by min_docs


def test_a_failed_rerank_is_no_score_evidence(tmp_path: Path) -> None:
    rows = run_probes(tmp_path, DeadReranker())
    assert rows["set_a/G"] == [
        "gold",
        "3",
        "2",
        "2",
        "-",
        "-",
        "rerank_failed",
    ]  # the order is kept, the 0.0 scores hidden
    assert rows["set_a/N"] == ["none", "3", "-", "-", "-", "-", "rerank_failed"]
