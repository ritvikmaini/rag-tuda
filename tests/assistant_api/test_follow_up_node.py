"""The iterative round: one extra node after the first search, run only for a question the analysis marked as needing more,
on a card that gates it on, with the profile's switch on. Its follow-up queries come from the analysis model's own call; the
search that follows covers the round-one queries and the new ones together."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from assistant_api.graph import streaming as ss
from assistant_api.graph.build import NODES, build_graph
from assistant_api.graph.history import question_message
from assistant_api.graph.nodes.follow_up import make_follow_up
from assistant_api.graph.state import GraphDeps
from assistant_api.settings import Messages
from rag_core.domain.filters import Filter
from rag_core.services.prompts import PromptService, load_profile_prompts
from rag_core.settings import load_profile
from rag_core.settings.card import Gated, load_card
from rag_core.settings.profile import Profile

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "profiles" / "example"
TU = ROOT / "profiles" / "tu-informatik"
CARD = ROOT / "deploy" / "env" / "ci-fake-card.toml"

ROUND_ONE = ["Frage?", "Ordnung"]


class Analysis:
    def __init__(self, needs_more: bool, intent: str = "lookup") -> None:
        self.reply = {
            "intent": intent,
            "standalone_question": "Frage?",
            "search_queries": ["Ordnung"],
            "needs_more": needs_more,
        }

    async def complete(self, messages):
        return json.dumps(self.reply)


class FollowUp:
    """The follow-up call's model: a scripted reply, or an exception."""

    def __init__(self, reply: Any) -> None:
        self.reply, self.calls = reply, []

    async def complete(self, messages):
        self.calls.append(list(messages))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def asked(queries, done=False):
    return {"reasoning": "Es fehlt die zweite Ordnung.", "queries": queries, "done": done}


class Counter:
    max_model_len, exact = 8192, True

    async def count_text(self, text: str) -> int:
        return len(text) // 2 + 1

    async def count_chat(self, messages) -> int:
        return sum(len(m["content"]) // 2 + 8 for m in messages) + 8


def source(name: str, score: float) -> Document:
    return Document(
        page_content=f"Text {name}",
        metadata={
            "filename": f"{name}.pdf",
            "section_id": name,
            "title": f"Ordnung {name}",
            "section_path": f"§ 1 {name}",
            "matched_child_text": f"Treffer {name}",
            "cross_encoder_score": score,
            "url": f"https://x/{name}.pdf",
        },
    )


class Retrieval:
    """Round one finds section a; a search that carries more queries finds b and a, best first (the service's joint rerank)."""

    def __init__(self, fail_second: bool = False) -> None:
        self.calls, self.fail_second = [], fail_second

    async def retrieve(
        self, queries, filter=None, *, question="", date_from=None, date_to=None, principal=None, language=None
    ):
        self.calls.append({"queries": list(queries), "filter": filter, "question": question})
        if len(self.calls) == 1:
            return [source("a", 0.7)]
        if self.fail_second:
            raise RuntimeError("retrieval unavailable: every query failed")
        return [source("b", 0.95), source("a", 0.9)]


def deps(
    needs_more: bool = True,
    *,
    follow_up: Any = None,
    gated: bool = True,
    switch: bool = True,
    intent: str = "lookup",
    retrieval: Any = None,
) -> GraphDeps:
    profile = load_profile(Profile, EXAMPLE / "profile.yaml")
    switches = profile.retrieval.switches.model_copy(update={"iterative_round": switch})
    profile = profile.model_copy(update={"retrieval": profile.retrieval.model_copy(update={"switches": switches})})
    card = load_card(CARD)
    card = card.model_copy(update={"gated": Gated(iterative_retrieval=gated, version_table=False)})
    messages = Messages.model_validate(yaml.safe_load((EXAMPLE / "messages.de.yaml").read_text(encoding="utf-8")))
    return GraphDeps(
        profile=profile,
        card=card,
        messages=messages,
        prompts=PromptService(load_profile_prompts(TU / "prompts")),
        analysis=Analysis(needs_more, intent),
        answer_llm=GenericFakeChatModel(messages=iter([AIMessage(content="Die Antwort [1].")])),
        counter=Counter(),
        retrieval=retrieval or Retrieval(),
        follow_up=follow_up,
    )


def inputs() -> dict:
    return {
        "messages": [question_message("Frage?", None, datetime.now(timezone.utc))],
        "variant": None,
        "formats": "",
    }


def run(d: GraphDeps) -> dict:
    return asyncio.run(build_graph(d).ainvoke(inputs(), config={"configurable": {"thread_id": "t"}}))


def events(d: GraphDeps) -> list[dict]:
    async def collect() -> list[dict]:
        graph = build_graph(d)
        config = {"configurable": {"thread_id": "t"}}
        return [e async for e in graph.astream_events(inputs(), config=config, version="v2")]

    return asyncio.run(collect())


def test_round_runs_only_when_needs_more_and_gated() -> None:
    follow_up = FollowUp(asked(["Prüfungsordnung Frist"]))
    retrieval = Retrieval()
    out = run(deps(follow_up=follow_up, retrieval=retrieval))
    assert len(follow_up.calls) == 1 and len(retrieval.calls) == 2
    assert out["follow_up_queries"] == ["Prüfungsordnung Frist"]
    # the prompt carries the question, the queries already searched and the kept section's header and matched child
    prompt = "\n".join(text for _, text in follow_up.calls[0])
    assert "Frage?" in prompt and "- Ordnung" in prompt
    assert "Ordnung a" in prompt and "§ 1 a" in prompt and "Treffer a" in prompt
    assert "Text a" not in prompt  # not the section's whole text

    # each condition alone keeps the graph as it was: no follow-up call, one search, no node
    for label, case in {
        "analysis does not ask for more": dict(needs_more=False),
        "card does not gate it": dict(gated=False),
        "profile switch off": dict(switch=False),
        "not a searching intent": dict(intent="out_of_scope"),
    }.items():
        follow_up, retrieval = FollowUp(asked(["x"])), Retrieval()
        started = [
            e["name"]
            for e in events(deps(**{"follow_up": follow_up, "retrieval": retrieval, **case}))
            if e["event"] == "on_chain_start"
        ]
        assert follow_up.calls == [] and len(retrieval.calls) <= 1 and "follow_up_search" not in started, label
    # the profile ships no follow-up prompt, so no client is built: nothing to run
    retrieval = Retrieval()
    run(deps(follow_up=None, retrieval=retrieval))
    assert len(retrieval.calls) == 1


def test_one_extra_round_max() -> None:
    # more than two queries are cut to two, and the round is one follow-up call and one extra search whatever the model says
    follow_up, retrieval = FollowUp(asked(["eins", "zwei", "drei"], done=False)), Retrieval()
    run(deps(follow_up=follow_up, retrieval=retrieval))
    assert len(follow_up.calls) == 1 and len(retrieval.calls) == 2
    assert retrieval.calls[1]["queries"] == [*ROUND_ONE, "eins", "zwei"]
    # done, no queries, a query round one already searched, or a reply that is not JSON: round one's documents stay
    for reply in (asked(["eins"], done=True), asked([]), asked(["frage?"]), "keine Antwort", RuntimeError("timeout")):
        follow_up, retrieval = FollowUp(reply), Retrieval()
        out = run(deps(follow_up=follow_up, retrieval=retrieval))
        assert len(follow_up.calls) == 1 and len(retrieval.calls) == 1, reply
        assert [d.metadata["section_id"] for d in out["raw_docs"]] == ["a"] and out["generation"], reply
        assert out["follow_up_queries"] == [], reply


def test_hits_reranked_with_round_one() -> None:
    follow_up, retrieval = FollowUp(asked(["Prüfungsordnung Frist"])), Retrieval()
    out = run(deps(follow_up=follow_up, retrieval=retrieval))
    first, second = retrieval.calls
    # one search over the round-one queries first (so the reranker keeps scoring against them) and the new ones after
    assert first["queries"] == ROUND_ONE and second["queries"] == [*ROUND_ONE, "Prüfungsordnung Frist"]
    assert second["question"] == first["question"] == "Frage?"
    # the service returns the joint ranking, already cut to final_k: it replaces round one's documents
    assert [d.metadata["section_id"] for d in out["raw_docs"]] == ["b", "a"]
    assert {s["section_id"] for s in out["context_sources"]} == {"a", "b"}  # both reach the answer context


def test_the_extra_search_keeps_the_turns_filter() -> None:
    d = deps(follow_up=FollowUp(asked(["zwei"])))
    state = {
        "messages": [question_message("Frage?", None, datetime.now(timezone.utc))],
        "raw_docs": [source("a", 0.7)],
        "retrieval_queries": ROUND_ONE,
        "standalone_question": "Frage?",
        "filter": Filter(("bachelor",), ("bsc-informatik",), (2015,), None, None, True, "chip"),
        "date_from": "2026-01-01",
        "date_to": "2026-12-31",
    }
    asyncio.run(make_follow_up(d)(state, {"configurable": {}}))
    call = d.retrieval.calls[0]
    assert call["filter"] is state["filter"] and call["queries"] == [*ROUND_ONE, "zwei"]


def test_a_failing_extra_search_keeps_round_one() -> None:
    retrieval = Retrieval(fail_second=True)
    out = run(deps(follow_up=FollowUp(asked(["zwei"])), retrieval=retrieval))
    assert len(retrieval.calls) == 2 and [d.metadata["section_id"] for d in out["raw_docs"]] == ["a"]
    assert out["generation"] and out["follow_up_queries"] == []


def test_status_step_streamed() -> None:
    d = deps(follow_up=FollowUp(asked(["Prüfungsordnung Frist"])))
    steps = [f.content["step"] for f in map(ss.status_frame, events(d)) if f is not None]
    assert steps == ["analysing", "understood", "searching", "searching_more", "found", "writing"]
    plain = [f.content["step"] for f in map(ss.status_frame, events(deps(needs_more=False))) if f is not None]
    assert plain == ["analysing", "understood", "searching", "found", "writing"]
    start = {"event": "on_chain_start", "name": "follow_up_search", "data": {"input": {}}}
    assert ss.status_frame(start).content == {"step": "searching_more"}
    assert "follow_up_search" in NODES
