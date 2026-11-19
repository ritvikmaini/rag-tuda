"""The graph: its nodes, edges and run names, the answer streaming through the generate node from inside its closure
(LangChain's callbacks reach astream_events), and the two partial-update nodes."""

import asyncio
import dataclasses
import types
import uuid
from typing import Any
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import yaml
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from assistant_api.graph.build import NODES, build_graph
from assistant_api.graph.history import documents, history_messages, question_message, variant_inputs
from assistant_api.graph.nodes.format_docs import make_format_docs
from assistant_api.graph.nodes.retrieve import make_retrieve
from assistant_api.chat.service import pipeline
from assistant_api.graph.state import GraphDeps
from assistant_api.sse import frames as fr
from assistant_api.settings import Messages
from rag_core.domain.access import ANONYMOUS, Principal
from rag_core.domain.filters import Filter
from rag_core.domain.history import PastTurn, VariantInput, context_turns
from rag_core.services.prompts import PromptService, load_profile_prompts
from rag_core.settings import load_profile
from rag_core.settings.card import load_card
from rag_core.settings.profile import Profile

PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "example" / "profile.yaml"
CARD = Path(__file__).resolve().parents[2] / "deploy" / "env" / "ci-fake-card.toml"


class Analysis:
    def __init__(self, intent: str) -> None:
        self.intent = intent

    async def complete(self, messages):
        return '{"intent": "%s", "standalone_question": "Frage?", "search_queries": ["Frage"]}' % self.intent


class Counter:
    max_model_len, exact = 8192, True

    async def count_text(self, text: str) -> int:
        return len(text) // 2 + 1

    async def count_chat(self, messages) -> int:
        return sum(len(m["content"]) // 2 + 8 for m in messages) + 8


SEEN: list[Any] = []  # the principals the retrieval was asked for
FILTERS: list[Any] = []  # the filters it was asked for


class Retrieval:
    async def retrieve(self, queries, filter=None, *, question="", date_from=None, date_to=None, principal=None):
        SEEN.append(principal)
        FILTERS.append(filter)
        return [
            Document(
                page_content="Quelle eins",
                metadata={
                    "filename": "a.pdf",
                    "section_id": "p1",
                    "cross_encoder_score": 0.9,
                    "url": "https://x/a.pdf",
                },
            )
        ]


def deps(intent: str) -> GraphDeps:
    import yaml

    profile = load_profile(Profile, PROFILE)
    messages = Messages.model_validate(
        yaml.safe_load((PROFILE.parent / "messages.de.yaml").read_text(encoding="utf-8"))
    )
    return GraphDeps(
        profile=profile,
        card=load_card(CARD),
        messages=messages,
        prompts=PromptService(load_profile_prompts(PROFILE.parent / "prompts")),
        analysis=Analysis(intent),
        answer_llm=GenericFakeChatModel(messages=iter([AIMessage(content="Die Antwort")])),
        counter=Counter(),
        retrieval=Retrieval(),
    )


async def events(intent: str) -> list[dict]:
    graph = build_graph(deps(intent))
    inputs = {
        "messages": [question_message("Frage?", None, datetime.now(timezone.utc))],
        "variant": None,
        "formats": "",
    }
    return [e async for e in graph.astream_events(inputs, config={"configurable": {"thread_id": "t"}}, version="v2")]


def test_the_graph_has_its_nodes_edges_and_run_names() -> None:
    graph = build_graph(deps("lookup"))
    drawn = graph.get_graph()
    assert set(drawn.nodes) == {"__start__", "__end__", *NODES}
    assert {(e.source, e.target) for e in drawn.edges} == {
        ("__start__", "generate"),
        ("__start__", "rewrite_follow_up_question"),
        ("rewrite_follow_up_question", "get_docs"),
        ("rewrite_follow_up_question", "generate"),
        ("get_docs", "format_docs"),
        ("get_docs", "generate"),
        ("format_docs", "generate"),
        ("generate", "__end__"),
    }
    starts = [e["name"] for e in asyncio.run(events("lookup")) if e["event"] == "on_chain_start"]
    assert starts == [
        "LangGraph",
        "__start__",
        "entry_route",
        "rewrite_follow_up_question",
        "IntentRouter",
        "get_docs",
        "CheckDocsFoundRouter",
        "format_docs",
        "generate",
    ]


def test_the_answer_streams_through_generate() -> None:
    for intent in ("lookup", "conversation"):
        chunks = [
            e["data"]["chunk"].content
            for e in asyncio.run(events(intent))
            if e["event"] == "on_chat_model_stream" and e["metadata"].get("langgraph_node") == "generate"
        ]
        assert "".join(chunks) == "Die Antwort", intent


def test_a_failing_model_call_logs_its_class_never_the_message(caplog) -> None:
    """vLLM's error body can echo the input; the line names the class (the node re-raises)."""
    import dataclasses

    class Failing:
        async def ainvoke(self, prompt):
            raise ValueError("input_value='Personalnummer 4711'")

    graph = build_graph(dataclasses.replace(deps("lookup"), answer_llm=Failing()))
    inputs = {
        "messages": [question_message("Frage?", None, datetime.now(timezone.utc))],
        "variant": None,
        "formats": "",
    }
    caplog.set_level("ERROR")
    with pytest.raises(ValueError):  # the node re-raises after it logged
        asyncio.run(graph.ainvoke(inputs, config={"configurable": {"thread_id": "t"}}))
    assert "Error during LLM invocation: ValueError" in [r.getMessage() for r in caplog.records]
    assert "4711" not in " ".join(r.getMessage() for r in caplog.records)


def provider_error(status: int, retry_after: str | None = None, code: str | None = None):
    """The exception the OpenAI client raises for a reply with this status (its body text is never read)."""
    import httpx
    import openai

    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(status, headers=headers, request=httpx.Request("POST", "http://llm.test/v1"))
    classes = {400: openai.BadRequestError, 429: openai.RateLimitError, 500: openai.InternalServerError}
    cls = classes.get(status, openai.APIStatusError)
    return cls("text that is never read", response=response, body={"code": code, "message": "x"})


class Raising:
    """The answer model: fails with each exception of the list in turn, then answers with usage."""

    def __init__(self, *errors: Exception) -> None:
        self.errors, self.calls = list(errors), 0

    async def ainvoke(self, prompt):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return AIMessage(
            content="Die Antwort", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
        )


def failed_run(llm, deadline_in: float | None = 60.0, card_update: dict | None = None) -> dict:
    import dataclasses
    import time

    base = deps("lookup")
    card = base.card.model_copy(update=card_update or {})
    graph = build_graph(dataclasses.replace(base, answer_llm=llm, card=card))
    inputs = {
        "messages": [question_message("Frage?", None, datetime.now(timezone.utc))],
        "variant": None,
        "formats": "",
    }
    deadline = None if deadline_in is None else time.monotonic() + deadline_in
    config = {"configurable": {"thread_id": "t", "deadline_at": deadline}}
    return asyncio.run(graph.ainvoke(inputs, config=config))


def test_context_length_error_maps_to_trim(caplog) -> None:
    """By status and code, never by text: the trim text asks the user to shorten the question."""
    texts = load_messages().answer
    caplog.set_level("ERROR")
    for error in (provider_error(413), provider_error(400)):  # the harness card is vLLM's: its 400 is the context error
        assert failed_run(Raising(error))["generation"] == texts.context_too_long
    groq = load_card(CARD).answer.model_copy(update={"provider": "groq"})
    assert (
        failed_run(Raising(provider_error(400, code="context_length_exceeded")), card_update={"answer": groq})[
            "generation"
        ]
        == texts.context_too_long
    )
    with pytest.raises(Exception, match="never read"):  # a rejected parameter at Groq is no context error
        failed_run(Raising(provider_error(400)), card_update={"answer": groq})
    assert "[ContextBudget] request rejected as too long: code=context_length_exceeded type=None" in caplog.text
    assert "exceeded the context window" not in caplog.text


def test_errors_map_to_the_busy_and_timeout_messages_and_others_are_raised() -> None:
    import httpx
    import openai

    guard = load_messages().guard
    assert failed_run(Raising(provider_error(429)))["generation"] == guard.busy  # no retry-after: nothing to wait for
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "http://llm.test/v1"))
    assert failed_run(Raising(timeout))["generation"] == guard.timeout
    with pytest.raises(openai.InternalServerError):  # a server error is no message of ours: the pipeline reports it
        failed_run(
            Raising(provider_error(500), provider_error(500)),
            card_update={"timeouts": load_card(CARD).timeouts.model_copy(update={"retries": 0})},
        )


def test_a_429_waits_the_retry_after_within_the_deadline_then_answers(monkeypatch) -> None:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("rag_core.adapters.chat_openai.asyncio.sleep", sleep)
    timeouts = load_card(CARD).timeouts.model_copy(update={"idle_s": 60, "deadline_s": 120})
    longer = {"timeouts": timeouts}
    llm = Raising(provider_error(429, retry_after="2"))
    out = failed_run(llm, card_update=longer)
    assert slept == [2.0] and llm.calls == 2 and out["generation"] == "Die Antwort"
    llm = Raising(provider_error(429, retry_after="500"))  # beyond the deadline and the silence limit: the busy message
    busy = load_messages().guard.busy
    assert failed_run(llm, card_update=longer, deadline_in=30)["generation"] == busy and llm.calls == 1
    llm = Raising(provider_error(429, retry_after="25"))  # fits the 30 s left, but not with the 10 s margin
    assert failed_run(llm, card_update=longer, deadline_in=30)["generation"] == busy and llm.calls == 1


def test_usage_recorded_per_answer() -> None:
    out = failed_run(Raising())
    assert out["usage"] == {"prompt_tokens": 120, "completion_tokens": 30}
    assert failed_run(Recording())["usage"] == {}  # a provider that returns none records none


def load_messages() -> Messages:
    import yaml

    return Messages.model_validate(yaml.safe_load((PROFILE.parent / "messages.de.yaml").read_text(encoding="utf-8")))


class Scored:
    """A retriever that returns one document per score (the retrieve node reads their cross_encoder_score)."""

    def __init__(self, *scores: float | None) -> None:
        self.scores, self.calls = scores, []

    async def retrieve(self, queries, filter=None, *, question="", date_from=None, date_to=None, principal=None):
        self.calls.append(list(queries))
        return [
            Document(
                page_content="Quelle eins" if i == 0 else f"Quelle {i + 1}",
                metadata={"filename": f"{i}.pdf", "section_id": f"p{i}", "cross_encoder_score": score},
            )
            for i, score in enumerate(self.scores)
        ]


class Recording:
    """The answer model: keeps the prompt it was asked with."""

    def __init__(self) -> None:
        self.prompts: list[Any] = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content="Die Antwort")


def fallback_run(
    intent: str, scores: tuple[float | None, ...], threshold: float | None, past=()
) -> tuple[dict, Scored, Recording]:
    import yaml

    raw = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    raw["retrieval"]["rerank"]["scope_fallback_min"] = threshold
    retrieval, llm = Scored(*scores), Recording()
    base = deps(intent)
    graph = build_graph(
        dataclasses.replace(base, profile=Profile.model_validate(raw), retrieval=retrieval, answer_llm=llm)
    )
    inputs = {
        "messages": history_messages(past) + [question_message("Frage?", None, datetime.now(timezone.utc))],
        "variant": None,
        "formats": "",
    }
    out = asyncio.run(graph.ainvoke(inputs, config={"configurable": {"thread_id": "t"}}))
    return out, retrieval, llm


def prompt_text(llm: Recording) -> str:
    return "\n".join(str(m.content) for m in llm.prompts[0])


def test_a_strong_hit_answers_an_out_of_scope_question_from_the_documents(caplog) -> None:
    caplog.set_level("INFO")
    out, retrieval, llm = fallback_run("out_of_scope", (0.8, 0.6), 0.7)
    assert out["intent"] == "out_of_scope" and out["scope_fallback"] is True
    assert "Quelle eins" in prompt_text(llm)  # the rag-generation prompt with its context, not the direct reply
    assert [s["section_id"] for s in out["context_sources"]] == ["p0", "p1"]
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[ScopeFallback]")]
    assert lines == ["[ScopeFallback] top1=0.800 threshold=0.700 answered=yes"]


def test_a_weak_hit_takes_the_direct_reply_without_sources() -> None:
    out, retrieval, llm = fallback_run("out_of_scope", (0.69, 0.4), 0.7)
    assert out["scope_fallback"] is False and out["raw_docs"] == [] and out["context_sources"] == []
    assert "Quelle eins" not in prompt_text(llm)
    assert len(retrieval.calls) == 1  # it did search; the hits were discarded


def test_a_top_hit_exactly_at_the_threshold_is_answered_and_no_hits_are_declined(caplog) -> None:
    caplog.set_level("INFO")
    out, _, llm = fallback_run("out_of_scope", (0.7,), 0.7)
    assert out["scope_fallback"] is True and "Quelle eins" in prompt_text(llm)
    out, _, llm = fallback_run("out_of_scope", (), 0.7)
    assert out["scope_fallback"] is False and out["raw_docs"] == []
    assert "Quelle eins" not in prompt_text(llm)
    assert [r.getMessage() for r in caplog.records if "[ScopeFallback]" in r.getMessage()][-1] == (
        "[ScopeFallback] top1=0.000 threshold=0.700 answered=no"
    )


def test_a_passage_without_a_score_counts_as_zero_and_no_hits_never_answer() -> None:
    out, _, llm = fallback_run("out_of_scope", (None, 0.4), 0.5)  # a parent may carry None (sort_score)
    assert out["scope_fallback"] is False and out["raw_docs"] == [] and "Quelle eins" not in prompt_text(llm)
    out, _, llm = fallback_run("out_of_scope", (), 0.0)  # no hits never answer, even at a threshold of 0.0
    assert out["scope_fallback"] is False and "Quelle eins" not in prompt_text(llm)


def test_a_searching_intent_is_unchanged_whatever_the_scores_and_the_threshold() -> None:
    for threshold in (None, 0.99):
        out, retrieval, llm = fallback_run("lookup", (0.2,), threshold)
        assert "scope_fallback" not in out or out["scope_fallback"] is False
        assert len(retrieval.calls) == 1 and "Quelle eins" in prompt_text(llm)  # the documents below T still answer
        assert [s["section_id"] for s in out["context_sources"]] == ["p0"]


def test_without_a_threshold_an_out_of_scope_question_never_searches() -> None:
    out, retrieval, llm = fallback_run("out_of_scope", (0.99,), None)
    assert retrieval.calls == [] and out["scope_fallback"] is False and out["raw_docs"] == []
    assert "Quelle eins" not in prompt_text(llm)


def test_a_follow_up_out_of_scope_turn_searches_with_the_analysis_queries() -> None:
    past = [PastTurn("t1", "Frage eins", "Antwort eins.", datetime.now(timezone.utc))]
    out, retrieval, llm = fallback_run("out_of_scope", (0.9,), 0.7, past)
    assert retrieval.calls == [out["retrieval_queries"]] and out["scope_fallback"] is True


def test_a_variant_of_a_fallback_answer_refits_from_its_stored_sources() -> None:
    now = datetime.now(timezone.utc)
    candidates = ({"page_content": "Quelle eins", "metadata": {"section_id": "p0", "filename": "0.pdf"}},)
    v = VariantInput(
        turn_id="t2",
        question="Frage?",
        asked_at=now,
        history=(),
        candidates=candidates,
        standalone_question="Frage?",
        intent="out_of_scope",
        language="Deutsch",
        shape="text",
        scope_fallback=True,
    )
    llm = Recording()
    graph = build_graph(dataclasses.replace(deps("out_of_scope"), answer_llm=llm))
    out = asyncio.run(graph.ainvoke(variant_inputs(v, "shorter", "v1"), config={"configurable": {"thread_id": "t"}}))
    assert "Quelle eins" in prompt_text(llm) and [s["section_id"] for s in out["context_sources"]] == ["p0"]
    assert variant_inputs(v, "shorter", "v1")["scope_fallback"] is True
    plain = dataclasses.replace(v, scope_fallback=False)
    llm2 = Recording()
    graph = build_graph(dataclasses.replace(deps("out_of_scope"), answer_llm=llm2))
    asyncio.run(graph.ainvoke(variant_inputs(plain, "shorter", "v1"), config={"configurable": {"thread_id": "t"}}))
    assert "Quelle eins" not in prompt_text(llm2)  # an out-of-scope variant takes the direct reply


def test_format_docs_returns_its_two_keys() -> None:
    state = {
        "raw_docs": [
            Document(page_content="x", metadata={"filename": "b.pdf"}),
            Document(page_content="y", metadata={"filename": "a.pdf"}),
        ],
        "messages": [],
    }
    out = asyncio.run(make_format_docs(deps("lookup"))(state))
    assert set(out) == {"grouped_raw_docs", "formatted_docs"}
    assert [d.page_content for d in out["grouped_raw_docs"]] == ["y", "x"]
    assert out["formatted_docs"].startswith("Es liegen 2 Quellen vor:")


def test_history_messages_are_what_context_turns_read_from_the_checkpoint() -> None:
    now = datetime.now(timezone.utc)
    past = [PastTurn("t1", "Frage eins", "Antwort eins [1].", now), PastTurn("t2", "Frage zwei", "Antwort zwei.", now)]
    messages = history_messages(past) + [question_message("Frage drei", "t3", now)]
    assert [(m.type, m.content, m.id) for m in messages[:2]] == [
        ("human", "Frage eins", "t1"),
        ("ai", "Antwort eins [1].", None),
    ]
    turns = context_turns(messages, 2, 12, now)
    assert [(q.content, a.content) for q, a in turns] == [
        ("Frage eins", "Antwort eins [1]."),
        ("Frage zwei", "Antwort zwei."),
    ]


def test_variant_inputs_hold_what_generate_read_from_the_checkpoint() -> None:
    now = datetime.now(timezone.utc)
    candidates = (
        {"page_content": "B", "metadata": {"section_id": "b"}, "section": "u|1|h"},
        {"page_content": "A", "metadata": {"section_id": "a"}, "section": "u|2|h"},
    )
    v = VariantInput(
        turn_id="t2",
        question="Wie melde ich mich krank?",
        asked_at=now,
        history=(PastTurn("t1", "Frage eins", "Antwort eins.", now),),
        candidates=candidates,
        standalone_question="Wie meldet man sich krank?",
        intent="lookup",
        language="Deutsch",
        shape="steps",
    )
    inputs = variant_inputs(v, "shorter", "v1")
    assert set(inputs) == {
        "messages",
        "variant",
        "answer_shape",
        "formats",
        "raw_docs",
        "grouped_raw_docs",
        "standalone_question",
        "intent",
        "language",
        "scope_fallback",
        "filter",
    }
    assert inputs["scope_fallback"] is False and inputs["filter"] is None
    kept = Filter(programme=("bsc-informatik",), po_version=(2015,), include_archived=True, po_source="chip")
    assert variant_inputs(dataclasses.replace(v, filter=kept), "shorter", "v1")["filter"] == kept
    assert [m.content for m in inputs["messages"]] == ["Frage eins", "Antwort eins.", "Wie melde ich mich krank?"]
    assert inputs["messages"][-1].id == "t2" and inputs["messages"][-1].type == "human"
    assert (
        [d.page_content for d in inputs["raw_docs"]]
        == ["B", "A"]
        == [d.page_content for d in inputs["grouped_raw_docs"]]
    )
    assert (
        inputs["variant"],
        inputs["answer_shape"],
        inputs["formats"],
        inputs["standalone_question"],
        inputs["intent"],
        inputs["language"],
    ) == ("shorter", "steps", "v1", "Wie meldet man sich krank?", "lookup", "Deutsch")
    assert documents(candidates)[0].metadata == {"section_id": "b"}


def runtime() -> types.SimpleNamespace:
    d = deps("lookup")
    overlay = types.SimpleNamespace(compute=types.SimpleNamespace(slot_wait_s=0.2))
    settings = types.SimpleNamespace(profile=d.profile, card=d.card, messages=d.messages, overlay=overlay)
    return types.SimpleNamespace(
        chain=build_graph(d),
        settings=settings,
        slot=asyncio.Semaphore(1),
        handler=None,
        collection="example",
    )


def frames(**kwargs) -> tuple[list[tuple[Any, Any]], asyncio.Semaphore]:
    """The chat service's frames (the guard's progress signals left out) over the real graph and fake models."""
    rt = runtime()
    slot = rt.slot

    async def run() -> list[Any]:
        if kwargs.pop("busy", False):
            await slot.acquire()
        return [f async for f in pipeline(rt, user="u", **kwargs)]

    # the frames' (type, content) as the page reads them; the guard's progress signals and declared outcomes left out
    sent = [fr.wire(f) for f in asyncio.run(run()) if not isinstance(f, str | fr.Declared)]
    return [(w["type"], w["content"]) for w in sent], slot


def test_a_question_streams_status_docs_chunks_and_citations_in_order() -> None:
    out, slot = frames(question="Wie beantrage ich Urlaub?", formats="v1")
    steps = [c["step"] if t == "status" else t for t, c in out]
    assert [s for i, s in enumerate(steps) if i == 0 or s != steps[i - 1]] == [
        "analysing",
        "understood",
        "shape",
        "searching",
        "found",
        "docs",
        "writing",
        "chunk",
        "citations",
    ]
    assert "".join(c for t, c in out if t == "chunk") == "Die Antwort"
    assert not slot.locked()


def test_a_question_over_the_limit_is_answered_without_the_model() -> None:
    out, slot = frames(question="x" * 2001)
    assert [t for t, _ in out] == ["chunk"] and "höchstens 2.000 Zeichen" in out[0][1]
    assert not slot.locked()


def test_a_busy_slot_ends_with_the_busy_error() -> None:
    out, _ = frames(question="Frage?", busy=True)
    assert out == [("error", "Service is currently busy. Please try again shortly.")]


def test_a_caller_with_only_a_turn_id_runs_its_question_under_it() -> None:
    rt, turn, seen = runtime(), str(uuid.uuid4()), []
    graph = rt.chain

    class Recorder:
        """The graph holds no checkpoint: the run's input is where the question's id shows."""

        def astream_events(self, inputs, **kwargs):
            seen.append(inputs)
            return graph.astream_events(inputs, **kwargs)

    rt.chain = Recorder()

    async def run() -> None:
        async for _ in pipeline(rt, user="u", question="Wie beantrage ich Urlaub?", turn_id=turn):
            pass

    asyncio.run(run())
    assert seen[0]["messages"][0].id == turn


def test_the_principal_reaches_the_retrieval_through_the_run_config() -> None:
    frames(question="Wie beantrage ich Urlaub?")
    assert SEEN[-1] == ANONYMOUS
    frames(question="Wie beantrage ich Urlaub?", principal=Principal("kollegin"))
    assert SEEN[-1] == Principal("kollegin")


def test_the_retrieve_node_hands_the_states_filter_to_the_retrieval() -> None:
    node = make_retrieve(deps("lookup"))
    messages = [question_message("Frage?", None, datetime.now(timezone.utc))]
    asyncio.run(node({"messages": messages, "filter": Filter(po_version=(2023,))}, {}))
    assert FILTERS[-1] == Filter(po_version=(2023,))
    asyncio.run(
        node({"messages": messages, "filter": None}, {})
    )  # filtering is off: the retrieval gets no filter at all
    assert FILTERS[-1] is None


def test_a_request_untraced_at_its_start_stays_untraced_when_the_login_heals_meanwhile(monkeypatch) -> None:
    """chat.py sent trace_id null in meta; the handler the heal task sets while the request waits for the slot must not
    attach the run to a random trace the page has no link to."""
    handlers = []

    def recording(handler, thread_id, collection, trace_id=None):
        handlers.append(handler)
        return {"configurable": {"thread_id": thread_id}}

    monkeypatch.setattr("assistant_api.chat.service.run_config", recording)

    async def run(trace_id) -> None:
        rt = runtime()  # the fake model answers once
        rt.handler = "handler"  # the heal task has set it since the route read it
        async for _ in pipeline(rt, user="u", question="Wie beantrage ich Urlaub?", trace_id=trace_id):
            pass

    for trace_id in (None, "abc"):
        asyncio.run(run(trace_id))
    assert handlers == [None, "handler"]


# A documents answer to a question about an event is sent whole, and replaced by one sentence when it writes a past date as
# coming.
class Dated:
    """A retriever with one document of the given text."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def retrieve(self, queries, filter=None, *, question="", date_from=None, date_to=None, principal=None):
        meta = {"filename": "a.pdf", "section_id": "p1", "cross_encoder_score": 0.9, "url": "https://x/a.pdf"}
        return [Document(page_content=self.text, metadata=meta)]


def dated_deps(answer: str, text: str, intent: str = "lookup") -> GraphDeps:
    llm = GenericFakeChatModel(messages=iter([AIMessage(content=answer)]))
    return dataclasses.replace(deps(intent), retrieval=Dated(text), answer_llm=llm)


def chunks(question: str, answer: str, text: str = "Sitzung am 12.03.2019.", intent: str = "lookup") -> list[str]:
    """The chunks the page receives for question when the answer model says answer."""
    rt = runtime()
    rt.chain = build_graph(dated_deps(answer, text, intent))
    sent = [fr.wire(f) for f in asyncio.run(collect(rt, question)) if not isinstance(f, str | fr.Declared)]
    return [w["content"] for w in sent if w["type"] == "chunk"]


async def collect(rt, question: str) -> list[Any]:
    return [f async for f in pipeline(rt, user="u", question=question)]


PAST_AS_COMING = "Die nächste Sitzung ist am 12.03.2019 [1]. Bitte melden Sie sich an."


def test_a_past_date_shown_as_coming_is_replaced_and_sent_whole(caplog) -> None:
    caplog.set_level("INFO")
    assert chunks("Wann ist die nächste Sitzung?", PAST_AS_COMING) == [
        "Der in den Dokumenten genannte Termin, 12.03.2019, ist bereits vorbei; einen neueren Termin nennen die Dokumente "
        "nicht [1]."
    ]
    assert "[PastEvent] held replaced=True" in [r.getMessage() for r in caplog.records]


def test_a_held_answer_without_a_past_date_shown_as_coming_is_sent_whole_unchanged() -> None:
    answer = "Die letzte Sitzung war am 12.03.2019 [1]. Einen neuen Termin nennen die Quellen nicht."
    assert chunks("Wann ist die nächste Sitzung?", answer) == [answer]


def test_a_question_about_no_event_streams_the_answer_unchanged() -> None:
    out = chunks("Wie beantrage ich Urlaub?", PAST_AS_COMING)
    assert len(out) > 1 and "".join(out) == PAST_AS_COMING


def test_a_direct_answer_is_never_held() -> None:
    out = chunks("Wann ist die nächste Sitzung?", PAST_AS_COMING, intent="conversation")
    assert len(out) > 1 and "".join(out) == PAST_AS_COMING


def test_a_later_date_in_the_sources_drops_the_no_newer_clause_and_several_dates_are_named() -> None:
    answer = "Die nächsten Termine sind am 12.03.2019 und am 1. Juli 2020 [2]."
    assert chunks("Wann sind die nächsten Termine?", answer, text="Termine 12.03.2019, 1. Juli 2020, 01.01.2030.") == [
        "Die in den Dokumenten genannten Termine, 12.03.2019 und 1. Juli 2020, sind bereits vorbei [2]."
    ]


def test_the_replacement_is_the_turns_answer() -> None:
    graph = build_graph(dated_deps(PAST_AS_COMING, "Sitzung am 12.03.2019."))
    inputs = {
        "messages": [question_message("Wann ist die nächste Sitzung?", None, datetime.now(timezone.utc))],
        "variant": None,
        "formats": "",
    }
    out = asyncio.run(graph.ainvoke(inputs, config={"configurable": {"thread_id": "t"}}))
    assert out["generation"] == out["messages"][-1].content
    assert out["messages"][-1].content.startswith(
        "Der in den Dokumenten genannte Termin, 12.03.2019, ist bereits vorbei"
    )


def state(question: str = "Frage?", standalone: str = "Frage?", **extra) -> dict:
    """An answer state as generate reads it: a lookup with one document unless extra says otherwise."""
    message = question_message(question, None, datetime.now(timezone.utc))
    return {"messages": [message], "standalone_question": standalone, "intent": "lookup", "raw_docs": ["d"], **extra}


@pytest.mark.parametrize(
    "kwargs, held",
    [
        ({"question": "Wann ist die nächste Sitzung?"}, True),
        ({"standalone": "Wann ist die nächste Sitzung des Personalrats?"}, True),  # a follow-up's standalone question
        ({"question": "Wann ist die nächste Sitzung?", "intent": "out_of_scope", "scope_fallback": True}, True),
        ({}, False),
        ({"question": "Wann ist die nächste Sitzung?", "intent": "conversation"}, False),
        ({"question": "Wann ist die nächste Sitzung?", "raw_docs": []}, False),
        ({"question": "Wann ist die nächste Sitzung?", "language": "Englisch"}, False),
        ({"question": "Wann ist die nächste Sitzung?", "formats": "v1", "variant": "english"}, False),
    ],
)
def test_which_answers_are_held(kwargs: dict, held: bool) -> None:
    from assistant_api.graph.nodes.generate import holds_answer

    assert holds_answer(state(**kwargs)) is held


@pytest.mark.parametrize(
    "answer, context, expected",
    [
        (
            "Termine: 12.03.2019 [1]. Danach am 1. Juli 2020 und erneut am 12.03.2019. Dann 04.08.2021 [3].",
            "",
            "Die in den Dokumenten genannten Termine, 12.03.2019, 1. Juli 2020 und 04.08.2021, sind bereits vorbei; "
            "einen neueren Termin nennen die Dokumente nicht [1].",
        ),
        (
            "Wir laden ein [4]. Die Sitzung ist am 12.03.2019.",
            "",
            "Der in den Dokumenten genannte Termin, 12.03.2019, ist bereits vorbei; einen neueren Termin nennen die Dokumente "
            "nicht [4].",
        ),
        (
            "Die Sitzung ist am 12.03.2019.",
            "Heute: 26.10.2026",
            "Der in den Dokumenten genannte Termin, 12.03.2019, ist bereits vorbei.",
        ),
        ("Die Sitzung war am 12.03.2019 [1].", "", None),
    ],
)
def test_the_replacing_sentence(answer: str, context: str, expected: str | None) -> None:
    from assistant_api.graph.nodes.generate import past_event_reply

    assert past_event_reply(answer, context, date(2026, 10, 26), deps("lookup").messages.answer) == expected


# The answer model's requests, recorded by the prompt log.
PROMPTS: list[Any] = []  # the messages the answer model was asked with


ANSWER_OPTIONS: list[Any] = []  # the options bound to the answer model's requests


class PromptLog(GenericFakeChatModel):
    def _stream(self, messages, *args, **kwargs):
        PROMPTS.append(messages)
        ANSWER_OPTIONS.append(kwargs)
        return super()._stream(messages, *args, **kwargs)


class Titled:
    """A retriever with one document per title, the first ranked best."""

    def __init__(self, *titles: str) -> None:
        self.titles = titles

    async def retrieve(self, queries, filter=None, *, question="", date_from=None, date_to=None, principal=None):
        return [
            Document(
                page_content="Regel %d" % i,
                metadata={
                    "title": t,
                    "section_id": "p%d" % i,
                    "cross_encoder_score": 0.9 - 0.01 * i,
                    "url": "https://x/%d.pdf" % i,
                },
            )
            for i, t in enumerate(self.titles)
        ]


def test_the_logit_bias_and_bad_words_are_bound_to_german_answers() -> None:
    from assistant_api.graph.nodes.generate import with_decoding_limits

    class Llm:
        extra_body = {"top_k": 50}

        def bind(self, **options):
            return options

    quirks = load_card(CARD).quirks.model_copy(update={"logit_bias": {12: -5.0}, "bad_words": ("Mitarbeiter",)})
    assert with_decoding_limits(Llm(), quirks, "vllm") == {
        "logit_bias": {"12": -5.0},
        "extra_body": {"top_k": 50, "bad_words": ["Mitarbeiter"]},
    }
    llm = Llm()
    assert with_decoding_limits(llm, load_card(CARD).quirks, "vllm") is llm
    assert with_decoding_limits(llm, quirks, "ollama") is llm  # a provider that accepts no extras gets none


ENGLISH_QUESTION = "How do I apply for leave?"


def answered(question: str, intent: str, *titles: str, **quirks: Any) -> tuple[Any, Any]:
    """The messages and the options of the answer model's request for the question; `quirks` overrides the card's."""
    PROMPTS.clear()
    ANSWER_OPTIONS.clear()
    base = deps(intent)
    d = dataclasses.replace(
        base,
        card=base.card.model_copy(update={"quirks": base.card.quirks.model_copy(update=quirks)}),
        retrieval=Titled(*titles),
        answer_llm=PromptLog(messages=iter([AIMessage(content="Antwort.")])),
    )
    rt = runtime()
    rt.chain = build_graph(d)
    asyncio.run(collect(rt, question))
    return PROMPTS[-1], ANSWER_OPTIONS[-1]


def test_the_english_reminder_ends_the_message_on_the_documents_no_documents_and_direct_reply_paths() -> None:
    from rag_core.domain.answer_shapes import ENGLISH_REMINDER

    for intent, titles in (("lookup", ("Urlaub",)), ("lookup", ()), ("conversation", ())):
        prompt, _ = answered(ENGLISH_QUESTION, intent, *titles)
        assert str(prompt[-1].content).endswith("\n\n" + ENGLISH_REMINDER), (intent, titles)
        assert "Antworte in: Englisch" in str(prompt[0].content)
        prompt, _ = answered("Wie beantrage ich Urlaub?", intent, *titles)
        assert ENGLISH_REMINDER not in str(prompt[-1].content)


def test_the_du_bias_and_bad_words_go_with_german_answers_only() -> None:
    limits = {"logit_bias": {3845: -5.0}, "bad_words": ("Mitarbeiter",)}
    _, german = answered("Wie beantrage ich Urlaub?", "lookup", "Urlaub", **limits)
    assert "3845" in german["logit_bias"] and "bad_words" in german["extra_body"]
    _, english = answered(ENGLISH_QUESTION, "lookup", "Urlaub", **limits)
    assert "logit_bias" not in english and "bad_words" not in (english.get("extra_body") or {})


def filtered_state(question: str, chips=None, roles=True, switch=True, previous=None) -> dict:
    """The final state of a run over the TU roles (the example profile has none) with the filters switch as given."""
    import dataclasses

    from rag_core.settings.profile import Roles, Switches

    tu = Path(__file__).resolve().parents[2] / "profiles" / "tu-informatik"
    base = deps("lookup")
    retrieval = base.profile.retrieval.model_copy(update={"switches": Switches(filters=switch)})
    profile = base.profile.model_copy(update={"retrieval": retrieval})
    graph = build_graph(
        dataclasses.replace(
            base, profile=profile, roles=load_profile(Roles, tu / "roles.yaml").model_dump() if roles else None
        )
    )
    inputs = {
        "messages": [question_message(question, None, datetime.now(timezone.utc))],
        "variant": None,
        "formats": "",
        "filter": previous,
    }
    config = {"configurable": {"thread_id": "t", "chips": chips}}
    return asyncio.run(graph.ainvoke(inputs, config=config))


def test_the_analysis_step_resolves_the_filter_from_the_question_the_chips_and_the_previous_turn() -> None:
    from rag_core.domain.filters import Chips, Filter

    own = filtered_state("Was gilt in der PO 2015?", Chips(degree="master"))["filter"]
    assert (own.degree, own.po_version, own.include_archived) == (("master",), (2015,), True)
    previous = Filter(programme=("msc-informatik",))
    followed = filtered_state("Und die Frist?", Chips(), previous=previous)["filter"]
    assert followed == previous  # the question names nothing: the previous turn's filter stays


def test_without_the_switch_or_roles_there_is_no_filter_at_all() -> None:
    assert filtered_state("Was gilt in der PO 2015?", switch=False)["filter"] is None
    assert filtered_state("Was gilt in der PO 2015?", roles=False)["filter"] is None


def no_docs_text(question: str, chips=None, roles=True, previous=None) -> str:
    """The reply of a run over the TU profile's roles and messages (German and English) whose search finds nothing."""
    from rag_core.settings.profile import Roles

    tu = Path(__file__).resolve().parents[2] / "profiles" / "tu-informatik"
    load = lambda name: Messages.model_validate(yaml.safe_load((tu / name).read_text(encoding="utf-8")))  # noqa: E731
    base = deps("lookup")
    graph = build_graph(
        dataclasses.replace(
            base,
            messages=load("messages.de.yaml"),
            messages_en=load("messages.en.yaml"),
            roles=load_profile(Roles, tu / "roles.yaml").model_dump() if roles else None,
            retrieval=Titled(),
            answer_llm=GenericFakeChatModel(messages=iter([AIMessage(content="Dazu steht nichts in den Dokumenten.")])),
        )
    )
    inputs = {
        "messages": [question_message(question, None, datetime.now(timezone.utc))],
        "variant": None,
        "formats": "",
        "filter": previous,
    }
    final = asyncio.run(graph.ainvoke(inputs, config={"configurable": {"thread_id": "t", "chips": chips}}))
    return str(final["generation"])


def test_an_empty_result_names_the_applied_filter_in_the_questions_language() -> None:
    from rag_core.domain.filters import Chips

    master = Chips("master", "msc-informatik", 2015)
    assert no_docs_text("Wann ist die Anmeldung?", master) == (
        "Dazu steht nichts in den Dokumenten.\n\nFür den M.Sc. Informatik, PO 2015, habe ich nichts gefunden."
    )
    assert no_docs_text(ENGLISH_QUESTION, master).endswith("\n\nI found nothing for the M.Sc. Informatik, PO 2015.")
    # what the filter names, whatever way it was set: the degree alone, the PO alone, the previous turn's filter
    assert no_docs_text("Wann?", Chips(degree="master")).endswith("\n\nFür den Master habe ich nichts gefunden.")
    assert no_docs_text("Wann?", Chips(po=2015)).endswith("\n\nFür PO 2015 habe ich nichts gefunden.")
    carried = Filter(programme=("msc-informatik", "msc-computer-science"), po_version=(2015, 2023))
    assert no_docs_text("Und wann?", previous=carried).endswith(
        "\n\nFür den M.Sc. Informatik / M.Sc. Computer Science, PO 2015/2023, habe ich nichts gefunden."
    )


def test_an_empty_result_without_a_named_filter_gets_no_line() -> None:
    from rag_core.domain.filters import Chips

    for text in (no_docs_text("Wann?", Chips()), no_docs_text("Wann?", roles=False)):
        assert text == "Dazu steht nichts in den Dokumenten."
