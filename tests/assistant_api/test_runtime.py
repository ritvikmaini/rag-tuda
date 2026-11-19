"""The runtime's reload: the new index and graph first, then the previous index's pool closed after the grace; a failed start
leaves the app degraded and a background task (heal) and the request path heal it, each at most once a minute."""

import asyncio
import logging
import re
import types
from pathlib import Path

import pytest
from assistant_api import runtime as rt_module
from rag_core.services.prompts import PromptService, load_profile_prompts
from rag_core.settings import load_profile
from rag_core.settings.card import load_card
from rag_core.settings.profile import Profile

PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "example" / "profile.yaml"


class Engine:
    def __init__(self) -> None:
        self.closed = False

    async def dispose(self) -> None:
        self.closed = True


def runtime() -> types.SimpleNamespace:
    settings = types.SimpleNamespace(
        profile=types.SimpleNamespace(
            collections=[types.SimpleNamespace(name="example")], retrieval=None, ranking=None
        ),
        overlay=types.SimpleNamespace(
            database=None,
            timeouts=types.SimpleNamespace(),
            model_servers=types.SimpleNamespace(reranker_url="http://reranker_service:8003"),
        ),
        messages=None,
        messages_en=None,
        roles=None,
        acronyms=None,
        card=types.SimpleNamespace(
            timeouts=types.SimpleNamespace(deadline_s=0),
            budget=types.SimpleNamespace(final_k=8, pre_rerank_k=20, rerank_queries=1),
            chars_per_token=3.3,
        ),
    )
    return types.SimpleNamespace(
        settings=settings,
        collection="example",
        embedder=None,
        prompts=None,
        analysis=None,
        follow_up=None,
        answer_llm=None,
        counter=None,
        turns=None,
        langfuse=None,
        handler=None,
        generation=None,
        index=None,
        chain=None,
        reload_lock=asyncio.Lock(),
        last_reload_attempt=float("-inf"),
        retiring={},
    )


def test_reload_swaps_then_disposes_the_old_engines(monkeypatch) -> None:
    order = []
    old = types.SimpleNamespace(engines=(Engine(), Engine()))

    async def open_index(collection, profile, database, acronyms):
        order.append(("open", collection, acronyms))
        return types.SimpleNamespace(vector=None, full_text=None, sections=None, engines=(Engine(), Engine()))

    def build_graph(deps):
        order.append(("graph", all(not e.closed for e in old.engines)))
        return "chain"

    monkeypatch.setattr(rt_module, "open_index", open_index)
    monkeypatch.setattr(rt_module, "build_graph", build_graph)
    monkeypatch.setattr(rt_module, "RetrievalService", lambda *a, **k: None)
    rt = runtime()
    rt.index = old
    rt.settings.acronyms = "the acronym groups"

    async def scenario() -> None:
        await rt_module.reload(rt)
        await asyncio.gather(*rt.retiring)

    asyncio.run(scenario())
    assert order == [
        ("open", "example", "the acronym groups"),
        ("graph", True),
    ]  # the index opens with the profile's groups
    assert rt.chain == "chain" and all(e.closed for e in old.engines)


def test_the_self_heal_tries_at_most_once_a_minute(monkeypatch) -> None:
    calls = []

    async def failing(rt):
        calls.append(1)
        raise RuntimeError("no tables yet")

    monkeypatch.setattr(rt_module, "reload", failing)
    rt = runtime()
    assert asyncio.run(rt_module.try_reload(rt)) is False
    assert asyncio.run(rt_module.try_reload(rt)) is False
    assert calls == [1]


def test_a_reload_closes_the_old_pool_after_the_request_deadline(monkeypatch) -> None:
    source = Path(rt_module.__file__).read_text(encoding="utf-8")
    assert "from rag_core.adapters.pg_index import IndexHandles, open_index" in source
    assert "pgvector_store" not in source and "index.search" not in source
    old = types.SimpleNamespace(engines=(Engine(),))

    async def open_index(collection, profile, database, acronyms):
        return types.SimpleNamespace(vector=None, full_text=None, sections=None, engines=(Engine(),))

    monkeypatch.setattr(rt_module, "open_index", open_index)
    monkeypatch.setattr(rt_module, "build_graph", lambda deps: "chain")
    monkeypatch.setattr(rt_module, "RetrievalService", lambda *a, **k: None)
    rt = runtime()
    rt.index = old

    async def scenario() -> bool:
        await rt_module.reload(rt)
        closed_at_the_swap = old.engines[0].closed
        await asyncio.gather(*rt.retiring)
        return closed_at_the_swap

    assert asyncio.run(scenario()) is False and old.engines[0].closed and rt.chain == "chain"


def test_shutdown_takes_the_waiting_pools_at_once() -> None:
    async def scenario() -> list:
        rt = runtime()
        rt_module.retire(rt, ("pool",), 3600)
        return rt_module.stop_retiring(rt)

    assert asyncio.run(scenario()) == ["pool"]


def test_the_self_heal_needs_no_checkpointer(monkeypatch) -> None:
    calls = []

    async def ok(rt):
        calls.append(1)
        rt.chain = "chain"

    monkeypatch.setattr(rt_module, "reload", ok)
    rt = runtime()
    rt.turns = None
    assert asyncio.run(rt_module.try_reload(rt)) is True and calls == [1]


def test_a_reload_reads_the_live_generation(monkeypatch) -> None:
    class Turns:
        async def live_generation(self, collection):
            return 11 if collection == "example" else None

    async def open_index(collection, profile, database, acronyms):
        return types.SimpleNamespace(vector=None, full_text=None, sections=None, engines=())

    monkeypatch.setattr(rt_module, "open_index", open_index)
    monkeypatch.setattr(rt_module, "build_graph", lambda deps: "chain")
    monkeypatch.setattr(rt_module, "RetrievalService", lambda *a, **k: None)
    rt = runtime()
    rt.turns = Turns()
    asyncio.run(rt_module.reload(rt))
    assert rt.generation == 11 and rt.chain == "chain"


@pytest.mark.parametrize("had_index", [False, True])
def test_a_failing_graph_build_keeps_the_old_index_and_closes_the_new_pool(monkeypatch, had_index) -> None:
    """A failing graph build must not leave the runtime on handles no graph used or the new pool unclosed (start-up and reload)."""
    old = types.SimpleNamespace(engines=(Engine(),)) if had_index else None
    new = types.SimpleNamespace(vector=None, full_text=None, sections=None, engines=(Engine(),))

    async def open_index(collection, profile, database, acronyms):
        return new

    def build_graph(deps):
        raise RuntimeError("graph broke")

    class Turns:
        async def live_generation(self, collection):
            return 12

    monkeypatch.setattr(rt_module, "open_index", open_index)
    monkeypatch.setattr(rt_module, "build_graph", build_graph)
    monkeypatch.setattr(rt_module, "RetrievalService", lambda *a, **k: None)
    rt = runtime()
    rt.index, rt.generation, rt.turns = old, 7 if had_index else None, Turns()
    with pytest.raises(RuntimeError, match="graph broke"):
        asyncio.run(rt_module.reload(rt))
    assert rt.index is old and rt.generation == (7 if had_index else None) and rt.chain is None
    assert new.engines[0].closed and (old is None or not old.engines[0].closed) and rt.retiring == {}


def test_run_info_names_what_a_turn_ran_with() -> None:
    profile = load_profile(Profile, PROFILE)
    card = load_card(PROFILE.parents[2] / "deploy" / "env" / "ci-fake-card.toml")
    bundled = load_profile_prompts(PROFILE.parent / "prompts")

    def make(profile: Profile, card=card) -> rt_module.Runtime:
        return rt_module.Runtime(
            settings=types.SimpleNamespace(profile=profile, card=card),  # pyright: ignore[reportArgumentType]
            langfuse=None,
            handler=None,
            prompts=PromptService(bundled),
            answer_llm=None,
            analysis=None,  # pyright: ignore[reportArgumentType]
            counter=None,  # pyright: ignore[reportArgumentType]
            embedder=None,  # pyright: ignore[reportArgumentType]
            slot=asyncio.Semaphore(1),
        )

    rt = make(profile)
    rt.generation = 17
    info = rt.run_info()
    assert info["collection"] == "rag" and info["index_generation"] == 17
    assert info["prompts"] == {name: prompt.version for name, prompt in bundled.items()} and len(info["prompts"]) == 4
    assert re.fullmatch(r"[0-9a-f]{16}", info["config_hash"]) and info == make(profile).run_info() | {
        "index_generation": 17
    }
    changed = profile.model_copy(
        update={"history": profile.history.model_copy(update={"max_age_hours": profile.history.max_age_hours + 1})}
    )
    assert make(changed).config_hash != rt.config_hash
    other = card.model_copy(update={"quirks": card.quirks.model_copy(update={"strip_cjk": not card.quirks.strip_cjk})})
    assert make(profile, other).config_hash != rt.config_hash  # the model card is part of the configuration


ENV = {
    "RAG_PROFILE": str(PROFILE),
    "RAG_CARDS_DIR": str(PROFILE.parents[2] / "models" / "cards"),
    "RAG__COMPUTE__SLOT_WAIT_S": "3",
    "RAG__TIMEOUTS__RETRIEVAL_S": "10",
    "RAG__TIMEOUTS__PARENT_LOOKUP_S": "5",
    "RAG__TIMEOUTS__RERANKER_S": "15",
    "RAG__HTTP__CORS_DOMAIN": "example.org",
    "RAG__TRACING__URL": "",
}


def runtime_of(card: str, tmp_path, monkeypatch):
    from assistant_api.settings import load_settings

    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")  # a hosted card's answer client
    settings = load_settings({**ENV, "RAG__CARD": card}, secrets_dir=tmp_path)
    return settings, rt_module.build_runtime(settings, None, False)


def test_the_clients_and_the_window_come_from_the_card(tmp_path, monkeypatch) -> None:
    from rag_core.settings.card import effective_window

    settings, rt = runtime_of("l-groq-gpt-oss-120b", tmp_path, monkeypatch)
    assert rt.answer_llm.model_name == "openai/gpt-oss-120b"
    assert str(rt.answer_llm.openai_api_base) == "https://api.groq.com/openai/v1"
    assert rt.analysis.runnable.bound.model_name == "qwen3:4b-instruct-2507-q4_K_M"  # the card's own analysis block
    assert str(rt.analysis.runnable.bound.openai_api_base) == "http://127.0.0.1:11434/v1"
    assert effective_window(settings.card) == 8000 == rt.counter.max_model_len
    assert not rt.counter.exact and rt.counter.chars_per_token == settings.card.chars_per_token  # no /tokenize at Groq
    # the options each provider accepts: Groq the reasoning level, Ollama (the analysis) its own and its cap as max_tokens,
    # neither the vLLM extras
    assert rt.answer_llm.extra_body == {"reasoning_effort": "low"}
    # (the analysis model on the machine has no reasoning)
    assert rt.analysis.runnable.bound.extra_body == {"reasoning_effort": "none", "max_tokens": 800}
    # the analysis client never streams (the answer's does) and sends the card's response_format; its provider here is
    # Ollama, so the schema is not strict
    assert rt.analysis.runnable.bound.disable_streaming and not rt.answer_llm.disable_streaming
    assert rt.analysis.runnable.kwargs["response_format"]["json_schema"]["strict"] is False
    assert rt.analysis.runnable.kwargs["max_tokens"] == settings.profile.generation.analysis_max_tokens == 800
    # the example profile has no roles: the base schema
    assert "search_queries" in rt.analysis.runnable.kwargs["response_format"]["json_schema"]["schema"]["properties"]


def test_the_analysis_schema_takes_its_enums_from_the_profiles_roles(tmp_path, monkeypatch) -> None:
    from assistant_api.settings import load_settings

    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    tu = PROFILE.parents[1] / "tu-informatik" / "profile.yaml"
    settings = load_settings(
        {**ENV, "RAG_PROFILE": str(tu), "RAG__CARD": "s-qwen3-4b-instruct-2507"}, secrets_dir=tmp_path
    )
    schema = rt_module.build_runtime(settings, None, False).analysis.runnable.kwargs["response_format"]["json_schema"][
        "schema"
    ]
    assert list(schema["properties"])[0] == "reasoning"
    assert schema["properties"]["programme"]["anyOf"][0]["items"]["enum"] == [
        "bsc-informatik",
        "msc-computer-science",
        "msc-informatik",
    ]


def test_a_hosted_card_sends_groqs_key_to_groq_only(tmp_path, monkeypatch) -> None:
    _, rt = runtime_of("l-groq-gpt-oss-120b", tmp_path, monkeypatch)
    assert rt.answer_llm.openai_api_key is not None and rt.answer_llm.openai_api_key.get_secret_value() == "gsk-test"
    analysis_key = rt.analysis.runnable.bound.openai_api_key
    assert analysis_key is not None and analysis_key.get_secret_value() == "not-needed"  # the local analysis model


def test_timeouts_come_from_card(tmp_path, monkeypatch) -> None:
    settings, rt = runtime_of("xl-vllm-example", tmp_path, monkeypatch)
    card = settings.card
    assert card.slots == 4 and rt.slot._value == 4  # vLLM batches: the card's slots, not a setting of the stack
    assert rt.answer_llm.request_timeout.read == card.timeouts.llm_read_s
    assert rt.analysis.runnable.bound.request_timeout.read == card.timeouts.llm_read_s
    assert rt.answer_llm.max_tokens == card.budget.answer_tokens  # the answer's tokens are the card's
    assert rt.answer_llm.max_retries == 0 and rt.analysis.runnable.bound.max_retries == card.timeouts.retries
    assert rt.answer_llm.temperature == card.sampling.temperature and rt.answer_llm.top_p == card.sampling.top_p
    assert rt.answer_llm.extra_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert rt.counter.url == card.answer.url.removesuffix("/v1") + "/tokenize"  # /tokenize at vLLM
    # the guard's deadline and silence limit are the card's
    short = card.model_copy(update={"timeouts": card.timeouts.model_copy(update={"deadline_s": 41, "idle_s": 17})})
    seen = {}

    async def fake_guard(*args, **kwargs):
        seen.update(kwargs)
        yield ""

    from assistant_api.api import chat as chat_route
    from assistant_api.app import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(chat_route, "guard", fake_guard)
    app = create_app()
    app.dependency_overrides[chat_route.verify_token] = lambda: None
    rt.settings = types.SimpleNamespace(
        card=short, profile=settings.profile, overlay=settings.overlay, messages_for=settings.messages_for
    )
    rt.handler, rt.turns = None, None
    app.state.runtime = rt
    TestClient(app).get("/api/chat", params={"question": "Frage", "conversation_id": "c-1"})
    assert (seen["deadline_s"], seen["idle_s"]) == (41, 17)


class Langfuse:
    """A client whose auth_check answers from a list (a value is returned, an exception raised)."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)

    def auth_check(self) -> bool:
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def heal_records(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[Heal]")]


def test_heal_logs_in_to_langfuse_and_turns_tracing_on(monkeypatch, caplog) -> None:
    monkeypatch.setattr(rt_module, "RELOAD_RETRY_S", 0)
    monkeypatch.setattr(rt_module, "make_handler", lambda authenticated: "handler" if authenticated else None)
    rt = runtime()
    rt.chain = "chain"
    rt.langfuse = Langfuse(False, ConnectionError("down"), True)
    with caplog.at_level(logging.INFO):
        asyncio.run(asyncio.wait_for(rt_module.heal(rt), 5))
    assert rt.handler == "handler"
    records = heal_records(caplog)
    assert records[0] == "[Heal] retrying every 60 s: langfuse" and len(records) == 2
    assert records[1] == "[Heal] Langfuse authenticated, tracing on"


def test_heal_says_once_that_it_retries_and_stays_quiet_while_it_fails(monkeypatch, caplog) -> None:
    monkeypatch.setattr(rt_module, "RELOAD_RETRY_S", 0)
    calls = []

    async def failing(rt):
        calls.append(1)
        raise RuntimeError("no tables yet")

    monkeypatch.setattr(rt_module, "reload", failing)
    rt = runtime()

    async def scenario() -> None:
        task = asyncio.create_task(rt_module.heal(rt))
        while len(calls) < 3 and not task.done():
            await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    assert len(heal_records(caplog)) == 1 and "retrying every 60 s: index" in heal_records(caplog)[0]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING and not r.getMessage().startswith("[Heal]")] == []
    assert not any("[Startup]" in r.getMessage() for r in caplog.records)


def test_heal_returns_once_the_index_loads(monkeypatch, caplog) -> None:
    monkeypatch.setattr(rt_module, "RELOAD_RETRY_S", 0)
    calls = []

    async def flaky(rt):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("no tables yet")
        rt.chain = "chain"

    monkeypatch.setattr(rt_module, "reload", flaky)
    rt = runtime()
    with caplog.at_level(logging.INFO):
        asyncio.run(asyncio.wait_for(rt_module.heal(rt), 5))
    assert calls == [1, 1, 1] and rt.chain == "chain"
    assert heal_records(caplog) == ["[Heal] retrying every 60 s: index", "[Heal] index loaded"]


def test_heal_logs_an_unexpected_error_and_tries_again(monkeypatch, caplog) -> None:
    monkeypatch.setattr(rt_module, "RELOAD_RETRY_S", 0)
    made = []

    def make_handler(authenticated):
        made.append(1)
        if len(made) == 1:
            raise ValueError("boom")
        return "handler"

    monkeypatch.setattr(rt_module, "make_handler", make_handler)
    rt = runtime()
    rt.chain = "chain"
    rt.langfuse = Langfuse(True)
    with caplog.at_level(logging.INFO):
        asyncio.run(asyncio.wait_for(rt_module.heal(rt), 5))
    assert made == [1, 1] and rt.handler == "handler"
    failed = [r for r in caplog.records if "attempt failed unexpectedly" in r.getMessage()]
    assert len(failed) == 1 and failed[0].levelno == logging.ERROR and failed[0].exc_info is not None


def test_heal_has_nothing_to_do_when_nothing_is_broken(caplog) -> None:
    rt = runtime()
    rt.chain = "chain"
    with caplog.at_level(logging.INFO):
        asyncio.run(asyncio.wait_for(rt_module.heal(rt), 5))
    rt.langfuse, rt.handler = Langfuse(True), "handler"  # a client that logged in at the start
    asyncio.run(asyncio.wait_for(rt_module.heal(rt), 5))
    assert heal_records(caplog) == []


def test_heal_never_retries_a_missing_langfuse_client(monkeypatch) -> None:
    monkeypatch.setattr(rt_module, "RELOAD_RETRY_S", 0)
    calls = []

    async def ok(rt):
        calls.append(1)
        rt.chain = "chain"

    monkeypatch.setattr(rt_module, "reload", ok)
    rt = runtime()  # langfuse=None: configuration (no key files), not a state that heals
    asyncio.run(asyncio.wait_for(rt_module.heal(rt), 5))
    assert calls == [1] and rt.handler is None


def test_a_quiet_reload_attempt_logs_nothing_when_it_fails(monkeypatch, caplog) -> None:
    async def failing(rt):
        raise RuntimeError("no tables yet")

    monkeypatch.setattr(rt_module, "reload", failing)
    with caplog.at_level(logging.INFO):
        assert asyncio.run(rt_module.try_reload(runtime(), quiet=True)) is False
        assert caplog.records == []
        assert asyncio.run(rt_module.try_reload(runtime())) is False
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_readiness_never_calls_a_remote_endpoint(tmp_path, monkeypatch) -> None:
    probed = []

    async def report(url, expected):
        probed.append(url)
        return "ok"

    monkeypatch.setattr(rt_module, "_report", report)
    _, hosted = runtime_of("l-groq-gpt-oss-120b", tmp_path, monkeypatch)
    reports = asyncio.run(rt_module.model_reports(hosted))
    assert reports["llm"] == "remote, not probed" and reports["analysis_llm"] == "ok"
    assert not any("groq" in url for url in probed) and "http://127.0.0.1:11434/v1/models" in probed
    probed.clear()
    _, local = runtime_of("s-qwen3-4b-instruct-2507", tmp_path, monkeypatch)
    assert set(asyncio.run(rt_module.model_reports(local))) == {"embedding", "reranker", "llm"}
    assert "http://127.0.0.1:11434/v1/models" in probed


def test_the_follow_up_client_is_the_analysis_client_with_its_own_schema(tmp_path, monkeypatch) -> None:
    from assistant_api.settings import load_settings

    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    tu = PROFILE.parents[1] / "tu-informatik" / "profile.yaml"
    settings = load_settings({**ENV, "RAG_PROFILE": str(tu), "RAG__CARD": "m-qwen3.8-27b"}, secrets_dir=tmp_path)
    rt = rt_module.build_runtime(settings, None, False)
    assert (
        rt.follow_up is not None and rt.follow_up.runnable.bound is rt.analysis.runnable.bound
    )  # one client, never streamed
    sent = rt.follow_up.runnable.kwargs
    assert sent["max_tokens"] == rt.analysis.runnable.kwargs["max_tokens"]
    assert sent["response_format"]["json_schema"]["name"] == "follow_up_queries"
    assert list(sent["response_format"]["json_schema"]["schema"]["properties"]) == ["reasoning", "queries", "done"]
    # the example profile ships no follow-up prompt: no client
    example, rt = runtime_of("l-groq-gpt-oss-120b", tmp_path, monkeypatch)
    assert rt.follow_up is None
