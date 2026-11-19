"""The app's routes: /healthz, /readyz (reports without gating on the model servers), the chat route's token rule and its's parameters."""

import asyncio
import json
import types
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr
from rag_core.domain.access import Principal

from assistant_api import app as app_module
from assistant_api import runtime as rt_module
from assistant_api.api import chat as chat_route
from assistant_api.api.deps import limiter
from assistant_api.app import create_app
from assistant_api.chat.turns_store import TurnRecord
from assistant_api.sse import frames as fr


def app_with(chain: object = None, token: str = "t0k"):
    app = create_app()
    rt = types.SimpleNamespace(
        chain=chain,
        index=None,
        langfuse=None,
        reload_lock=__import__("asyncio").Lock(),
        settings=types.SimpleNamespace(
            overlay=types.SimpleNamespace(
                server=types.SimpleNamespace(token=SecretStr(token)),
                http=types.SimpleNamespace(cors_domain="example.org"),
            )
        ),
    )
    app.state.runtime = rt
    return TestClient(app), rt


def test_healthz_and_no_health_alias() -> None:
    client, _ = app_with()
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/health").status_code == 404


def test_readyz_is_503_without_a_loaded_index(monkeypatch) -> None:
    async def reports(rt):
        return {"embedding": "unavailable: ConnectError"}

    monkeypatch.setattr(rt_module, "model_reports", reports)
    client, _ = app_with()
    reply = client.get("/readyz")
    assert reply.status_code == 503 and reply.json()["ready"] is False
    assert reply.json()["checks"]["index"] == "missing" and "embedding" in reply.json()["reports"]


def test_the_chat_route_takes_the_token_and_the_removed_routes_are_gone() -> None:
    client, _ = app_with()
    ask = {"question": "Hallo", "conversation_id": "u"}
    assert client.get("/api/chat", params=ask).status_code == 401
    assert client.get("/api/chat", params=ask, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/v1/admin/reload", headers={"Authorization": "Bearer t0k"}).status_code == 404


def test_without_the_secret_file_no_token_matches(monkeypatch) -> None:
    monkeypatch.setenv("BACKEND_REFRESH_TOKEN", "t0k")  # the environment variable is not read
    client, rt = app_with()
    rt.settings.overlay.server.token = None
    ask = {"question": "Hallo", "conversation_id": "u"}
    assert client.get("/api/chat", params=ask, headers={"Authorization": "Bearer t0k"}).status_code == 401


def test_the_chat_route_keeps_its_parameters() -> None:
    client, _ = app_with()
    route = next(r for r in client.app.routes if getattr(r, "path", "") == "/api/chat")
    names = [p.name for p in route.dependant.query_params]
    assert names == [
        "question",
        "conversation_id",
        "turn_id",
        "replace_from",
        "kept",
        "formats",
        "variant",
        "variant_of",
        "variant_select",
        "degree",
        "programme",
        "po",
        "lang",
    ]


def test_security_headers_and_cors_follow_the_settings() -> None:
    client, _ = app_with()
    plain = client.get("/healthz")
    assert plain.headers["X-Frame-Options"] == "DENY" and "Access-Control-Allow-Origin" not in plain.headers
    for origin, allowed in (
        ("https://www.example.org", True),  # the overlay's cors_domain and its subdomains
        ("http://localhost:3000", True),
        ("https://evil.example", False),
    ):
        reply = client.get("/healthz", headers={"Origin": origin})
        assert (reply.headers.get("Access-Control-Allow-Origin") == origin) is allowed, origin


def test_the_chat_route_makes_a_turn_record_for_a_question_and_ends_it_through_the_guard(monkeypatch) -> None:
    turn, seen, ended = "11111111-1111-4111-8111-111111111111", [], []

    async def pipeline(rt, **kwargs):
        seen.append(kwargs["record"])
        yield fr.Chunk(content="Antwort")

    class Turns:
        async def finish(self, record, outcome):
            ended.append((record, outcome))

    monkeypatch.setattr(chat_route, "pipeline", pipeline)
    client, rt = app_with()
    rt.turns, rt.handler, rt.messages_for = Turns(), None, lambda lang: {"guard.error": "Fehler"}
    rt.settings.profile = types.SimpleNamespace(guard=types.SimpleNamespace(max_question_chars=2000))
    rt.settings.card = types.SimpleNamespace(timeouts=types.SimpleNamespace(deadline_s=5, idle_s=5))
    ask = {"conversation_id": "c-1"}

    def get(**params):
        response = client.get("/api/chat", params={**ask, **params}, headers={"Authorization": "Bearer t0k"})
        assert response.status_code == 200

    get(question="Frage", turn_id=turn)
    assert seen[0] == TurnRecord(turn, "c-1", "Frage", started=seen[0].started)
    assert ended == [(seen[0], "completed")]
    get(question="Frage", variant="shorter", variant_of=turn, formats="v1")  # a variant and a switch have no turn row
    get(question="Frage", variant_select="original", variant_of=turn, formats="v1")
    assert seen[1:] == [None, None] and len(ended) == 1


def chat_client(monkeypatch):
    """The app with a pipeline that answers "Antwort"."""

    async def pipeline(rt, **kwargs):
        yield fr.Chunk(content="Antwort")

    class Turns:
        async def finish(self, record, outcome):
            pass

    monkeypatch.setattr(chat_route, "pipeline", pipeline)
    client, rt = app_with()
    rt.turns, rt.handler, rt.messages_for = Turns(), None, lambda lang: {"guard.error": "Fehler"}
    rt.settings.profile = types.SimpleNamespace(guard=types.SimpleNamespace(max_question_chars=2000))
    rt.settings.card = types.SimpleNamespace(timeouts=types.SimpleNamespace(deadline_s=5, idle_s=5))
    return client


def test_the_done_frame_names_the_records_generation_and_prompts(monkeypatch) -> None:
    async def pipeline(rt, **kwargs):
        if kwargs["record"] is not None:
            kwargs["record"].run = {
                "collection": "c",
                "index_generation": 9,
                "config_hash": "h",
                "prompts": {"rag-generation": 4},
            }
        yield fr.Chunk(content="Antwort")

    client = chat_client(monkeypatch)
    monkeypatch.setattr(chat_route, "pipeline", pipeline)  # in place of chat_client's own
    auth = {"Authorization": "Bearer t0k"}

    def done(**params):
        reply = client.get(
            "/api/chat",
            params={"conversation_id": "done-6", "formats": "v1", "question": "Frage", **params},
            headers=auth,
        )
        frames = [json.loads(line[len("data: ") :]) for line in reply.text.split("\n") if line.startswith("data: ")]
        assert frames[-1]["type"] == "done"
        return frames[-1]["content"]

    try:
        assert done() == {"outcome": "completed", "generation": 9, "prompts": {"rag-generation": 4}}
        turn = "11111111-1111-4111-8111-111111111111"
        assert done(variant="shorter", variant_of=turn) == {"outcome": "completed"}  # a variant has no turn row
    finally:
        limiter.reset()


def test_rate_limit_by_client_address(monkeypatch) -> None:
    client = chat_client(monkeypatch)
    auth = {"Authorization": "Bearer t0k"}

    def get(conversation):
        return client.get("/api/chat", params={"question": "Frage", "conversation_id": conversation}, headers=auth)

    try:
        # twenty questions a minute from one address, whatever conversation ids the caller invents
        assert [get(f"rate-5a-{n}").status_code for n in range(20)] == [200] * 20
        limited = get("rate-5a-new")
        assert limited.status_code == 429
        assert limited.json() == {"error": "Rate limit exceeded: 20 per 1 minute"}
        assert limited.headers["X-Frame-Options"] == "DENY"
        assert get("rate-5a-0").status_code == 429
    finally:
        limiter.reset()  # the limiter is module-global and in memory


def test_the_answer_stream_carries_the_security_headers_and_every_frame(monkeypatch) -> None:
    client = chat_client(monkeypatch)
    reply = client.get(
        "/api/chat",
        params={"question": "Frage", "conversation_id": "stream-5a", "formats": "v1"},
        headers={"Authorization": "Bearer t0k", "Origin": "https://www.example.org"},
    )
    assert reply.status_code == 200 and reply.headers["content-type"].startswith("text/event-stream")
    assert reply.headers["X-Frame-Options"] == "DENY"
    assert reply.headers["Access-Control-Allow-Origin"] == "https://www.example.org"
    sent = [json.loads(line[len("data: ") :]) for line in reply.text.split("\n") if line.startswith("data: ")]
    assert [frame["type"] for frame in sent] == ["meta", "chunk", "done"]


def test_the_route_s_principal_reaches_the_turn_record_and_the_service(monkeypatch) -> None:
    seen = []

    async def pipeline(rt, **kwargs):
        seen.append((kwargs["principal"], kwargs["record"]))
        yield fr.Chunk(content="Antwort")

    principal = Principal(name="amt-32")
    monkeypatch.setattr(chat_route, "pipeline", pipeline)
    monkeypatch.setattr(chat_route, "ANONYMOUS", principal)
    client, rt = app_with()
    rt.turns, rt.handler, rt.messages_for = None, None, lambda lang: {"guard.error": "Fehler"}
    rt.settings.profile = types.SimpleNamespace(guard=types.SimpleNamespace(max_question_chars=2000))
    rt.settings.card = types.SimpleNamespace(timeouts=types.SimpleNamespace(deadline_s=5, idle_s=5))
    params = {"question": "Frage", "conversation_id": "c-2"}
    assert client.get("/api/chat", params=params, headers={"Authorization": "Bearer t0k"}).status_code == 200
    assert seen[0][0] is principal and seen[0][1].principal == "amt-32"


def test_the_page_lang_reaches_the_service_and_chooses_the_guard_s_texts(monkeypatch) -> None:
    seen, asked = [], []

    async def pipeline(rt, **kwargs):
        seen.append(kwargs["lang"])
        yield fr.Error(content="busy", code="busy")

    monkeypatch.setattr(chat_route, "pipeline", pipeline)
    client, rt = app_with()
    rt.turns, rt.handler = None, None
    rt.messages_for = lambda lang: (
        asked.append(lang) or {"guard.busy": {"de": "Beschäftigt", "en": "Busy"}[lang or "de"]}
    )
    rt.settings.profile = types.SimpleNamespace(guard=types.SimpleNamespace(max_question_chars=2000))
    rt.settings.card = types.SimpleNamespace(timeouts=types.SimpleNamespace(deadline_s=5, idle_s=5))
    base = {"question": "Frage", "conversation_id": "c-3"}
    got = {}
    for lang in ("en", "de", "fr", None):
        params = {**base, **({"lang": lang} if lang else {})}
        reply = client.get("/api/chat", params=params, headers={"Authorization": "Bearer t0k"})
        assert reply.status_code == 200
        got[lang] = [json.loads(x[6:]) for x in reply.text.split("\n") if x.startswith("data: ")][2]["content"]
    assert got == {"en": "Busy", "de": "Beschäftigt", "fr": "Beschäftigt", None: "Beschäftigt"}
    assert seen == ["en", "de", None, None] and asked == ["en", "de", None, None]  # a language the page lacks is none


def test_the_lifespan_starts_the_heal_task_after_a_failed_load_and_cancels_it_before_the_dispose(monkeypatch) -> None:
    events = []

    @asynccontextmanager
    async def read_pool(*args, **kwargs):
        yield None

    async def failing_reload(rt):
        raise RuntimeError("no tables yet")

    async def heal(rt):
        events.append("heal started")
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            events.append("heal cancelled")
            raise

    async def dispose_engines(*engines):
        events.append("pools disposed")
        return 0

    settings = types.SimpleNamespace(
        overlay=types.SimpleNamespace(
            tracing=types.SimpleNamespace(url="x"), database=None, server=types.SimpleNamespace(rate_limit="20/minute")
        )
    )
    runtime = types.SimpleNamespace(index=None, retiring={}, chain=None, turns=None)
    monkeypatch.setattr(app_module, "init_langfuse", lambda url: (None, False))
    monkeypatch.setattr(app_module, "read_pool", read_pool)
    monkeypatch.setattr(app_module, "store_conninfo", lambda db: "")
    monkeypatch.setattr(app_module, "build_runtime", lambda *args: runtime)
    monkeypatch.setattr(app_module, "reload", failing_reload)
    monkeypatch.setattr(app_module, "heal", heal)
    monkeypatch.setattr(app_module, "dispose_engines", dispose_engines)
    with TestClient(create_app(settings)) as client:  # pyright: ignore[reportArgumentType]
        client.get("/healthz")  # one round trip through the loop: the task has started
        assert events == ["heal started"]
    assert events == ["heal started", "heal cancelled", "pools disposed"]  # no reload can open a pool after the dispose


def chip_client(monkeypatch, roles):
    """The app with a pipeline that records the chips it is given."""
    chips: list[object] = []

    async def pipeline(rt, **kwargs):
        chips.append(kwargs["chips"])
        yield fr.Chunk(content="Antwort")

    class Turns:
        async def finish(self, record, outcome):
            pass

    monkeypatch.setattr(chat_route, "pipeline", pipeline)
    client, rt = app_with()
    rt.turns, rt.handler, rt.messages_for = Turns(), None, lambda lang: {"guard.error": "Fehler"}
    rt.settings.roles = roles
    rt.settings.profile = types.SimpleNamespace(guard=types.SimpleNamespace(max_question_chars=2000))
    rt.settings.card = types.SimpleNamespace(timeouts=types.SimpleNamespace(deadline_s=5, idle_s=5))

    def get(**params):
        return client.get(
            "/api/chat",
            params={"question": "Frage", "conversation_id": "c-chips", **params},
            headers={"Authorization": "Bearer t0k"},
        )

    return get, chips


def test_the_chat_route_passes_the_chips_and_refuses_values_the_roles_do_not_know(monkeypatch) -> None:
    from rag_core.domain.filters import Chips
    from rag_core.settings import load_profile
    from rag_core.settings.profile import Roles

    roles = load_profile(Roles, Path(__file__).resolve().parents[2] / "profiles" / "tu-informatik" / "roles.yaml")
    get, chips = chip_client(monkeypatch, roles)
    assert get().status_code == 200 and get(degree="", programme="").status_code == 200
    assert chips == [Chips(), Chips()]  # an empty selector is "not chosen"
    assert get(degree="master", programme="msc-informatik", po="2015").status_code == 200
    assert chips[-1] == Chips("master", "msc-informatik", 2015)
    for bad in (
        {"degree": "doktor"},
        {"programme": "nope"},
        {"po": "1999"},
        {"po": "x"},
        {"degree": "bachelor", "programme": "msc-informatik"},
    ):
        reply = get(**bad)
        assert reply.status_code == 422, bad
    assert (
        "unknown filter" in get(programme="nope").json()["detail"] and len(chips) == 3
    )  # nothing ran for a refused chip
    none, none_chips = chip_client(monkeypatch, None)  # a profile without roles.yaml takes no chips
    assert none(degree="master").status_code == 422 and none_chips == []
