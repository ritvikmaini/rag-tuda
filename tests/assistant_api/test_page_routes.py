"""The first frame of an answer names the request's limits and the model that answers."""

import json
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from rag_core.settings.card import load_card
from rag_core.settings.profile import Roles

from assistant_api.api import chat as chat_route
from assistant_api.api.deps import limiter
from assistant_api.app import create_app
from assistant_api.sse import frames as fr

ROOT = Path(__file__).resolve().parents[2]
AUTH = {"Authorization": "Bearer t0k"}
FAKE_CARD = ROOT / "deploy" / "env" / "ci-fake-card.toml"
GROQ_CARD = ROOT / "models" / "cards" / "l-groq-gpt-oss-120b.toml"


def app_with(card: Path = FAKE_CARD, roles: Roles | None = None) -> tuple[TestClient, types.SimpleNamespace]:
    app = create_app()
    rt = types.SimpleNamespace(
        chain=None,
        index=None,
        langfuse=None,
        settings=types.SimpleNamespace(
            overlay=types.SimpleNamespace(
                server=types.SimpleNamespace(token=SecretStr("t0k")),
                http=types.SimpleNamespace(cors_domain="example.org"),
            ),
            card=load_card(card),
            roles=roles,
            profile=types.SimpleNamespace(guard=types.SimpleNamespace(max_question_chars=2000)),
        ),
    )
    app.state.runtime = rt
    return TestClient(app), rt


def test_meta_carries_deadline_and_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    async def pipeline(rt, **kwargs):
        yield fr.Chunk(content="Antwort")

    class Turns:
        async def finish(self, record, outcome):
            pass

    monkeypatch.setattr(chat_route, "pipeline", pipeline)
    client, rt = app_with()
    rt.turns, rt.handler, rt.messages_for = Turns(), None, lambda lang: {"guard.error": "Fehler"}
    try:
        reply = client.get("/api/chat", params={"question": "Frage", "conversation_id": "meta-1"}, headers=AUTH)
    finally:
        limiter.reset()
    first = json.loads(reply.text.split("\n")[0][len("data: ") :])
    assert first["type"] == "meta"
    assert first["content"] == {
        "format": 1,
        "turn_id": first["content"]["turn_id"],
        "trace_id": None,
        "deadline_s": 25.0,
        "idle_s": 5.0,
        "model": "Qwen/Qwen3-14B-AWQ",
        "provider": "vllm",
    }
