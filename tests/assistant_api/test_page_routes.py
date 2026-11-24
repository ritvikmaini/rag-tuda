"""The page and the API beside it: the page is static and open, /api/* wants the Bearer token, /api/profile lists the
programmes and PO versions of roles.yaml and the model that answers, and the first frame of an answer names the request's limits."""

import json
import types
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from pydantic import SecretStr
from rag_core.settings import load_profile
from rag_core.settings.card import load_card
from rag_core.settings.profile import Roles

from assistant_api.api import chat as chat_route
from assistant_api.api.deps import limiter
from assistant_api.app import create_app
from assistant_api.sse import frames as fr

ROOT = Path(__file__).resolve().parents[2]
AUTH = {"Authorization": "Bearer t0k"}
POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'"
)
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
            ),
            card=load_card(card),
            roles=roles,
            profile=types.SimpleNamespace(guard=types.SimpleNamespace(max_question_chars=2000)),
        ),
    )
    app.state.runtime = rt
    return TestClient(app), rt


@pytest.fixture
def page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "page"
    for name in ("ui", "tests", ".git"):
        (folder / name).mkdir(parents=True)
    (folder / "index.html").write_text("<!DOCTYPE html><title>Assistant</title>", encoding="utf-8")
    (folder / "ui" / "start.js").write_text("export const x = 1;", encoding="utf-8")
    (folder / "tests" / "page.test.mjs").write_text("test", encoding="utf-8")
    (folder / ".env").write_text("secret", encoding="utf-8")
    (folder / ".git" / "config").write_text("secret", encoding="utf-8")
    (folder / "ui" / ".DS_Store").write_text("junk", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    monkeypatch.setenv("RAG_PAGE_DIR", str(folder))
    return folder


def test_the_page_is_open_and_the_api_wants_the_token(page: Path) -> None:
    client, _ = app_with()
    assert "<title>Assistant</title>" in client.get("/").text
    assert client.get("/ui/start.js").status_code == 200
    assert client.get("/healthz").json() == {"status": "ok"}
    for path in ("/api/profile", "/api/chat?question=Hallo&conversation_id=u"):
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401, path
    assert client.get("/api/profile", headers=AUTH).status_code == 200
    # the static mount answers only GET: a route that does not exist stays a 404 for every method
    assert client.get("/nothing.js").status_code == 404
    assert client.post("/v1/admin/reload", headers=AUTH).status_code == 404
    assert client.get("/").headers["X-Frame-Options"] == "DENY"
    assert client.get("/").headers["Content-Security-Policy"] == POLICY


def test_the_page_serves_no_dotfiles_tests_listings_or_files_outside(page: Path) -> None:
    client, _ = app_with()
    assert client.get("/index.html").status_code == 200
    for path in (
        "/.env",
        "/.git/config",
        "/ui/.DS_Store",
        "/tests/page.test.mjs",
        "/ui/",  # a folder without an index.html: no listing
        "/%2e%2e/secret.txt",
        "/..%2fsecret.txt",
        "/ui/%2e%2e/%2e%2e/secret.txt",
    ):
        assert client.get(path).status_code == 404, path


def test_the_unserved_folders_match_in_any_case(page: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reached: list[str] = []

    async def record(self, path, scope):  # a case-insensitive file system would serve /Tests/ from tests/
        reached.append(path)
        raise HTTPException(status_code=404)

    monkeypatch.setattr(StaticFiles, "get_response", record)
    client, _ = app_with()
    for path in ("/Tests/page.test.mjs", "/NODE_MODULES/x.js", "/TESTS/page.test.mjs"):
        assert client.get(path).status_code == 404 and reached == [], path
    client.get("/ui/start.js")
    assert reached == ["ui/start.js"]


def test_without_a_page_folder_the_app_serves_the_api_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PAGE_DIR", str(tmp_path / "missing"))
    client, _ = app_with()
    assert client.get("/").status_code == 404
    assert client.get("/healthz").status_code == 200


def test_profile_endpoint_lists_programmes() -> None:
    roles = load_profile(Roles, ROOT / "profiles" / "tu-informatik" / "roles.yaml")
    client, _ = app_with(roles=roles)
    body = client.get("/api/profile", headers=AUTH).json()
    assert body["degrees"] == ["bachelor", "master", "promotion"]
    by_id = {p["id"]: p for p in body["programmes"]}
    assert list(by_id) == list(roles.programmes)
    assert by_id["bsc-informatik"] == {
        "id": "bsc-informatik",
        "label_de": "B.Sc. Informatik",
        "label_en": "B.Sc. Informatik",
        "degree": "bachelor",
        "po_versions": [2023, 2015],
    }
    assert [p["po_versions"] for p in body["programmes"] if p["degree"] == "master"] == [[2023, 2015], [2023]]
    assert body["po_versions"] == [2023, 2015]
    assert body["answer"] == {"model": "Qwen/Qwen3-14B-AWQ", "provider": "vllm", "remote": False}


def test_profile_without_roles_lists_nothing_and_names_a_remote_provider() -> None:
    client, _ = app_with(GROQ_CARD)
    body = client.get("/api/profile", headers=AUTH).json()
    assert (body["degrees"], body["programmes"], body["po_versions"]) == ([], [], [])
    assert body["answer"]["provider"] == "groq" and body["answer"]["remote"] is True


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
        "deadline_s": 90.0,
        "idle_s": 5.0,
        "model": "Qwen/Qwen3-14B-AWQ",
        "provider": "vllm",
    }
