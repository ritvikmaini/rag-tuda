"""The page and the API beside it on the real backend: the page is served at / without a token, /api/* needs the Bearer token,
/api/profile lists the programmes of the profile's roles.yaml, the first frame of an answer names the request's limits and the
answering model, and the backend's log holds no question (the access log is off). Black box: HTTP and the compose CLI."""

import subprocess
import uuid
from pathlib import Path
from urllib.parse import quote, quote_plus

import httpx
import yaml
from rag_testing import endpoints, harness

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(ROOT / "deploy" / "compose.ci.yaml")]


def test_static_without_token_api_with_token(
    ready: None, tokens: dict[str, str], assistant: harness.Assistant, user: str
) -> None:
    auth = {"Authorization": "Bearer " + tokens["BACKEND_REFRESH_TOKEN"]}
    with httpx.Client(base_url=harness.BACKEND, timeout=30) as client:
        index = client.get("/")
        assert index.status_code == 200 and index.headers["content-type"].startswith("text/html")
        assert '<script type="module" src="app.js">' in index.text
        assert client.get("/app.js").status_code == 200 and client.get("/ui/start.js").status_code == 200
        assert client.get("/tests/page.test.mjs").status_code == 404  # the image carries the page, not its tests
        for path in ("/api/profile", endpoints.TOKEN_ROUTES[0][1]):
            assert client.get(path).status_code == 401, path
            assert client.get(path, headers={"Authorization": "Bearer not-the-token"}).status_code == 401, path
        assert client.get("/api/profile", headers=auth).status_code == 200
    assert assistant.ask(harness.THESIS, user).status == 200


def test_profile_endpoint_lists_programmes(tokens: dict[str, str]) -> None:
    roles = yaml.safe_load((ROOT / "profiles" / "tu-informatik" / "roles.yaml").read_text(encoding="utf-8"))
    auth = {"Authorization": "Bearer " + tokens["BACKEND_REFRESH_TOKEN"]}
    with httpx.Client(base_url=harness.BACKEND, timeout=30) as client:
        body = client.get("/api/profile", headers=auth).json()
    served = {p["id"]: (p["degree"], p["label_de"], p["po_versions"]) for p in body["programmes"]}
    expected = {
        name: (p["degree"], p["label_de"], [v["year"] for v in p["po_versions"]])
        for name, p in roles["programmes"].items()
    }
    assert served == expected and len(served) >= 3
    assert body["degrees"] == ["bachelor", "master", "promotion"] and body["po_versions"] == [2023, 2015]
    assert body["answer"] == {"model": "Qwen/Qwen3-14B-AWQ", "provider": "vllm", "remote": False}


def test_meta_carries_deadline_and_provider(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    reply = assistant.ask(harness.THESIS, user)
    meta = reply.stream[0]
    # the harness card's request limits (deploy/env/ci-fake-card.toml) and model
    assert meta["type"] == "meta" and meta["content"]["format"] == 1
    assert {k: meta["content"][k] for k in ("deadline_s", "idle_s", "model", "provider")} == {
        "deadline_s": 90.0,
        "idle_s": 5.0,
        "model": "Qwen/Qwen3-14B-AWQ",
        "provider": "vllm",
    }


def test_question_not_in_logs(ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str) -> None:
    canary = harness.THESIS.rstrip("?") + " seite" + uuid.uuid4().hex[:12] + "?"
    llm.script({"purpose": "answer", "match": canary, "reply": "Antwort [1]."})
    assert assistant.ask(canary, user).status == 200
    out = subprocess.run(
        COMPOSE + ["logs", "--no-color", "backend"], check=True, capture_output=True, text=True, timeout=120
    )
    log = out.stdout + out.stderr
    assert canary not in log and quote(canary) not in log and quote_plus(canary) not in log
    assert "GET /api/chat" not in log and "GET /api/profile" not in log  # no access-log line at all
