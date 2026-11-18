"""No question text in the backend's log: three questions carrying one unique canary
(a scripted answer, an analysis failure, an answer failure) go through the page's path; the backend container's whole log then
holds neither the canary nor its URL-encoded form (the access log is off, `[QueryAnalysis]` counts, no `[FTS]` terms,
no `query[:60]` in an error), and each turn is traceable: its `ref=<first 8 characters of the turn id>` is on the `[Turn]`
line, and a failed turn's `chat.turns.error` names the code the page saw. The rule holds by the backend's own logging
configuration (apps/assistant_api/gunicorn_conf.py), checked first: no access log, every line through the ref filter, and a
formatter that prints an exception's class and frames, never its message. Black box: HTTP, SQL as rag_app, the compose CLI."""

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus

import psycopg
from rag_testing import harness

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(ROOT / "deploy" / "compose.ci.yaml")]
GUNICORN_CONF = ROOT / "apps" / "assistant_api" / "gunicorn_conf.py"


def logging_config() -> dict[str, Any]:
    """The `logconfig_dict` the backend's gunicorn config applies, read in a child process (the file configures logging on import)."""
    code = "import json, runpy, sys; print(json.dumps(runpy.run_path(sys.argv[1])['logconfig_dict']))"
    out = subprocess.run(
        [sys.executable, "-c", code, str(GUNICORN_CONF)], check=True, capture_output=True, text=True, timeout=60
    )
    return json.loads(out.stdout)


def test_the_logging_config_keeps_questions_out() -> None:
    config = logging_config()
    access = config["loggers"]["gunicorn.access"]  # UvicornWorker copies its handlers and level onto uvicorn.access
    assert access["handlers"] == [] and access["level"] == "WARNING" and access["propagate"] is False
    handler = config["handlers"]["default"]
    assert handler["filters"] == ["ref"] and config["filters"]["ref"]["()"] == "assistant_api.logref.RefFilter"
    assert config["formatters"]["default"]["()"] == "assistant_api.logref.ClassOnlyFormatter"
    assert config["root"]["handlers"] == ["default"]


def test_no_backend_line_holds_a_question(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str, db: psycopg.Connection[Any]
) -> None:
    tag = "kanarie" + uuid.uuid4().hex[:12]
    canary = "Wo steht der Fixturbrunnen " + tag
    answered, analysis_fails, answer_fails = canary + "?", canary + " zwei?", canary + " drei?"
    llm.script(
        {"purpose": "answer", "match": answered, "reply": "Antwort eins [1]."},
        {"purpose": "analysis", "match": analysis_fails, "fail": 500},
        {"purpose": "answer", "match": answer_fails, "fail": 500},
    )
    ids = {q: str(uuid.uuid4()) for q in (answered, analysis_fails, answer_fails)}
    replies = {q: assistant.ask(q, user, turn_id=turn) for q, turn in ids.items()}
    code = next(f["code"] for f in replies[answer_fails].stream if f.get("type") == "error")

    out = subprocess.run(
        COMPOSE + ["logs", "--no-color", "backend"], check=True, capture_output=True, text=True, timeout=120
    )
    log = out.stdout + out.stderr
    for text in (tag, canary, quote(canary), quote_plus(canary), quote(answered)):
        assert text not in log, "the backend's log holds the question: " + text[:40]
    for q, turn in ids.items():
        assert any("ref=" + turn[:8] in line and "[Turn] outcome=" in line for line in log.splitlines()), turn
    assert "ref=%s - [Turn] outcome=completed code=-" % ids[answered][:8] in log
    assert "ref=%s - [Turn] outcome=failed code=%s " % (ids[answer_fails][:8], code) in log
    error = db.execute("SELECT error FROM chat.turns WHERE turn_id = %s", (ids[answer_fails],)).fetchone()
    assert error is not None and (error[0] == code or str(error[0]).startswith(code + ":")), (error, code)
    assert db.execute("SELECT error FROM chat.turns WHERE turn_id = %s", (ids[answered],)).fetchone() == (None,)
