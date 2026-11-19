"""The harness's client side: the stream frames (both forms: legacy and
format 1), a reply's frame order, the assistant and fake-LLM clients and the texts the tests expect. Here,
not in tests/integration/, because pytest's importlib mode lets a test module import an installed package
but not a sibling module."""

import json
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from rag_testing import endpoints


def _port(name: str, default: int) -> int:
    """The host port `just harness` exports in a linked worktree (Docker picked it), else the default."""
    return int(os.environ.get("RAG_CI_%s_PORT" % name) or default)


BACKEND, LLM = "http://127.0.0.1:%d" % _port("BACKEND", 28002), "http://127.0.0.1:%d" % _port("LLM", 28000)
DB_PORT = _port("PG", 25432)  # pgvector itself
STEPS = ("analysing", "understood", "searching", "found", "writing")

# The assistant's fixed texts as they appear on the wire
MSG_BUSY = (
    "Gerade sind sehr viele Anfragen gleichzeitig eingegangen. Bitte versuchen Sie es in einem Moment noch einmal."
)
MSG_NOT_READY = "Der Assistent wird gerade aktualisiert und ist in wenigen Minuten wieder für Sie da."
MSG_TIMEOUT = (
    "Die Antwort dauert gerade ungewöhnlich lange, deshalb habe ich abgebrochen. Bitte stellen Sie Ihre Frage in ein "
    "paar Minuten noch einmal."
)
TOO_LONG = (
    "Ihre Nachricht ist leider zu lang für mich (höchstens 2.000 Zeichen). Bitte kürzen Sie Ihre Frage auf das "
    "Wesentliche oder stellen Sie mehrere kürzere Fragen nacheinander, dann helfe ich Ihnen gern weiter."
)
REFUSED = "Diese Umwandlung ist nur für die letzte Antwort mit Quellen möglich."
BUSY_ERROR = MSG_BUSY  # the error frame carries the text the chunk after it shows
NOT_READY_ERROR = MSG_NOT_READY
ENGLISH_REMINDER = "Answer in English only: the whole answer, although the question and the sources are German."
NO_TURNS = "Vor der aktuellen Nachricht gibt es keine früheren Fragen und Antworten."
ONE_TURN = "Vor der aktuellen Nachricht steht eine frühere Frage mit Antwort."
NO_HISTORY = "Kein Verlauf"

# Frame orders per path: status steps named, a run of chunks as "chunk*"
LOOKUP = [
    "status:analysing",
    "status:understood",
    "status:searching",
    "status:found",
    "docs",
    "status:writing",
    "chunk*",
    "citations",
]
LOOKUP_FORMATS = [
    "status:analysing",
    "status:understood",
    "shape",
    "status:searching",
    "status:found",
    "docs",
    "status:writing",
    "chunk*",
    "citations",
]
DIRECT = ["status:analysing", "status:understood", "status:writing", "chunk*"]
DIRECT_FORMATS = ["status:analysing", "status:understood", "shape", "status:writing", "chunk*"]
VARIANT = ["shape", "status:writing", "chunk*", "citations", "variant"]

# Questions the scripted model answers, about the fixture mini-site's regulations and study office; none carries a year or a date
THESIS = "Wie lange dauert die Bearbeitung der Bachelor-Thesis?"
OFFICE_HOURS = "Wann hat das Studienbüro Sprechstunde?"
JOKE = "Erzähl mir einen Witz."  # the smoke's X16 question
RETAKE = "Wie oft kann ich eine nicht bestandene Prüfung wiederholen?"
OUT_OF_SCOPE = {"purpose": "analysis", "match": JOKE, "reply": {"intent": "out_of_scope"}}
ANSWER_ONE = {"purpose": "answer", "match": THESIS, "reply": "Antwort eins [1]."}


def frames(raw: str) -> tuple[list[dict[str, Any]], int]:
    """The complete frames of a stream and where the last one ends (the evals runner's rule: a frame ends where its JSON
    object ends). A stopped request can end inside a frame: that tail is no frame and is left out."""
    decoder, out, pos = json.JSONDecoder(), [], 0
    while (start := raw.find("data:", pos)) >= 0:
        i = start + 5
        while i < len(raw) and raw[i] in " \r\n":
            i += 1
        try:
            obj, end = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        out.append(obj)
        pos = end
    return out, pos


def serialise(items: list[dict[str, Any]]) -> str:
    """Frames in the legacy form: "data: " + json.dumps(...), Python's
    defaults, no terminator. serialise_v1 writes the form of format 1."""
    return "".join("data: " + json.dumps(item) for item in items)


# Stream format 1: a terminator after each frame, meta first, done last, code beside type and content on error and on a
# refused or failed variant. The harness tests read the legacy view (Reply.frames), so they do not depend on the format.
NEW_TYPES = ("meta", "done")


def serialise_v1(items: list[dict[str, Any]]) -> str:
    """Frames as the assistant writes them (format 1): Python's separators, ASCII, no NaN, a blank line after each."""
    return "".join("data: " + json.dumps(item, ensure_ascii=True, allow_nan=False) + "\n\n" for item in items)


def legacy(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The frames as a reader of the legacy stream sees them: no meta or done, no code."""
    return [{k: v for k, v in f.items() if k != "code"} for f in items if f.get("type") not in NEW_TYPES]


def path_of(items: list[dict[str, Any]]) -> list[str]:
    """The frame order with status steps named and chunk runs collapsed: ['status:analysing', ..., 'chunk*', ...]."""
    out: list[str] = []
    for frame in items:
        kind = str(frame.get("type"))
        if kind == "status":
            kind = "status:" + str((frame.get("content") or {}).get("step"))
        if kind == "chunk":
            if out[-1:] == ["chunk*"]:
                continue
            kind = "chunk*"
        out.append(kind)
    return out


@dataclass
class Reply:
    """One response. `stream` holds every frame as received and `raw` the text; `frames` (and everything built on it) is the
    legacy view, so the harness tests read the same frames in either format."""

    status: int
    stream: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""
    seconds: float = 0.0

    @property
    def frames(self) -> list[dict[str, Any]]:
        return legacy(self.stream)

    @property
    def types(self) -> list[str]:
        return [str(f.get("type")) for f in self.frames]

    @property
    def path(self) -> list[str]:
        return path_of(self.frames)

    @property
    def text(self) -> str:
        return "".join(str(f.get("content")) for f in self.frames if f.get("type") == "chunk")

    def first(self, kind: str) -> Any:
        return next(f.get("content") for f in self.frames if f.get("type") == kind)


class Assistant:
    """GET /api/chat. Every reply's frame order is appended to `log` (two runs of `just harness` must agree)."""

    def __init__(self, token: str, log: Path | None = None, test_name: Callable[[], str] = lambda: ""):
        self.token, self.log, self.test_name = token, log, test_name
        self.client = httpx.Client(base_url=BACKEND, timeout=60)

    def ask(self, question: str, user: str, stop_after: str | None = None, **params: str) -> Reply:
        """One request; stop_after closes the connection right after the first frame of that type (a client stop)."""
        query = {"question": question, "conversation_id": user, **params}
        started, buffer = time.monotonic(), ""
        auth = {"Authorization": "Bearer " + self.token}
        with self.client.stream("GET", endpoints.ASK, params=query, headers=auth) as resp:
            reply = Reply(resp.status_code)
            for text in resp.iter_text():
                reply.raw += text
                found, end = frames(buffer + text)
                buffer = (buffer + text)[end:]
                reply.stream += found
                if stop_after and stop_after in reply.types:
                    break
        reply.seconds = time.monotonic() - started
        if self.log is not None:
            with open(self.log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"test": self.test_name(), "path": reply.path}) + "\n")
        return reply

    def wait_ready(self, seconds: float = 240.0) -> None:
        """Until /readyz answers 200. A backend that started before the first build loads the index by itself within about
        two minutes (its retry runs every minute and a request counts as an attempt); no route makes it load sooner."""

        def ready() -> bool:
            try:
                return self.client.get(endpoints.READYZ, timeout=5).status_code == 200
            except httpx.HTTPError:
                return False

        wait_until(ready, seconds)


class FakeLLM:
    """The fake LLM's control endpoints: a per-test script, the request log, a reset."""

    def __init__(self) -> None:
        self.client = httpx.Client(base_url=LLM, timeout=10)

    def reset(self) -> None:
        self.client.post("/_reset").raise_for_status()

    def script(self, *rules: dict[str, Any]) -> None:
        self.client.post("/_script", json={"rules": list(rules)}).raise_for_status()

    def calls(self, purpose: str | None = None) -> list[dict[str, Any]]:
        recorded: list[dict[str, Any]] = self.client.get("/_calls").raise_for_status().json()
        return [c for c in recorded if purpose is None or c["purpose"] == purpose]


def build_index(root: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    """`indexer build [flags]` in a one-off container of the harness project (indexer, under the compose profile `build`),
    run from the repository root. The compose files are those of `just harness example` unless RAG_CI_COMPOSE_FILES names
    others (`-f` arguments, as `just harness` sets them for its profile)."""
    files = shlex.split(
        os.environ.get("RAG_CI_COMPOSE_FILES", "-f deploy/compose.ci.yaml -f deploy/compose.ci.example.yaml")
    )
    command = ["docker", "compose", *files, "--profile", "build", "run", "--rm", "--no-deps", "indexer"]
    return subprocess.run([*command, "build", *flags], cwd=root, capture_output=True, text=True, timeout=900)


def live_generation(db: Any) -> int | None:
    """The live generation's number from the registry, read on a psycopg connection (None before the first publish, when
    the registry does not exist)."""
    if db.execute("SELECT to_regclass('public.index_generations')").fetchone() == (None,):
        return None
    row = db.execute("SELECT max(generation) FROM index_generations WHERE status = 'live'").fetchone()
    return None if row is None or row[0] is None else int(row[0])


def messages(call: dict[str, Any]) -> list[tuple[str, str]]:
    """(role, content) of a recorded model call."""
    return [(str(m["role"]), str(m["content"])) for m in call["messages"]]


def history(fake: FakeLLM, question: str) -> str:
    """The conversation the query analysis saw before `question` (its newest call): "Kein Verlauf" or "User: …" lines."""
    call = [c for c in fake.calls("analysis") if c["question"] == question][-1]
    text = messages(call)[-1][1]
    return text.split("Gesprächsverlauf:\n", 1)[1].rsplit("\n\nNachricht: ", 1)[0]


def turn(question: str, answer: str) -> str:
    """One earlier turn as the query analysis sees it (answers without their citation marks)."""
    return "User: %s\n\nAssistant: %s" % (question, answer)


def wait_until(condition: Callable[[], bool], seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met within %g s" % seconds)
        time.sleep(0.1)


def in_thread(work: Callable[[], Any]) -> tuple[threading.Thread, dict[str, Any]]:
    """Run work() in a thread; its result or exception lands in the returned dict."""
    out: dict[str, Any] = {}

    def run() -> None:
        try:
            out["result"] = work()
        except Exception as exc:  # the test that joins the thread reports it
            out["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    return thread, out
