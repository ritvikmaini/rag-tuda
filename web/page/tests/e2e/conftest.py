"""The page's end-to-end harness: Python Playwright drives Chromium.

A local HTTP server serves one page folder (by default web/page) and answers /api/chat from a
queue, one write per frame, so a reply can stay open (stop, reload mid-answer) or be cut. The clock is paused, UUIDs are
counted, and clipboard, mailto and print are captured. Every step is stored as the body's DOM (layout numbers masked in the
stored form), the page's sessionStorage and the captures. Nothing here branches on the page folder except the files it serves.
"""

import json
import os
import re
import select
import socket
import sys
import threading
import urllib.parse
from collections.abc import Callable, Iterator
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

HERE = Path(__file__).resolve().parent  # web/page/tests/e2e/
REPO = HERE.parents[1]
PAGE = REPO
FIXTURES = REPO / "tests" / "fixtures"  # what the page's own tests read: the profile, the backend's codes
# the recorded streams, shared with the backend's contract tests (tests/contracts/)
TRANSCRIPTS = REPO.parents[1] / "tests" / "contracts" / "fixtures" / "transcripts"
VENDOR = json.loads((REPO / "vendor.json").read_text(encoding="utf-8"))["libraries"]
PAGE_DIR = os.environ.get("PAGE_DIR", str(PAGE))
# the backend's codes, as the evals runner reads them (the stream contract's fixture)
_CONTRACT = json.loads((FIXTURES / "contract_constants.json").read_text(encoding="utf-8"))
PROFILE = (FIXTURES / "profile.json").read_bytes()  # GET /api/profile: the chips and the footer read it
ERROR_CODES, OUTCOMES, VARIANT_CODES = _CONTRACT["ERROR_CODES"], _CONTRACT["OUTCOMES"], _CONTRACT["VARIANT_CODES"]
# the backend's Content-Security-Policy (tests/repo/test_page_policy.py keeps the two texts equal): every flow runs under it
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}
CONTEXT: dict[str, Any] = {
    "viewport": {"width": 1280, "height": 900},
    "locale": "de-DE",
    "timezone_id": "Europe/Berlin",
}
BERLIN = ZoneInfo("Europe/Berlin")
PROXY_ERROR = 'event: error\ndata: {"message":"Proxy-Fehler"}\n\n'  # what the proxy writes
DOC = "https://docs.example.org/handbook/agreements/working-hours/?layout=file"
TEXT_ANSWER = "Die Kernarbeitszeit endet um 15:00 Uhr [1]. Mobile Arbeit regelt die Dienstvereinbarung [2]."
# the backend's last paragraph, appended in code after the answer whatever its shape (a table, a mind map, a list)
PO_LINE = "Grundlage: Prüfungsordnung 2023 des B.Sc. Informatik"
NUMBER_ANSWER = (
    "Die Studierendenzahl ist gestiegen [1].\n\n| Jahr | Studierende |\n| --- | --- |\n| 2019 | 81.352 [1] |\n"
    "| 2020 | 82.019 [1] |\n| 2021 | 82.746 [2] |\n| 2022 | 83.590 [2] |\n\n" + PO_LINE
)
# the mind maps of the recordings are masked text: the drawn map is checked on a hand-made answer
MINDMAP_ANSWER = (
    "Die Regelungen im Überblick [1].\n\n```mindmap\n# Mobile Arbeit\n## Umfang\n- bis zu zwei Tage pro Woche [1]\n"
    "## Antrag\n- über die Führungskraft [2]\n- Vereinbarung für ein Jahr [2]\n```\n\n" + PO_LINE
)

Reply = dict[str, Any]
Step = dict[str, Any]


# ------------------------------------------------------------------ replies


def frame(kind: str, content: Any) -> str:
    return "data: " + json.dumps({"type": kind, "content": content}, ensure_ascii=False)


def transcript(name: str, **extra: Any) -> Reply:
    """A committed transcript: its frames exactly as the backend wrote them (no terminator), one write per frame."""
    text = (TRANSCRIPTS / (name + ".sse")).read_text(encoding="utf-8")
    meta = json.loads((TRANSCRIPTS / (name + ".json")).read_text(encoding="utf-8"))
    frames = ['data: {"type": ' + piece for piece in text.split('data: {"type": ') if piece]
    for f in frames:
        json.loads(f[6:])  # a cut inside a frame would not parse (a JSON string escapes its quotes)
    p = meta["params"]
    expect = {"question": p.get("question"), "variant": p.get("variant"), "variant_select": p.get("variant_select")}
    expect.update(extra.pop("expect", {}))
    return {"name": name, "frames": frames, "path": meta["path"], "expect": expect, **extra}


def lookup(question: str, shape: str, answer: str) -> Reply:
    """A lookup in the unterminated format, for what the redacted transcripts cannot show: cited sources, a table of numbers."""
    docs = [
        {
            "page_content": f"Abschnitt {n}",
            "metadata": {"download_url": DOC, "page_number": n, "title": "Dienstvereinbarung Arbeitszeit"},
        }
        for n in (2, 5)
    ]
    citations = [
        {
            "n": i + 1,
            "url": DOC,
            "page": page,
            "title": "Dienstvereinbarung Arbeitszeit",
            "type": "Dienstvereinbarung",
            "origin": "Handbook",
            "date": "2024",
            "quote": f"Zitat {i + 1}",
        }
        for i, page in enumerate((2, 5))
    ]
    frames = [
        frame("status", {"step": "analysing"}),
        frame("status", {"step": "understood", "question": question, "intent": "lookup"}),
        frame("shape", shape),
        frame("status", {"step": "searching", "queries": [question]}),
        frame("status", {"step": "found", "documents": 1, "passages": 2}),
        frame("docs", docs),
        frame("status", {"step": "writing"}),
        *(frame("chunk", answer[i : i + 24]) for i in range(0, len(answer), 24)),
        frame("citations", citations),
    ]
    return {
        "name": "lookup",
        "path": "lookup_formats",
        "frames": frames,
        "expect": {"question": question, "variant": None, "variant_select": None},
    }


def synthetic(kind: str, question: str) -> Reply:
    """Streams in the unterminated format that no transcript holds: known losses reproduced, not fixed."""
    start = [
        frame("status", {"step": "analysing"}),
        frame("status", {"step": "understood", "question": question, "intent": "lookup"}),
    ]
    frames = {
        "proxy-error": [PROXY_ERROR],
        "proxy-error-glued": [
            *start,
            frame("status", {"step": "writing"}),
            frame("chunk", "Ein Teil der Antwort.") + PROXY_ERROR,
        ],
        "guard-timeout": [
            *start,
            frame("error", "timeout"),
            frame("chunk", "\n\nDie Antwort hat zu lange gedauert. Bitte versuchen Sie es noch einmal."),
        ],
        "nan-docs": [
            *start,
            'data: {"type": "docs", "content": [{"page_content": "x", "metadata": {"page_number": NaN}}]}',
            frame("status", {"step": "writing"}),
            frame("chunk", "Eine Antwort [1]."),
            frame("citations", []),
        ],
    }[kind]
    return {
        "name": kind,
        "path": "synthetic",
        "frames": frames,
        "expect": {"question": question, "variant": None, "variant_select": None},
    }


def terminated(reply: Reply) -> Reply:
    """The terminated form: `meta` first on every response, "\\n\\n" after every
    frame, `done` last unless the client stops or the connection is cut, `code` on `error` and on a `variant` with ok false.
    The page must render it exactly as the original (it tolerates the new frames and acts on none)."""
    frames, failed = [], False
    for f in reply["frames"]:
        item = json.loads(f[6:])
        if item["type"] == "error":
            item["code"], failed = ERROR_CODES.get(str(item["content"]), "exception"), True
        if item["type"] == "variant" and not (item["content"] or {}).get("ok"):
            item["code"] = VARIANT_CODES.get(reply["path"], "failed")
        frames.append("data: " + json.dumps(item, ensure_ascii=False) + "\n\n")
    meta = {
        "format": 1,
        "turn_id": "00000000-0000-4000-8000-00000000abcd",
        "trace_id": "e2e",
        "deadline_s": 90.0,
        "idle_s": 5.0,
        "model": "Qwen/Qwen3-14B-AWQ",
        "provider": "vllm",
    }
    is_open = "hold" in reply or "cut" in reply or reply["path"] == "client_stop"
    outcome = OUTCOMES.get(reply["path"], "failed" if failed else "completed")
    done = [] if is_open else [frame("done", {"outcome": outcome}) + "\n\n"]
    shifted = {k: reply[k] + 1 for k in ("hold", "cut") if k in reply}
    return {**reply, "frames": [frame("meta", meta) + "\n\n", *frames, *done], **shifted}


# ------------------------------------------------------------------ the local server


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root: Path) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.root = root.resolve()
        self.queue: list[Reply] = []
        self.requests: list[str] = []
        self.mismatches: list[dict[str, Any]] = []
        # what the page must have read before its clock moves: the bytes of every /api/chat reply plus one for each
        # reply's end (the page counts a read that ends or fails, and response.text())
        self.sent = 0
        self.holding = False  # a held reply is open
        self.release = threading.Event()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"


class Handler(BaseHTTPRequestHandler):
    server: Server

    def log_message(self, format: str, *args: Any) -> None:  # quiet
        pass

    def do_GET(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        if url.path == "/api/profile":
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-security-policy", CONTENT_SECURITY_POLICY)
            self.send_header("content-length", str(len(PROFILE)))
            self.end_headers()
            self.wfile.write(PROFILE)
            return
        if url.path != "/api/chat":
            file = (self.server.root / urllib.parse.unquote(url.path).lstrip("/")).resolve()
            if not file.is_relative_to(self.server.root) or not file.is_file():
                self.send_error(404)
                return
            body = file.read_bytes()
            self.send_response(200)
            self.send_header("content-type", TYPES.get(file.suffix, "application/octet-stream"))
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("content-security-policy", CONTENT_SECURITY_POLICY)
            self.end_headers()
            self.wfile.write(body)
            return
        # the live-page fixture's extra parameter is left out of the record, so the golden files hold for both pages
        self.server.requests.append(url.path + "?" + url.query.replace("&chat_with_doc_use_case=false", ""))
        params = urllib.parse.parse_qs(url.query)
        got = {k: params.get(k, [None])[0] for k in ("question", "variant", "variant_select")}
        reply = self.server.queue[0] if self.server.queue else None
        if reply is None or reply["expect"] != got:
            self.server.queue[:1] = []
            self.server.mismatches.append({"want": reply and reply["expect"], "got": got})
            self.send_error(500)
            return
        data = [f.encode("utf-8") for f in reply["frames"][: reply.get("hold", reply.get("cut", len(reply["frames"])))]]
        # counted before the reply leaves the queue, so settle() never sees an empty queue with bytes still uncounted
        self.server.holding = "hold" in reply
        self.server.sent += sum(map(len, data)) + (not self.server.holding)
        self.server.queue.pop(0)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("content-security-policy", CONTENT_SECURITY_POLICY)
        self.end_headers()
        for d in data:
            self.wfile.write(d)
            self.wfile.flush()
        if "hold" in reply:  # open until the page lets go (stop, reload) or the session ends
            while not self.server.release.is_set():
                ready, _, _ = select.select([self.connection], [], [], 0.05)
                if ready and not self.connection.recv(1, socket.MSG_PEEK):
                    self.server.sent += 1  # the page's read fails (stop; a reload counts it in act())
                    break
            self.server.holding = False
        elif "cut" in reply:
            self.connection.shutdown(socket.SHUT_RDWR)
        self.close_connection = True


# ------------------------------------------------------------------ the browser side

INIT = """(() => {
    // counted UUIDs (they survive a reload, like the page's ids in sessionStorage); captures instead of side effects
    crypto.randomUUID = () => {
        const n = Number(sessionStorage.getItem("__e2e_uuid") || 0) + 1;
        sessionStorage.setItem("__e2e_uuid", String(n));
        return "00000000-0000-4000-8000-" + String(n).padStart(12, "0");
    };
    let seed = 4;   // a seeded Math.random (mulberry32): markmap names its SVG class with it
    Math.random = () => {
        seed = (seed + 0x6d2b79f5) >>> 0;
        let t = Math.imul(seed ^ (seed >>> 15), seed | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    window.__captured = { clipboard: [], mailto: [], print: 0 };
    // what the page has read of response bodies (bytes, and one per end or failure): the harness moves the clock only once
    // the page has read every frame and end the server sent (it handles them in the microtasks after its read, before any
    // evaluate can run), so every frame meets the same page time on every run without a real-time sleep
    window.__read = 0;
    const read = ReadableStreamDefaultReader.prototype.read;
    ReadableStreamDefaultReader.prototype.read = function () {
        return read.call(this).then(
            r => { window.__read += r.done ? 1 : r.value.byteLength; return r; },
            e => { window.__read += 1; throw e; });
    };
    const text = Response.prototype.text;
    Response.prototype.text = function () {
        return text.call(this).then(
            t => { window.__read += new TextEncoder().encode(t).length + 1; return t; },
            e => { window.__read += 1; throw e; });
    };
    if (navigator.clipboard) navigator.clipboard.writeText = text => { window.__captured.clipboard.push(text); return Promise.resolve(); };
    const click = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function () {
        if (this.href.startsWith("mailto:")) window.__captured.mailto.push(this.href);
        else click.call(this);
    };
    window.print = () => { window.__captured.print += 1; };
    document.addEventListener("securitypolicyviolation", e => window.__policy_violation(e.violatedDirective + " " + e.blockedURI));
})();"""

SNAPSHOT = """() => {
    // layout numbers differ between machines (fonts): the stored form masks them; raw is compared within one run only
    const GEOMETRY = /^(style|width|height|x|y|x1|x2|y1|y2|cx|cy|r|d|dx|dy|transform|viewBox|points)$/;
    const body = document.createElement("template");   // inert: a cloned <img> would load its source again
    body.innerHTML = document.body.innerHTML;
    const all = Array.from(body.content.querySelectorAll("*"));
    // attributes in name order: Chromium writes a style set through el.style into the attribute list lazily, so its place
    // among the others varies from run to run (the composer's style vs aria-invalid on Linux)
    for (const el of all) {
        const attrs = Array.from(el.attributes, a => [a.name, a.value]).sort((a, b) => (a[0] < b[0] ? -1 : 1));
        for (const [name] of attrs) el.removeAttribute(name);
        for (const [name, value] of attrs) el.setAttribute(name, value);
    }
    const raw = body.innerHTML;
    for (const el of all) {
        for (const a of Array.from(el.attributes)) {
            if (GEOMETRY.test(a.name)) el.setAttribute(a.name, a.value.replace(/-?\\d+(?:\\.\\d+)?(?:e[-+]?\\d+)?/gi, "#"));
        }
    }
    return { raw, html: body.innerHTML, storage: sessionStorage.getItem("assistant-chat-v1"),
        captured: JSON.stringify(window.__captured), title: document.title };
}"""


SESSIONS: list["Session"] = []


class Session:
    """One browser context on one page folder; the clock stands at a fixed time, 10:00 Berlin, from the first question on."""

    def __init__(
        self, browser: Browser, folder: str, query: str = "", color_scheme: str = "light", block: str = ""
    ) -> None:
        self.server = Server(Path(folder))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.context = browser.new_context(**CONTEXT, color_scheme=color_scheme)
        self.context.route(
            "https://cdn.jsdelivr.net/npm/**", self._jsdelivr
        )  # the same bytes as the vendored files, offline
        if block:
            self.context.route(lambda url: block in url, lambda route: route.abort())
        self.page: Page = self.context.new_page()
        self.pending: set[Any] = set()
        self.read_before = 0  # what earlier documents of this session read (a reload starts the page's count anew)
        self.urls: list[str] = []
        self.page_errors: list[str] = []
        self.violations: list[
            str
        ] = []  # Content-Security-Policy violations (events and console errors) of this session
        SESSIONS.append(self)
        self.page.expose_function("__policy_violation", lambda report: self.violations.append(report))
        self.page.on(
            "console", lambda m: self.violations.append(m.text) if "Content Security Policy" in m.text else None
        )
        self.page.on("request", self._sent)
        self.page.on("requestfinished", lambda request: self.pending.discard(request))
        self.page.on("requestfailed", lambda request: self.pending.discard(request))
        self.page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        self.page.add_init_script(INIT)
        self.page.clock.install(time=datetime(2026, 10, 10, 9, tzinfo=BERLIN))
        self.page.goto(self.server.base + "index.html" + query)
        self.page.clock.pause_at(datetime(2026, 10, 10, 10, tzinfo=BERLIN))

    def _sent(self, request: Any) -> None:
        self.pending.add(request)
        self.urls.append(request.url)

    @staticmethod
    def _jsdelivr(route: Any) -> None:
        lib = next((lib for lib in VENDOR if lib["source"] == route.request.url), None)
        if lib is None:
            route.fulfill(status=404)
            return
        route.fulfill(
            status=200,
            body=(PAGE / "vendor" / lib["file"]).read_bytes(),
            headers={"content-type": "text/javascript; charset=utf-8", "access-control-allow-origin": "*"},
        )

    def settle(self) -> dict[str, Any]:
        """Waits (real time, clock still) while a reply waits for its request, a file is loading or the page has not read exactly
        what the server sent, so every frame meets the same page time on every run, and while the page is still scrolling (the
        browser's smooth scroll runs in real time, not on the paused clock, and would move a later click or a clamped
        scroll position into a different moment); then runs the clock in 200 ms steps until two
        snapshots are equal. A read counter that is not a number fails at once."""
        last, ticks, read = None, 0, 0
        for _ in range(600):
            if self.server.queue or any("/api/chat" not in r.url for r in self.pending):
                self.page.wait_for_timeout(20)
                last = None
                continue
            read = self.page.evaluate("Number.isFinite(window.__read) ? window.__read : null")
            if read is None:
                raise AssertionError("the page's read counter is not a number")
            if self.read_before + read != self.server.sent:  # an overshoot waits too and ends in the error below
                self.page.wait_for_timeout(20)
                last = None
                continue
            if self._scrolling():
                last = None
                continue
            self.page.clock.run_for(200)
            now = self.page.evaluate(SNAPSHOT)
            if last and (now["raw"], now["storage"], now["captured"]) == (
                last["raw"],
                last["storage"],
                last["captured"],
            ):
                return now
            last, ticks = now, ticks + 1
            if ticks > 80:
                break
        raise AssertionError(f"the page did not settle (read {self.read_before + read}, sent {self.server.sent})")

    def _scrolling(self) -> bool:
        """True while window.scrollY moves: two real-time reads 50 ms apart differ."""
        before = self.page.evaluate("window.scrollY")
        self.page.wait_for_timeout(50)
        return self.page.evaluate("window.scrollY") != before

    def snap(self, label: str) -> dict[str, Any]:
        now = self.settle()
        return {
            "label": label,
            **now,
            "requests": list(self.server.requests),
            "mismatches": list(self.server.mismatches),
            "pageErrors": list(self.page_errors),
        }

    def act(self, step: Step) -> dict[str, Any]:
        """One step of a scenario, then its snapshot."""
        page, last = self.page, self.page.locator("section.turn").last
        if "edit" in step:
            page.locator("section.turn").nth(step["edit"]).locator(".q-edit").click(force=True)
            self.settle()
        if "reply" in step:
            self.server.queue.append(step["reply"])
        if "ask" in step:
            page.fill("#question", step["ask"])
            page.press("#question", "Enter")
        elif "menu" in step:
            last.locator(".act-more").click()
            page.clock.run_for(50)
            last.locator(".menu-item", has_text=step["menu"]).click()
        elif "tab" in step:
            last.locator(".versions .version", has_text=step["tab"]).click()
        elif "retry" in step:
            last.locator(".act", has_text="Nochmal versuchen").click()
        elif "stop" in step:
            page.click("#send")
        elif "reload" in step:
            # settle() saw the old document read everything; a held answer ends with it, uncounted by the page
            self.read_before = self.server.sent + self.server.holding
            page.reload()
        elif "click" in step and step.get("reloads"):
            # the click starts a reload after its handler returns: without waiting for the new document, settle() could run
            # a tick of the shared clock on the old one, and a run would have one tick more than another
            with page.expect_navigation():
                page.locator(step["click"]).first.click()
        elif "click" in step:
            page.locator(step["click"]).first.click()
        elif "choose" in step:  # the chips, in the order the selectors narrow each other: degree, programme, PO year
            for position, value in enumerate(step["choose"]):
                page.locator(".chip-select").nth(position).select_option(value)
        elif "type" in step:
            page.fill("#question", step["type"])
        return self.snap(step.get("label", ""))

    def close(self) -> None:
        self.server.release.set()
        self.context.close()
        self.server.shutdown()
        self.server.server_close()


def run(
    browser: Browser, folder: str, scenario: dict[str, Any], transform: Callable[[Reply], Reply] = lambda r: r
) -> list[dict[str, Any]]:
    """A whole scenario on one page folder: the snapshot after loading and after every step."""
    s = Session(browser, folder, **scenario.get("open", {}))
    try:
        shots = [s.snap("loaded")]
        for step in scenario["steps"]:
            shots.append(s.act({**step, "reply": transform(step["reply"])} if "reply" in step else step))
        return shots
    finally:
        s.close()


# ------------------------------------------------------------------ the scenarios (together they serve every transcript)

T = transcript


def asked_alone(name: str) -> bool:
    """A transcript the page can ask as a first question: no variant or version switch, and a question the composer accepts."""
    params = json.loads((TRANSCRIPTS / (name + ".json")).read_text(encoding="utf-8"))["params"]
    return (
        not {"variant", "variant_select"} & params.keys() and len(params["question"]) <= 2000 and "toolong" not in name
    )


FIRST = [name for name in sorted(p.stem for p in TRANSCRIPTS.glob("*.json")) if asked_alone(name)]
THESIS_T, CARRY_1, CARRY_2 = "plain_thesis_t1", "plain_carry_t1", "plain_carry_t2"
CHIPS_T, ENGLISH_T, DIRECT_T = "plain_chips_t1", "plain_english_t1", "plain_direct_t1"
TOO_LONG_T, CHECKLIST_T, REFUSED_T = "plain_toolong_t1", "formats_checklist_t1", "formats_refused_t1"
BUSY_T, ANSWERED_T = "busy_busy-2_t1", "busy_busy-1_t1"
S01, FV01 = THESIS_T, "formats_variants_t1"
SHORT = "Bitte lies diesen Absatz genau."
NUMBERS_Q, SOURCES_Q = "Wie hat sich die Studierendenzahl entwickelt?", "Wann endet die Kernarbeitszeit?"
MINDMAP_Q = "Wie ist die mobile Arbeit geregelt?"


def q(name: str) -> str:
    return T(name)["expect"]["question"]


def ask(name: str) -> Step:
    return {"label": name, "ask": q(name), "reply": T(name)}


def first_chunk(name: str) -> int:
    return next(i for i, f in enumerate(T(name)["frames"]) if f.startswith('data: {"type": "chunk"'))


SCENARIOS: list[dict[str, Any]] = [
    {"id": "start", "steps": []},
    {"id": "no-header", "open": {"query": "?m=nh"}, "steps": []},
    *({"id": "first-" + name, "steps": [ask(name)]} for name in FIRST),
    # the page never sends a question over its limit: the too-long reply is served to a short question to pin its rendering
    {
        "id": "too-long",
        "steps": [{"label": "too long", "ask": SHORT, "reply": T(TOO_LONG_T, expect={"question": SHORT})}],
    },
    {
        "id": "variants",
        "steps": [
            ask(FV01),
            {"label": "shorter", "menu": "Kürzer", "reply": T("formats_variants_t2")},
            {"label": "english", "menu": "Auf Englisch", "reply": T("formats_variants_t3")},
            {"label": "switch to the original", "tab": "Original", "reply": T("formats_variants_t5")},
            # the backend lost a version the page still shows: the switch fails and the notice appears
            {
                "label": "switch to a lost version",
                "tab": "Kürzer",
                "reply": T("formats_variants_t6", expect={"variant_select": "shorter"}),
            },
            # last: the drawn map watches its container, and leaving the version hides the container
            {"label": "mindmap", "menu": "Als Mindmap", "reply": T("formats_variants_t4")},
        ],
    },
    # a direct reply and a lookup without formats offer no variant (menuEntries needs citations and a shape): the refused
    # variant is asked of a formats lookup
    {
        "id": "refused",
        "steps": [ask(REFUSED_T), {"label": "refused", "menu": "Kürzer", "reply": T("formats_refused_t2")}],
    },
    {
        "id": "edit-retry",
        "steps": [
            ask(CARRY_1),
            ask(CARRY_2),
            {"label": "edit the first question", "edit": 0, "ask": q(DIRECT_T), "reply": T(DIRECT_T)},
            ask(ENGLISH_T),
            ask(BUSY_T),
            {"label": "retry", "retry": True, "reply": T(ANSWERED_T, expect={"question": q(BUSY_T)})},
        ],
    },
    {
        "id": "stop",
        "steps": [
            ask(CARRY_1),
            {"label": "streaming", "ask": q(CARRY_2), "reply": T(CARRY_2, hold=len(T(CARRY_2)["frames"]))},
            {"label": "stopped", "stop": True},
        ],
    },
    {"id": "reload-done", "steps": [ask(CHECKLIST_T), {"label": "reloaded", "reload": True}]},
    {
        "id": "reload-mid-variant",
        "steps": [
            ask(FV01),
            {
                "label": "variant streaming",
                "menu": "Kürzer",
                "reply": T("formats_variants_t2", hold=first_chunk("formats_variants_t2") + 1),
            },
            {"label": "reloaded", "reload": True},
        ],
    },
    {
        "id": "tick",
        "steps": [ask(CHECKLIST_T), {"label": "ticked", "click": ".tick"}, {"label": "reloaded", "reload": True}],
    },
    # the chips offered by the profile, chosen before the question: the request carries them
    {
        "id": "chips",
        "steps": [
            {"label": "chosen", "choose": ["bachelor", "bsc-informatik", "2015"]},
            ask(CHIPS_T),
            {"label": "next question", "ask": q(CARRY_2), "reply": T(CARRY_2)},
        ],
    },
    # the page in English: the switch reloads it, the question then carries lang=en
    {"id": "english", "steps": [{"label": "switched", "click": "#lang-switch", "reloads": True}, ask(ENGLISH_T)]},
    {
        "id": "chart",
        "synthetic": True,
        "steps": [
            {"label": "number table", "ask": NUMBERS_Q, "reply": lookup(NUMBERS_Q, "table", NUMBER_ANSWER)},
            {"label": "chart", "click": ".chart-toggle"},
            {"label": "table again", "click": ".chart-toggle"},
        ],
    },
    {
        "id": "mindmap",
        "synthetic": True,
        "steps": [{"label": "mind map", "ask": MINDMAP_Q, "reply": lookup(MINDMAP_Q, "mindmap", MINDMAP_ANSWER)}],
    },
    {
        "id": "mindmap-fallback",
        "synthetic": True,
        "open": {"block": "markmap-view"},
        "steps": [
            {
                "label": "mind map without its library",
                "ask": MINDMAP_Q,
                "reply": lookup(MINDMAP_Q, "mindmap", MINDMAP_ANSWER),
            }
        ],
    },
    {"id": "copy", "steps": [ask(S01), {"label": "copied", "click": "section.turn .act:has-text('Kopieren')"}]},
    {
        "id": "mail-print",
        "steps": [ask(S01), {"label": "mail", "menu": "Als E-Mail"}, {"label": "print", "menu": "Drucken / PDF"}],
    },
    {
        "id": "length",
        "steps": [{"label": "near the limit", "type": "x" * 1600}, {"label": "over the limit", "type": "x" * 2001}],
    },
    {
        "id": "sources",
        "synthetic": True,
        "steps": [{"label": "cited sources", "ask": SOURCES_Q, "reply": lookup(SOURCES_Q, "text", TEXT_ANSWER)}],
    },
    {"id": "cut-after-text", "steps": [{"label": "cut", "ask": q(S01), "reply": T(S01, cut=first_chunk(S01) + 3)}]},
    *(
        {
            "id": kind,
            "synthetic": True,
            "steps": [{"label": kind, "ask": "Eine Frage", "reply": synthetic(kind, "Eine Frage")}],
        }
        for kind in ("proxy-error", "proxy-error-glued", "guard-timeout", "nan-docs")
    ),
]
NEW_FORMAT = [s for s in SCENARIOS if not s.get("synthetic") and any("reply" in step for step in s["steps"])]


# ------------------------------------------------------------------ pytest wiring


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "scenario" in metafunc.fixturenames:
        chosen = NEW_FORMAT if metafunc.function.__name__.startswith("test_terminated") else SCENARIOS
        metafunc.parametrize("scenario", chosen, ids=[s["id"] for s in chosen])


@pytest.fixture(scope="session")
def browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch()
        yield chromium
        chromium.close()


@pytest.fixture(autouse=True)
def no_policy_violation() -> Iterator[None]:
    """Every flow of every test runs under the backend's policy: a session that saw a violation fails its test."""
    SESSIONS.clear()
    yield
    assert [s.violations for s in SESSIONS if s.violations] == []


@pytest.fixture
def folder() -> str:
    return PAGE_DIR


@pytest.fixture
def e2e() -> ModuleType:
    """This module: the tests reach the harness through it (pytest's importlib mode does not import a conftest by name)."""
    return sys.modules[__name__]


def as_stored(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The golden form: masked DOM, parsed storage and captures; requests, mismatches and page errors as recorded."""
    keys = ("label", "title", "html", "storage", "captured", "requests", "mismatches", "pageErrors")
    return [{k: json.loads(s[k] or "null") if k in ("storage", "captured") else s[k] for k in keys} for s in shots]


def relative_times(value: Any) -> Any:
    """`thinking.startedAt` and `endedAt` are the paused clock's time at the question and at the last frame. On a slower machine
    both land one or more 200 ms settle steps later together (a harness race, not a page difference). In every object that
    has both keys, startedAt becomes 0 and a numeric endedAt becomes endedAt - startedAt, so the duration is still compared
    and only the shared offset is dropped. A null stays null; an object with only one of the keys is untouched."""
    if isinstance(value, list):
        return [relative_times(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: relative_times(v) for k, v in value.items()}
    started, ended = out.get("startedAt"), out.get("endedAt")
    if "startedAt" in out and "endedAt" in out and isinstance(started, int | float):
        out["startedAt"] = 0
        if isinstance(ended, int | float):
            out["endedAt"] = ended - started
    return out


def relative_storage(text: str | None) -> str | None:
    """`relative_times` on the raw sessionStorage string."""
    return None if text is None else json.dumps(relative_times(json.loads(text)))


def comparable(shots: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    """The raw form for comparisons within one run. SVG `transform` values are masked: the mind map's zoom transform
    differs in the fourth decimal from run to run on one page folder (d3's transition timing). The stored thinking times are made relative."""
    return [
        {
            **{k: s[k] for k in ("label", "captured", "title", "pageErrors", *keys)},
            "storage": relative_storage(s["storage"]),
            "raw": TRANSFORM.sub('transform="#"', s["raw"]),
        }
        for s in shots
    ]


TRANSFORM = re.compile(r'transform="[^"]*"')


ID_RE = re.compile(r"^p-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
