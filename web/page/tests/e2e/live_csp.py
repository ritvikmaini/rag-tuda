"""The page served by the real backend (the harness stack kept up with HARNESS_KEEP=1 just harness example), loaded in Chromium
with its Content-Security-Policy: no violation on load nor for an answer with a table, a mind map and a chart (the harness's fake
LLM writes the answer). Run from the repository root:

    uv run --frozen --project web/page/tests/e2e python web/page/tests/e2e/live_csp.py

The ports come from RAG_CI_BACKEND_PORT and RAG_CI_LLM_PORT (a linked worktree; see deploy/ci/.generated/harness.env), the token
from deploy/ci/.generated/ci.env."""

import json
import os
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

GEN = Path(__file__).resolve().parents[4] / "deploy" / "ci" / ".generated"
BACKEND = "http://127.0.0.1:" + os.environ.get("RAG_CI_BACKEND_PORT", "28002")
LLM = "http://127.0.0.1:" + os.environ.get("RAG_CI_LLM_PORT", "28000")
QUESTION = "Wie lange dauert die Bearbeitung der Bachelor-Thesis?"
ANSWER = (
    "Die Bearbeitung dauert fünf Monate [1].\n\n| Jahr | Abschlussarbeiten |\n| --- | --- |\n| 2021 | 81 [1] |\n"
    "| 2022 | 83 [1] |\n| 2023 | 87 [1] |\n\n```mindmap\n# Bachelor-Thesis\n## Dauer\n- fünf Monate [1]\n"
    "## Anmeldung\n- beim Prüfungsausschuss [1]\n```"
)
CONTROL = (
    "http://csp-control.invalid/x.png"  # blocked by the policy before any lookup: proves the detector sees a violation
)
RECORD = """() => { window.__csp = [];
  document.addEventListener("securitypolicyviolation", e => window.__csp.push(e.violatedDirective + " " + e.blockedURI)); }"""


def post(path: str, body: dict) -> None:
    request = urllib.request.Request(LLM + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(request, timeout=10).read()


def main() -> None:
    token = dict(
        line.split("=", 1) for line in (GEN / "ci.env").read_text(encoding="utf-8").splitlines() if "=" in line
    )["BACKEND_REFRESH_TOKEN"]
    post("/_reset", {})
    post("/_script", {"rules": [{"purpose": "answer", "match": QUESTION, "reply": ANSWER}]})
    seen: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.add_init_script("(" + RECORD + ")()")
        page.on("console", lambda m: seen.append(m.type + ": " + m.text))
        page.on("pageerror", lambda e: seen.append("pageerror: " + str(e)))
        page.on("requestfailed", lambda r: seen.append("requestfailed: " + r.url))
        reply = page.goto(BACKEND + "/#" + token)
        policy = reply.headers.get("content-security-policy", "") if reply else ""
        assert policy, "the page came without a Content-Security-Policy"
        page.wait_for_selector("#question:not([disabled])")
        page.evaluate("src => { const i = new Image(); i.src = src; }", CONTROL)
        page.wait_for_function("window.__csp.length > 0")
        assert page.evaluate("window.__csp") == ["img-src " + CONTROL], "the detector missed the control"
        page.evaluate("window.__csp.length = 0")
        seen.clear()
        page.fill("#question", QUESTION)
        page.click("#send")
        page.wait_for_selector("section.turn .md table", timeout=60000)
        page.wait_for_selector("svg.mindmap-svg", timeout=60000)
        page.click(".chart-toggle")
        page.wait_for_selector(".md canvas", timeout=15000)
        with page.expect_download() as download:  # the mind map's SVG export: a blob: link
            page.click(".mindmap-tools .act >> nth=1")
        assert download.value.suggested_filename == "mindmap.svg"
        violations = page.evaluate("window.__csp") + [m for m in seen if "Content Security Policy" in m]
        browser.close()
    print("policy:", policy)
    print("console:", seen)
    assert violations == [], violations
    print("OK: table, mind map, chart and the SVG export with no policy violation")


if __name__ == "__main__":
    main()
