"""The page in Chromium. Run by `npm run test:page`.

- golden: every scenario reproduces its stored snapshots (regenerate: UPDATE_GOLDEN=1 npm run test:page).
- terminated frames: the terminated form renders exactly what the unterminated form renders (the page tolerates, acts on nothing).
- the surface: the AI disclosure; one module entry, no page function on window; no request leaves the page's
  own server; conversation ids are p-<uuid>.
- axe: no violation outside the baseline (UPDATE_AXE=1 records it).
A golden file or the baseline changes only with an intended change of the page's behaviour.
"""

import json
import os
import re
import urllib.parse
from types import ModuleType
from typing import Any

from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import Browser, expect


def test_golden(browser: Browser, folder: str, scenario: dict[str, Any], e2e: ModuleType) -> None:
    shots = e2e.as_stored(e2e.run(browser, folder, scenario))
    file = e2e.HERE / "golden" / (scenario["id"] + ".json")
    text = json.dumps(shots, ensure_ascii=False, indent=1) + "\n"
    if os.environ.get("UPDATE_GOLDEN") == "1":
        file.parent.mkdir(exist_ok=True)
        file.write_text(text, encoding="utf-8")
    assert all(not s["mismatches"] for s in shots), "every request matched its reply"
    assert text == file.read_text(encoding="utf-8")


def test_terminated_frames_render_the_same(
    browser: Browser, folder: str, scenario: dict[str, Any], e2e: ModuleType
) -> None:
    plain = e2e.comparable(e2e.run(browser, folder, scenario))
    assert e2e.comparable(e2e.run(browser, folder, scenario, e2e.terminated)) == plain


def test_the_ai_disclosure_is_shown(browser: Browser, e2e: ModuleType) -> None:
    s = e2e.Session(browser, str(e2e.PAGE))
    try:
        s.settle()
        name, note = s.page.evaluate(
            "async () => { const c = await import('./config.js'); return [c.ASSISTANT_NAME, c.TEXT.sourcesNote]; }"
        )
        expect(s.page.locator(".a-card .md").first).to_contain_text("ein KI-Assistent")
        expect(s.page.locator(".disclaimer").first).to_contain_text(
            "Studierendenprojekt, kein offizieller Dienst der TU Darmstadt."
        )
        expect(s.page.locator(".disclaimer").first).to_contain_text(
            "Alle Antworten werden KI-generiert und sind ohne Gewähr."
        )
        expect(s.page).to_have_title(name)
        expect(s.page.locator(".brand-name")).to_have_text(name)
        expect(s.page.locator(".brand-sub")).to_contain_text("TU Darmstadt")
        s.act({"ask": e2e.SOURCES_Q, "reply": e2e.lookup(e2e.SOURCES_Q, "text", e2e.TEXT_ANSWER)})
        expect(s.page.locator("section.turn .sources-note")).to_have_text(note)
        assert "KI-generiert" in note
    finally:
        s.close()


def test_the_entry_is_one_module_and_leaves_no_page_function_on_window(browser: Browser, e2e: ModuleType) -> None:
    html = (e2e.PAGE / "index.html").read_text(encoding="utf-8")
    assert re.findall(r"<script[^>]*app\.js[^>]*>", html) == ['<script type="module" src="app.js">']
    s = e2e.Session(browser, str(e2e.PAGE))
    try:
        s.settle()
        assert (
            s.page.evaluate("() => [typeof window.createFrameParser, typeof window.init, typeof window.TEXT]")
            == ["undefined"] * 3
        )
    finally:
        s.close()


def test_every_request_stays_on_the_page_server_and_ids_are_p_uuid(browser: Browser, e2e: ModuleType) -> None:
    s = e2e.Session(browser, str(e2e.PAGE))
    try:
        s.settle()
        s.act({"ask": e2e.MINDMAP_Q, "reply": e2e.lookup(e2e.MINDMAP_Q, "mindmap", e2e.MINDMAP_ANSWER)})
        s.act({"ask": e2e.NUMBERS_Q, "reply": e2e.lookup(e2e.NUMBERS_Q, "table", e2e.NUMBER_ANSWER)})
        s.act({"click": ".chart-toggle"})
        own = urllib.parse.urlsplit(s.server.base).netloc
        assert {urllib.parse.urlsplit(u).netloc for u in s.urls} - {own} == set()
        for lib in ("vendor/d3-7.9.0.min.js", "vendor/markmap-view-0.18.12.js", "vendor/chart-4.5.1.umd.min.js"):
            assert s.server.base + lib in s.urls
        ids = [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["conversation_id"][0] for u in s.server.requests]
        assert ids and all(e2e.ID_RE.match(i) for i in ids), ids
    finally:
        s.close()


AXE_STATES = {
    "first load": [],
    "lookup with sources": ["sources"],
    "open menu": ["sources", {"click": "section.turn .act-more"}],
    "version tabs": ["fv01", {"menu": "Kürzer", "reply": "formats_variants_t2"}],
    "mind map": ["mindmap"],
}


def test_no_accessibility_violation_outside_the_baseline(browser: Browser, folder: str, e2e: ModuleType) -> None:
    found = set()
    for scheme in ("light", "dark"):
        for state, steps in AXE_STATES.items():
            s = e2e.Session(browser, folder, color_scheme=scheme)
            try:
                s.settle()
                for step in steps:
                    if step == "sources":
                        step = {"ask": e2e.SOURCES_Q, "reply": e2e.lookup(e2e.SOURCES_Q, "text", e2e.TEXT_ANSWER)}
                    elif step == "mindmap":
                        step = {"ask": e2e.MINDMAP_Q, "reply": e2e.lookup(e2e.MINDMAP_Q, "mindmap", e2e.MINDMAP_ANSWER)}
                    elif step == "fv01":
                        step = {"ask": e2e.q(e2e.FV01), "reply": e2e.T(e2e.FV01)}
                    elif "reply" in step:
                        step = {**step, "reply": e2e.T(step["reply"])}
                    s.act(step)
                # CSS animations run in real time whatever the clock says: axe sees the end state, never a fade
                s.page.add_style_tag(
                    content="*, *::before, *::after { animation: none !important; transition: none !important; }"
                )
                s.page.clock.resume()  # axe-core runs on timers: the paused clock would stop it
                for v in Axe().run(s.page).response["violations"]:
                    found |= {f"{scheme} | {state} | {v['id']} | {' '.join(n['target'])}" for n in v["nodes"]}
            finally:
                s.close()
    baseline = e2e.HERE / "axe-baseline.json"
    if os.environ.get("UPDATE_AXE") == "1":
        baseline.write_text(json.dumps(sorted(found), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    assert sorted(found - set(json.loads(baseline.read_text(encoding="utf-8")))) == []
