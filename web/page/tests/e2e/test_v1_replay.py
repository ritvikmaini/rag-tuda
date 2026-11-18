"""The harness-recorded v1 streams through the page.

- render: each recording (meta first, `\\n\\n` after every frame, done last, codes) is served to the page at the place its path
  has in a conversation (a first question, a variant from the menu, a version switch); the page must report no error and render
  exactly what its legacy form (no meta, done or code, no terminator) renders.
- terminated: the synthetic terminated form of the legacy frames equals the recording (ids aside), so the synthetic form the
  golden-equivalence tests use is the form the backend really writes.
"""

import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from playwright.sync_api import Browser

V1 = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts_v1"
# the native recordings carry a sidecar; the converted rec-* streams beside them serve the contract test only
NAMES = sorted(p.stem for p in V1.glob("*.json"))


def pieces(name: str) -> tuple[dict[str, Any], list[str]]:
    """The sidecar and the complete frames (a client-stop recording may end inside a frame: that tail is dropped)."""
    side = json.loads((V1 / (name + ".json")).read_text(encoding="utf-8"))
    text = (V1 / (name + ".sse")).read_text(encoding="utf-8")
    return side, [piece + "\n\n" for piece in text.split("\n\n")[:-1]]


def legacy(frames: list[str]) -> list[str]:
    """The legacy form: no meta, no done, no code, no terminator, one write per frame."""
    kept = []
    for frame in frames:
        item = json.loads(frame[6:])
        if item["type"] not in ("meta", "done"):
            item.pop("code", None)
            kept.append("data: " + json.dumps(item, ensure_ascii=False))
    return kept


def scenario(e2e: ModuleType, name: str, frames: list[str], path: str) -> dict[str, Any]:
    """The recording as the reply at its path's place in a conversation; the request it expects is the page's own."""
    T, ask = e2e.transcript, e2e.ask

    def swap(base: dict[str, Any]) -> dict[str, Any]:
        return {
            **base,
            "name": name,
            "path": path,
            "frames": frames,
            **({"cut": len(frames)} if path == "client_stop" else {}),
        }

    first = {  # path -> the committed transcript whose request the recording answers
        "lookup_formats": e2e.FV01,
        "direct": e2e.S01,
        "busy": "busy_tr-busy-1_t1",
        "not_ready": "busy_tr-busy-1_t1",
        "client_stop": e2e.S01,
    }
    if path in first:
        return {"id": name, "steps": [{"label": path, "ask": e2e.q(first[path]), "reply": swap(T(first[path]))}]}
    if path == "too_long":
        base = T("plain_tr-H5-t2_t1", expect={"question": e2e.SHORT})
        return {"id": name, "steps": [{"label": path, "ask": e2e.SHORT, "reply": swap(base)}]}
    prior = [ask(e2e.FV02 if path == "refused" else e2e.FV01)]
    shorter = {"label": "shorter", "menu": "Kürzer", "reply": T("formats_tr-FV01_t2")}
    last = {
        "variant": {"label": path, "menu": "Kürzer", "reply": swap(T("formats_tr-FV01_t2"))},
        "refused": {
            "label": path,
            "menu": "Kürzer",
            "reply": swap(T("formats_tr-refused_t2", expect={"question": e2e.q(e2e.FV02)})),
        },
        "version_switch": {"label": path, "tab": "Original", "reply": swap(T("formats_tr-FV01_t5"))},
        "version_switch_missing": {
            "label": path,
            "tab": "Kürzer",
            "reply": swap(T("formats_tr-FV01_t6", expect={"variant_select": "shorter"})),
        },
    }[path]
    # a switch to the version already on screen sends nothing: the lost-version switch needs the original shown first
    original = {"label": "switch to the original", "tab": "Original", "reply": T("formats_tr-FV01_t5")}
    steps = [
        *prior,
        *([shorter] if path.startswith("version_switch") else []),
        *([original] if path == "version_switch_missing" else []),
    ]
    return {"id": name, "steps": [*steps, last]}


def ids_aside(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A recording's meta names the turn of the harness run that recorded it, not the page's own id: the error reference
    (turn.ref in the stored turn, the digits of the reference line) is the one place the page shows it."""
    out = []
    for shot in shots:
        storage = json.loads(shot["storage"] or "null")
        for turn in (storage or {}).get("turns", []):
            turn.pop("ref", None)
        raw = re.sub(r"Referenz [0-9a-f]{8}\.", "Referenz #.", shot["raw"])
        out.append({**shot, "storage": storage, "raw": raw})
    return out


@pytest.mark.parametrize("name", NAMES)
def test_the_page_renders_a_recorded_v1_stream_as_its_legacy_form(
    browser: Browser, folder: str, e2e: ModuleType, name: str
) -> None:
    side, frames = pieces(name)
    shots = e2e.run(browser, folder, scenario(e2e, name, frames, side["path"]))
    assert all(not s["pageErrors"] and not s["mismatches"] for s in shots), name
    today = e2e.run(browser, folder, scenario(e2e, name, legacy(frames), side["path"]))
    assert ids_aside(e2e.comparable(shots)) == ids_aside(e2e.comparable(today)), name


@pytest.mark.parametrize("name", NAMES)
def test_the_synthetic_terminated_form_is_the_recording(e2e: ModuleType, name: str) -> None:
    side, frames = pieces(name)
    synthetic = e2e.terminated({"name": name, "path": side["path"], "frames": legacy(frames), "expect": {}})["frames"]

    def normal(fs: list[str]) -> list[dict[str, Any]]:
        items = [json.loads(f[6:]) for f in fs]
        for item in items:
            if item["type"] == "meta":  # ids aside (the not-ready code comes from the harness's table)
                item["content"] = {"format": item["content"]["format"]}
        return items

    assert normal(synthetic) == normal(frames), name
    assert all(f.endswith("\n\n") for f in synthetic)
