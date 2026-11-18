"""Fixtures of the evals tests: the profile, temporary runs and out dirs and a token in the environment, a fake assistant."""

import json
import os

import httpx
import pytest

PROFILE = os.path.join(os.path.dirname(__file__), "profile")  # gate.yaml and a few small probe sets
STEPS = ("analysing", "understood", "searching", "found", "writing")


def frames(*items):
    return "".join("data: " + json.dumps({"type": t, "content": c}, ensure_ascii=False) for t, c in items).encode(
        "utf-8"
    )


LOOKUP = frames(
    *[("status", {"step": s}) for s in STEPS],
    ("docs", [{"page_content": "Abschnitt", "metadata": {"url": "https://docs.example.org/a", "part_index": 0}}]),
    ("chunk", "Die Antwort [1]."),
    ("citations", [{"n": 1}]),
)
DIRECT = frames(*[("status", {"step": s}) for s in ("analysing", "understood", "writing")], ("chunk", "Gern."))
TOO_LONG = frames(("chunk", "Ihre Nachricht ist leider zu lang für mich (höchstens 2.000 Zeichen)."))


@pytest.fixture
def env(monkeypatch, tmp_path):
    """The evals test profile, tmp_path as runs_dir and out_dir, a token, no pause between requests."""
    monkeypatch.setenv("RAG__EVALS__PROFILE_DIR", PROFILE)
    monkeypatch.setenv("RAG__EVALS__RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("RAG__EVALS__OUT_DIR", str(tmp_path))
    monkeypatch.setenv("RAG__EVALS__TOKEN", "test-token")
    monkeypatch.setenv("RAG__EVALS__MIN_INTERVAL_S", "0")
    return tmp_path


@pytest.fixture
def lookup_body():
    return LOOKUP


@pytest.fixture
def fake_backend():
    """fake_backend(seen=None, error_on=None) -> a client factory. Answers like the assistant: the too-long reply above 2,000
    characters, a direct reply to "Erzähl …", an error frame for a question containing error_on, else a lookup."""

    def make(seen=None, error_on=None):
        def handler(request):
            params = dict(request.url.params)
            if seen is not None:
                seen.append(params)
            question = params["question"]
            if error_on and error_on in question:
                body = frames(("error", "Es ist ein Fehler aufgetreten."))
            elif len(question) > 2000:
                body = TOO_LONG
            elif question.startswith("Erzähl"):
                body = DIRECT
            else:
                body = LOOKUP
            return httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"}, content=body)

        return lambda: httpx.Client(transport=httpx.MockTransport(handler))

    return make


def _conv(cid, *turns):
    return {"id": cid, "turns": [t if isinstance(t, dict) else {"q": t} for t in turns]}


@pytest.fixture
def mini(env, monkeypatch, tmp_path_factory):
    """A small profile (the gate.yaml of the test profile, its own sets): gold G01 G02(2 turns) G03, general S01, history H1 (a replace
    turn), synthetic SY1-SY4 and the panels synthetic_sample (SY1 SY2), defects (SY3), past_event (SY4); core = gold + general."""
    import yaml

    profile = tmp_path_factory.mktemp("profile")
    probes = {
        "gold": [_conv("G01", "a"), _conv("G02", "a", "b"), _conv("G03", "a")],
        "general": [_conv("S01", "a")],
        "history": [_conv("H1", "a", {"q": "b", "replace_turn": 1})],
        "synthetic": [_conv("SY%d" % i, "a") for i in range(1, 5)],
    }
    (profile / "probes").mkdir()
    for name, convs in probes.items():
        (profile / "probes" / ("%s.json" % name)).write_text(json.dumps({"conversations": convs}), encoding="utf-8")
    cfg = yaml.safe_load(open(os.path.join(PROFILE, "gate.yaml"), encoding="utf-8"))
    file = lambda name: "probes/%s.json" % name  # noqa: E731
    cfg["sets"] = {
        "gold": {"file": file("gold")},
        "general": {"file": file("general")},
        "history": {"file": file("history")},
        "synthetic": {"file": file("synthetic")},
        "synthetic_sample": {"file": file("synthetic"), "ids": ["SY1", "SY2"]},
        "defects": {"file": file("synthetic"), "ids": ["SY3"]},
        "past_event": {"file": file("synthetic"), "ids": ["SY4"]},
    }
    cfg["core"], cfg["smoke"] = ["gold", "general"], {"max_ttft_s": 20, "probes": []}
    (profile / "gate.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("RAG__EVALS__PROFILE_DIR", str(profile))
    return profile
