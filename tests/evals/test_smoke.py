"""evals smoke: five fixed probes, structure only; exit 0, 10, or 2 when it cannot run."""

import json

from evals import smoke

LOOKUP_ROW = {
    "checks": {"errors": []},
    "first_token_s": 6.0,
    "event_types": ["status"] * 5 + ["docs", "chunk", "citations"],
}


def test_a_well_formed_assistant_passes_all_six_requests(env, fake_backend, capsys):
    assert smoke.main(["--label", "t"], client_factory=fake_backend()) == 0
    (folder,) = [p for p in env.iterdir() if p.name.startswith("t-smoke_")]
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["id"], r["smoke"]) for r in rows] == [
        ("G01", []),
        ("F01", []),
        ("X16", []),
        ("H5", []),
        ("H2", []),
        ("H2", []),
    ]
    assert "RESULT PASS" in capsys.readouterr().out


def test_a_failed_check_exits_10(env, fake_backend):
    assert smoke.main(["--label", "t", "--max-ttft", "0"], client_factory=fake_backend()) == 10  # a forced failure
    assert (
        smoke.main(["--label", "u"], client_factory=fake_backend(error_on="Kreditkarte")) == 10
    )  # G01 gets an error frame


def test_without_a_token_the_smoke_cannot_run(env, fake_backend, monkeypatch):
    monkeypatch.delenv("RAG__EVALS__TOKEN")
    assert smoke.main([], client_factory=fake_backend()) == 2


def test_the_structural_rules_per_kind():
    steps = list(smoke.STEPS)
    assert smoke.structural("lookup", True, LOOKUP_ROW, steps, 20) == []
    assert smoke.structural(
        "lookup", True, dict(LOOKUP_ROW, event_types=["status"] * 5 + ["docs", "chunk"]), steps, 20
    ) == ["no citations"]
    assert smoke.structural("lookup", True, LOOKUP_ROW, steps[:4], 20) == ["status steps"]
    assert smoke.structural("direct", True, LOOKUP_ROW, steps, 20) == ["docs"]
    assert smoke.structural("too_long", True, dict(LOOKUP_ROW, event_types=["status", "chunk"]), [], 20) == [
        "frames status,chunk"
    ]
    assert smoke.structural("edit", True, dict(LOOKUP_ROW, event_types=["status"]), [], 20) == ["no chunk"]
    assert smoke.structural("edit", False, dict(LOOKUP_ROW, checks={"errors": ["http 500: x"]}), [], 20) == ["error"]
    assert smoke.structural("direct", True, dict(LOOKUP_ROW, first_token_s=20.0, event_types=["chunk"]), [], 20) == [
        "ttft"
    ]
    v1 = ["meta", "chunk", "done"]
    assert smoke.structural("too_long", True, dict(LOOKUP_ROW, event_types=v1), [], 20) == []
    assert smoke.structural(
        "too_long", True, dict(LOOKUP_ROW, event_types=["meta", "status", "chunk", "done"]), [], 20
    ) == ["frames status,chunk"]


def test_an_empty_token_is_no_token(env, fake_backend, monkeypatch):
    monkeypatch.setenv("RAG__EVALS__TOKEN", " ")
    assert smoke.main([], client_factory=fake_backend()) == 2
