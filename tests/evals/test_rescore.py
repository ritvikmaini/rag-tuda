"""evals rescore: every input evaluate_turn had at run time is rebuilt from the row."""

import json
import os

from evals import rescore
from evals.gate import abort_turns, conversations, load_config


def write_run(folder, rows):
    folder.mkdir()
    (folder / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return str(folder)


def row(cid, turn, answer, failed, expect=None, **extra):
    return dict(
        {
            "id": cid,
            "turn": turn,
            "answer": answer,
            "sources": [],
            "expect": expect or {},
            "checks": {"failed": failed, "errors": []},
        },
        **extra,
    )


def test_rescore_writes_a_sibling_run_with_current_checks(env, tmp_path):
    run = write_run(
        tmp_path / "general-x_20260928-120000",
        [
            row(
                "T1",
                4,
                "Seit 2024 明确 gilt sie, siehe https://docs.example.org/a/?layout=file",
                ["leak"],
                {"contains_any": ["2024"]},
                sources=[{"url": "https://docs.example.org/a/?layout=file", "title": "A", "internal": True}],
            )
        ],
    )
    assert rescore.main([run]) == 0
    new = json.loads((tmp_path / "general-x_20260928-120000_rescored" / "results.jsonl").read_text(encoding="utf-8"))
    assert new["checks"]["failed"] == ["non_latin"] and new["answer"].startswith("Seit 2024")


def test_meta_is_rebuilt_for_shape_and_variant_turns():
    table = "Vorher [1].\n\n| a | b |\n| --- | --- |\n| 1 [1] | 2 [1] |"
    rows = [
        row("FV01", 6, table, [], {"variant_ok": True, "shape": "table", "cited_min": 0.8}, shape="table"),
        row("FV01", 7, table, ["variant"], {"variant_ok": True, "cited_min": 0.8}, shape="table"),
        row("FV02", 2, table, [], {"variant_ok": True}, shape="table", variant_ok=True),
    ]
    new, changed = rescore.rescore_rows(rows, set())
    assert changed == 0 and [r["checks"]["failed"] for r in new] == [[], ["variant"], []]


def test_a_client_stop_stays_a_passed_set_up_step(env):
    profile = os.environ["RAG__EVALS__PROFILE_DIR"]
    aborts = abort_turns(conversations(profile, load_config(profile)))
    assert ("H1", 2) in aborts
    new, changed = rescore.rescore_rows([row("H1", 2, "", [], {})], aborts)
    assert changed == 0 and new[0]["checks"]["auto_pass"] is True


def test_check_writes_nothing_and_fails_on_a_changed_turn(env, tmp_path, capsys):
    same = write_run(tmp_path / "a_20261008-100000", [row("G01", 1, "Antwort [1].", [])])
    stale = write_run(tmp_path / "b_20261008-100000", [row("G02", 1, "Antwort [1].", ["contains_any"])])
    assert rescore.main(["--check", same]) == 0
    assert rescore.main(["--check", same, stale]) == 1
    assert "rescored 1 turns (0 changed)" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a_20261008-100000", "b_20261008-100000"]


def test_the_rescored_copy_goes_into_out_dir_not_next_to_the_run(env, tmp_path_factory, monkeypatch):
    out = tmp_path_factory.mktemp("out")  # in the container the runs are a read-only mount
    monkeypatch.setenv("RAG__EVALS__OUT_DIR", str(out))
    run = write_run(env / "g_20261008-100000", [row("G01", 1, "Antwort [1].", [])])
    assert rescore.main([run]) == 0
    assert (out / "g_20261008-100000_rescored" / "results.jsonl").is_file()
    assert sorted(p.name for p in env.iterdir()) == ["g_20261008-100000"]
