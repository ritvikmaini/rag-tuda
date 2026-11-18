"""evals gate on synthetic runs of the real gold and general ids."""

import json
import os

import pytest

from evals import gate

SECTIONS = ["https://x/%d|0|h%d" % (i, i) for i in range(14)]


def row(cid, turn=1, ok=True, ttft=6.0, total=8.0, sections=SECTIONS, errors=(), generation=None, prompts=None):
    found = {
        "id": cid,
        "turn": turn,
        "first_token_s": ttft,
        "total_s": total,
        "sources": [{"url": "u", "title": "", "internal": False, "section": s} for s in sections],
        "checks": {"auto_pass": ok, "errors": list(errors), "failed": [] if ok else ["contains_any"]},
    }
    if generation is not None:  # older rows have neither key
        found["generation"] = generation
    if prompts is not None:
        found["prompts"] = prompts
    return found


def write(out, name, rows):
    folder = out / name
    folder.mkdir()
    (folder / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(folder)


GOLD = [row("G01"), row("G02"), row("G03", ok=False)]


def test_identical_runs_pass_and_the_report_is_written_into_out_dir(env, tmp_path_factory, monkeypatch):
    out = tmp_path_factory.mktemp("out")  # the container's own folder; the runs sit in the read-only mount (runs_dir)
    monkeypatch.setenv("RAG__EVALS__OUT_DIR", str(out))
    write(env, "r-gold_20261008-100000", GOLD)
    write(env, "c-gold_20261008-110000", GOLD)
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 0
    report = json.loads(next(out.glob("c-gate_*.json")).read_text(encoding="utf-8"))
    assert report["exit"] == 0 and "gpu" not in report["checks"]
    assert sorted(p.name for p in env.iterdir()) == ["c-gold_20261008-110000", "r-gold_20261008-100000"]


def test_labels_take_the_newest_run_per_set_and_never_a_re_run_folder(env):
    write(env, "r-gold_20261008-090000", [row("G01", ok=False), row("G02"), row("G03", ok=False)])
    write(env, "r-gold_20261008-100000", GOLD)
    write(env, "c-gold_20261008-110000", GOLD)
    write(env, "c-rerun1-gold_20261008-120000", [row("G01", ok=False)])
    assert gate.newest_runs(str(env), "c", {"gold"}) == {"gold": str(env / "c-gold_20261008-110000")}
    assert gate.newest_runs(str(env), "r", {"gold"}) == {"gold": str(env / "r-gold_20261008-100000")}
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 0


def test_a_flipped_turn_asks_for_two_re_runs_with_exit_2(env, capsys):
    write(env, "r-gold_20261008-100000", GOLD)
    write(env, "c-gold_20261008-110000", [row("G01"), row("G02", ok=False), row("G03", ok=False)])
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 2
    out = capsys.readouterr().out
    assert "  evals run gold --ids G02 --label c-rerun1" in out and "  evals run gold --ids G02 --label c-rerun2" in out


def test_a_turn_failing_two_of_three_runs_is_a_regression(env):
    write(env, "r-gold_20261008-100000", GOLD)
    write(env, "c-gold_20261008-110000", [row("G01"), row("G02", ok=False), row("G03", ok=False)])
    write(env, "c-rerun1-gold_20261008-120000", [row("G02", ok=False)])
    write(env, "c-rerun2-gold_20261008-121000", [row("G02")])
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 1


def test_a_turn_passing_two_of_three_runs_counts_as_passed(env):
    write(env, "r-gold_20261008-100000", GOLD)
    write(env, "c-gold_20261008-110000", [row("G01"), row("G02", ok=False), row("G03", ok=False)])
    write(env, "c-rerun1-gold_20261008-120000", [row("G02")])
    write(env, "c-rerun2-gold_20261008-121000", [row("G02")])
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 0


def test_a_re_run_older_than_the_candidate_does_not_count(env):
    write(env, "r-gold_20261008-100000", GOLD)
    write(env, "c-rerun1-gold_20261008-103000", [row("G02")])  # left by an earlier attempt
    write(env, "c-rerun2-gold_20261008-104000", [row("G02")])
    write(env, "c-gold_20261008-110000", [row("G01"), row("G02", ok=False), row("G03", ok=False)])
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 2


def test_flaky_turns_are_left_out(env):
    ref = write(env, "r-general_20261008-100000", [row("C1", 1), row("S01", 1)])
    cand = write(env, "c-general_20261008-110000", [row("C1", 1, ok=False), row("S01", 1)])
    assert gate.main(["--cand", cand, "--ref", ref]) == 0


def test_latency_fails_above_the_reference_median_plus_the_margin(env):
    ref = write(env, "r-gold_20261008-100000", GOLD)
    at = write(env, "c-gold_20261008-110000", [dict(r, first_token_s=6.30) for r in GOLD])
    over = write(env, "d-gold_20261008-110000", [dict(r, first_token_s=6.31) for r in GOLD])
    assert gate.main(["--cand", at, "--ref", ref]) == 0
    assert gate.main(["--cand", over, "--ref", ref]) == 1


def test_retrieval_below_the_floor_or_an_incomplete_run_fails(env):
    ref = write(env, "r-gold_20261008-100000", GOLD)
    moved = write(
        env,
        "c-gold_20261008-110000",
        [row("G01", sections=SECTIONS[:7] + ["y%d" % i for i in range(7)]), row("G02"), row("G03", ok=False)],
    )
    short = write(env, "d-gold_20261008-110000", GOLD[:2])
    assert gate.main(["--cand", moved, "--ref", ref]) == 1  # (0.5 + 1 + 1) / 3 = 0.83 < the hard floor 0.97
    assert gate.main(["--cand", short, "--ref", ref]) == 1


def test_an_error_or_a_hang_fails(env):
    ref = write(env, "r-gold_20261008-100000", GOLD)
    erred = write(env, "c-gold_20261008-110000", [row("G01", errors=["Fehler"]), GOLD[1], GOLD[2]])
    hung = write(env, "d-gold_20261008-110000", [row("G01", total=125.5), GOLD[1], GOLD[2]])
    assert gate.main(["--cand", erred, "--ref", ref]) == 1
    assert gate.main(["--cand", hung, "--ref", ref]) == 1


def test_a_folder_mixing_sets_is_refused(env):
    profile = os.environ["RAG__EVALS__PROFILE_DIR"]
    index = gate.conversations(profile, gate.load_config(profile))
    mixed = write(env, "m_20261008-100000", [row("G01"), row("F01")])
    with pytest.raises(ValueError, match="not one probe set"):
        gate.set_of(gate.read_rows(mixed), index, mixed)


def test_a_pending_turn_whose_re_runs_lack_it_still_needs_re_runs(env, capsys):
    # two re-run folders exist but neither holds the flipped turn; the numbering counts the turn's own runs,
    # so it asks for its first and second
    write(env, "r-gold_20261008-100000", GOLD)
    write(env, "c-gold_20261008-110000", [row("G01"), row("G02", ok=False), row("G03", ok=False)])
    write(env, "c-rerun1-gold_20261008-120000", [row("G01")])
    write(env, "c-rerun2-gold_20261008-121000", [row("G01")])
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 2
    out = capsys.readouterr().out
    assert "  evals run gold --ids G02 --label c-rerun1" in out and "  evals run gold --ids G02 --label c-rerun2" in out


def test_agreement_between_the_hard_floor_and_the_noise_floor_asks_for_explanations(env, capsys):
    ref = write(env, "r-gold_20261008-100000", GOLD)
    cand = write(env, "c-gold_20261008-110000", [row("G01", sections=SECTIONS[:13] + ["other|0|h"]), GOLD[1], GOLD[2]])
    assert gate.main(["--cand", cand, "--ref", ref]) == 3  # (13/14 + 1 + 1) / 3 = 0.976: in [0.97, 0.9852)
    out = capsys.readouterr().out
    assert "retrieval  EXPLAIN" in out and "differing gold G01 overlap 0.9286 (needs an explanation)" in out
    assert "RESULT EXPLAIN" in out


def test_re_runs_come_before_explanations(env):
    write(env, "r-gold_20261008-100000", GOLD)
    write(
        env,
        "c-gold_20261008-110000",
        [row("G01", sections=SECTIONS[:13] + ["other|0|h"]), row("G02", ok=False), row("G03", ok=False)],
    )
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 2


def test_volatile_questions_are_reported_and_never_block(env, capsys):
    general = [row("C1"), row("A6"), row("S01")]
    ref = write(env, "r-general_20261008-100000", general)
    cand = write(
        env,
        "c-general_20261008-110000",
        [row("C1", sections=["other%d|0|h" % i for i in range(14)]), general[1], general[2]],
    )
    assert gate.main(["--cand", cand, "--ref", ref]) == 0  # C1 counted would give 0.67, a hard fail
    out = capsys.readouterr().out
    assert (
        "volatile general C1 overlap 0.0000 (reported, not counted)" in out and "retrieval  PASS  pooled 1.0000" in out
    )


def test_usage_errors_exit_64_and_an_unreadable_run_70_through_the_console_script(env):
    from evals import cli

    assert cli.main(["gate"]) == 64  # neither labels nor folders
    assert cli.main(["gate", "c", "--against", "r"]) == 64  # labels without runs
    old = write(
        env, "o-gold_20261008-100000", [dict(row("G01"), sources=[{"url": "u", "title": "", "internal": False}])]
    )
    assert cli.main(["gate", "--cand", old, "--ref", old]) == 70  # recorded before section identities


def test_a_re_run_folder_counts_once_however_often_it_is_named(env):
    # a re-run found by its label and named again with --reruns, a folder named twice, or the candidate
    # named as a re-run must not add runs: one real re-run of a flipped turn still needs a second one (exit 2, not 1)
    write(env, "r-gold_20261008-100000", GOLD)
    cand = write(env, "c-gold_20261008-110000", [row("G01"), row("G02", ok=False), row("G03", ok=False)])
    rerun = write(env, "c-rerun1-gold_20261008-120000", [row("G02", ok=False)])
    assert gate.main(["c", "--against", "r", "--sets", "gold", "--reruns", rerun]) == 2
    assert gate.main(["c", "--against", "r", "--sets", "gold", "--reruns", rerun + "/", cand]) == 2
    ref = str(env / "r-gold_20261008-100000")
    assert gate.main(["--cand", cand, "--ref", ref, "--reruns", rerun, rerun]) == 2


# --- provenance: a pair across index generations or a side mixing prompt versions is void


def provenance_line(capsys):
    return next(line for line in capsys.readouterr().out.splitlines() if line.startswith("provenance"))


def test_rows_on_two_generations_void_the_pair_with_exit_4(env, capsys):
    write(env, "r-gold_20261008-100000", [row(i, generation=41) for i in ("G01", "G02", "G03")])
    write(
        env, "c-gold_20261008-110000", [row("G01", generation=41), row("G02", generation=41), row("G03", generation=42)]
    )
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 4
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith("provenance VOID"))
    assert "41 (5 rows)" in line and "42 (1 rows)" in line and "RESULT VOID" in out
    report = json.loads(next(env.glob("c-gate_*.json")).read_text(encoding="utf-8"))
    assert report["exit"] == 4 and report["checks"]["provenance"]["void"] is True
    assert report["checks"]["provenance"]["generations"] == {"41": 5, "42": 1}


def test_a_void_pair_exits_4_also_when_the_candidate_would_fail_the_answers(env):
    write(env, "r-gold_20261008-100000", [row(i, generation=41) for i in ("G01", "G02", "G03")])
    write(
        env,
        "c-gold_20261008-110000",
        [row("G01", generation=42), row("G02", ok=False, generation=42), row("G03", errors=["x"])],
    )
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 4


def test_a_prompt_version_off_its_sides_majority_voids(env, capsys):
    write(env, "r-gold_20261008-100000", GOLD)
    cand = [
        row("G01", prompts={"rag-generation": 5}),
        row("G02", prompts={"rag-generation": 5}),
        row("G03", prompts={"rag-generation": 6}),
    ]
    write(env, "c-gold_20261008-110000", cand)
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 4
    line = provenance_line(capsys)
    assert line.startswith("provenance VOID") and "candidate rag-generation v6 off v5" in line and "gold G03 t1" in line


def test_a_prompt_not_yet_served_by_the_process_is_no_mix(env, capsys):
    # null = Langfuse has not served that prompt in this backend process yet; it names no text, so it is no version
    write(env, "r-gold_20261008-100000", GOLD)
    cand = [
        row("G01", prompts={"rag-generation": 5, "no-docs-found": None}),
        row("G02", prompts={"rag-generation": 5, "no-docs-found": None}),
        row("G03", prompts={"rag-generation": 5, "no-docs-found": 3}),
    ]
    write(env, "c-gold_20261008-110000", cand)
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 0
    line = provenance_line(capsys)
    assert line.startswith("provenance OK") and "no-docs-found v3" in line


def test_two_served_versions_still_void_beside_a_not_yet_served_one(env, capsys):
    write(env, "r-gold_20261008-100000", GOLD)
    cand = [
        row("G01", prompts={"no-docs-found": None}),
        row("G02", prompts={"no-docs-found": 2}),
        row("G03", prompts={"no-docs-found": 3}),
    ]
    write(env, "c-gold_20261008-110000", cand)
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 4
    assert "no-docs-found" in provenance_line(capsys)


def test_a_prompt_publish_between_reference_and_candidate_is_no_void(env, capsys):
    write(env, "r-gold_20261008-100000", [row(r["id"], prompts={"rag-generation": 5}) for r in GOLD])
    write(env, "c-gold_20261008-110000", [row(r["id"], prompts={"rag-generation": 6}) for r in GOLD])
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 0
    line = provenance_line(capsys)
    assert (
        line.startswith("provenance OK")
        and "reference rag-generation v5" in line
        and "candidate rag-generation v6" in line
    )


def test_unrecorded_rows_never_void(env, capsys):
    write(env, "r-gold_20261008-100000", GOLD)
    write(env, "c-gold_20261008-110000", [row(r["id"], generation=41) for r in GOLD])
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 0
    line = provenance_line(capsys)
    assert line.startswith("provenance OK") and "generation 41" in line and "unrecorded 3 of 6 rows" in line


def test_rows_without_prompts_take_no_part(env, capsys):
    write(env, "r-gold_20261008-100000", GOLD)
    cand = [
        row("G01", prompts={"rag-generation": 5}),
        row("G02", prompts=None),
        row("G03", prompts={}),
    ]  # an early exit has none
    write(env, "c-gold_20261008-110000", cand)
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 0
    assert provenance_line(capsys).startswith("provenance OK")


def test_re_run_rows_count_for_the_candidate_side(env):
    write(env, "r-gold_20261008-100000", [row(r["id"], generation=41, prompts={"p": 1}) for r in GOLD])
    write(
        env,
        "c-gold_20261008-110000",
        [
            row("G01", generation=41, prompts={"p": 1}),
            row("G02", ok=False, generation=41, prompts={"p": 1}),
            row("G03", ok=False, generation=41, prompts={"p": 1}),
        ],
    )
    write(env, "c-rerun1-gold_20261008-120000", [row("G02", generation=42, prompts={"p": 1})])  # a build ran in between
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 4


def test_a_usage_error_comes_before_a_void_pair(env):
    from evals import cli

    write(env, "r-gold_20261008-100000", [row(i, generation=41) for i in ("G01", "G02", "G03")])
    write(env, "c-gold_20261008-110000", [row(i, generation=42) for i in ("G01", "G02", "G03")])
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 4
    assert cli.main(["gate", "c", "--against", "r", "--sets", "gold,history"]) == 64  # no history run


def test_a_re_run_of_an_undeclared_set_is_a_usage_error(mini, env):
    from evals import cli

    for name in ("r", "c"):
        write(env, "%s-gold_20261008-%s" % (name, "100000" if name == "r" else "110000"), GOLD)
    stray = write(env, "c-rerun1-general_20261008-120000", [row("S01", errors=["x"])])
    assert cli.main(["gate", "c", "--against", "r", "--sets", "gold"]) == 0
    assert cli.main(["gate", "c", "--against", "r", "--sets", "gold", "--reruns", stray]) == 64


# the gate scores only the declared sets, a panel is a view on a file, a failed turn is replayed


def row_ids(cid, turn, ok=True):
    return dict(row(cid, turn, ok=ok), user="probe-c-%s" % cid, turn_id="id-%s-%d" % (cid, turn))


def test_sets_default_to_core_and_an_undeclared_set_of_the_label_is_ignored(mini, env):
    for label, stamp in (("r", "100000"), ("c", "110000")):
        write(env, "%s-gold_20261008-%s" % (label, stamp), [row("G01"), row("G02"), row("G03")])
        write(env, "%s-general_20261008-%s" % (label, stamp), [row("S01")])
    write(env, "r-synthetic_20261008-100000", [row("SY1")])
    write(env, "c-synthetic_20261008-110000", [row("SY1", ok=False)])  # not declared, so not scored
    assert gate.main(["c", "--against", "r"]) == 0
    assert gate.main(["c", "--against", "r", "--sets", "gold,synthetic"]) == 2
    assert gate.main(["c", "--against", "r", "--sets", "core,gold,gold"]) == 0  # core expands, no duplicates


def test_an_explicit_sets_list_needs_no_other_set(mini, env):
    write(env, "r-gold_20261008-100000", [row("G01"), row("G02"), row("G03")])
    write(env, "c-gold_20261008-110000", [row("G01"), row("G02"), row("G03")])
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 0


def test_a_declared_set_without_a_candidate_or_a_reference_run_is_a_usage_error(mini, env):
    from evals import cli

    write(env, "r-gold_20261008-100000", [row("G01"), row("G02"), row("G03")])
    write(env, "c-gold_20261008-110000", [row("G01"), row("G02"), row("G03")])
    write(env, "r-general_20261008-100000", [row("S01")])
    assert cli.main(["gate", "c", "--against", "r"]) == 64  # core has general: no candidate run
    write(env, "c-general_20261008-110000", [row("S01")])
    write(env, "c-synthetic_20261008-110000", [row("SY1")])
    assert cli.main(["gate", "c", "--against", "r"]) == 0
    assert cli.main(["gate", "c", "--against", "r", "--sets", "gold,synthetic"]) == 64  # no reference run
    assert cli.main(["gate", "c", "--against", "r", "--sets", "gold,nope"]) == 64  # not in gate.yaml
    cand, ref = str(env / "c-gold_20261008-110000"), str(env / "r-gold_20261008-100000")
    assert cli.main(["gate", "--cand", cand, "--ref", ref, "--sets", "gold"]) == 64  # folders declare themselves


def test_a_panel_folder_resolves_to_the_panel_and_a_file_folder_to_the_set(mini, env):
    profile = str(mini)
    cfg = gate.load_config(profile)
    index = gate.conversations(profile, cfg)
    members = {name: set(gate.set_ids(profile, cfg, name)) for name in cfg.sets}
    assert "SY1" not in index or index["SY1"][0] == "synthetic"  # a panel never enters the index
    assert members["synthetic_sample"] == {"SY1", "SY2"} and members["synthetic"] == {"SY1", "SY2", "SY3", "SY4"}
    panel = write(env, "c-rerun1-synthetic_sample_20261008-110000", [row("SY1"), row("SY2")])
    whole = write(env, "c-synthetic_20261008-110000", [row("SY1"), row("SY2"), row("SY3")])
    other = write(env, "c-synthetic_sample_20261008-120000", [row("SY3")])  # not covered by the panel: the ids decide
    assert gate.set_of(gate.read_rows(panel), index, panel, members) == "synthetic_sample"
    assert gate.set_of(gate.read_rows(whole), index, whole, members) == "synthetic"
    assert gate.set_of(gate.read_rows(other), index, other, members) == "synthetic"
    assert gate.set_of(gate.read_rows(panel), index, panel) == "synthetic"  # without members


def test_a_panel_is_gated_by_label_and_its_re_run_folders_resolve(mini, env, capsys):
    write(env, "r-gold_20261008-100000", [row("G01")])  # retrieval and latency need a first-turn set
    write(env, "c-gold_20261008-110000", [row("G01")])
    write(env, "r-synthetic_sample_20261008-100000", [row("SY1"), row("SY2")])
    write(env, "c-synthetic_sample_20261008-110000", [row("SY1"), row("SY2", ok=False)])
    write(env, "c-rerun1-synthetic_sample_20261008-120000", [row("SY2", ok=False)])
    write(env, "c-rerun2-synthetic_sample_20261008-121000", [row("SY2", ok=False)])
    assert gate.main(["c", "--against", "r", "--sets", "gold,synthetic_sample"]) == 1
    assert "synthetic_sample 2 -> 1 regressions SY2 t1" in capsys.readouterr().out


def test_a_replayable_turn_prints_a_replay_command_without_a_space_in_its_turns(mini, env, capsys):
    write(env, "r-gold_20261008-100000", [row("G01"), row("G02"), row("G02", 2), row("G03")])
    write(env, "c-gold_20261008-110000", [row("G01"), row_ids("G02", 1), row_ids("G02", 2, ok=False), row("G03")])
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 2
    out = capsys.readouterr().out
    commands = [c.strip() for c in out.splitlines() if c.startswith("  evals run")]
    assert commands == [
        "evals run gold --replay c-gold_20261008-110000 --turns G02:t2 --label c-rerun1",
        "evals run gold --replay c-gold_20261008-110000 --turns G02:t2 --label c-rerun2",
    ]
    values = [c.split(" --turns ")[1].split(" --label ")[0] for c in commands]
    assert values == ["G02:t2", "G02:t2"] and all(" " not in v and '"' not in v for v in values)


def test_history_and_old_runs_are_re_run_whole_with_ids(mini, env, capsys):
    # H1 has a replace turn; G03's rows predate turn_id (no user, no turn_id): both fall back to --ids
    write(env, "r-gold_20261008-100000", [row("G01"), row("G02"), row("G02", 2), row("G03")])
    write(env, "r-history_20261008-100000", [row("H1", 1), row("H1", 2)])
    write(env, "c-history_20261008-110000", [row_ids("H1", 1), row_ids("H1", 2, ok=False)])
    write(env, "c-gold_20261008-110000", [row("G01"), row("G02"), row("G02", 2), row("G03", ok=False)])
    assert gate.main(["c", "--against", "r", "--sets", "gold,history"]) == 2
    out = capsys.readouterr().out
    assert (
        "  evals run gold --ids G03 --label c-rerun1" in out and "  evals run history --ids H1 --label c-rerun1" in out
    )
    assert "--replay" not in out


def test_the_re_run_number_counts_a_turns_own_runs_so_a_mixed_pair_does_not_shift_it(mini, env, capsys):
    write(env, "r-gold_20261008-100000", [row("G01"), row("G02"), row("G02", 2), row("G03")])
    write(
        env,
        "c-gold_20261008-110000",
        [row("G01"), row_ids("G02", 1), row_ids("G02", 2, ok=False), row("G03", ok=False)],
    )
    write(env, "c-rerun1-gold_20261008-120000", [row_ids("G02", 2, ok=False)])  # the replay of G02 t2
    write(env, "c-rerun1-gold_20261008-120100", [row("G03", ok=False)])  # the --ids re-run of G03, the same k
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 2
    commands = [c.strip() for c in capsys.readouterr().out.splitlines() if c.startswith("  evals run")]
    assert commands == [
        "evals run gold --replay c-gold_20261008-110000 --turns G02:t2 --label c-rerun2",
        "evals run gold --ids G03 --label c-rerun2",
    ]


def test_every_replay_of_a_conversation_lists_all_its_pending_turns_from_its_least_run_one(mini, env, capsys):
    write(env, "r-gold_20261008-100000", [row("G01"), row("G02"), row("G02", 2), row("G03")])
    write(
        env,
        "c-gold_20261008-110000",
        [row("G01"), row_ids("G02", 1, ok=False), row_ids("G02", 2, ok=False), row("G03")],
    )
    write(
        env, "c-rerun1-gold_20261008-120000", [row_ids("G02", 2, ok=False)]
    )  # a replay that stopped part-way: t2 only
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 2
    commands = [c.strip() for c in capsys.readouterr().out.splitlines() if c.startswith("  evals run")]
    both = "evals run gold --replay c-gold_20261008-110000 --turns G02:t1,G02:t2 --label c-rerun%d"
    assert commands == [both % 1, both % 2]


def test_a_replay_starts_at_the_earliest_flipped_turn_even_when_that_turn_is_settled(mini, env, capsys):
    write(env, "r-gold_20261008-100000", [row("G01"), row("G02"), row("G02", 2), row("G03")])
    write(
        env,
        "c-gold_20261008-110000",
        [row("G01"), row_ids("G02", 1, ok=False), row_ids("G02", 2, ok=False), row("G03")],
    )
    write(env, "c-rerun1-gold_20261008-120000", [row_ids("G02", 1), row_ids("G02", 2, ok=False)])
    write(
        env, "c-rerun2-gold_20261008-121000", [row_ids("G02", 1)]
    )  # stopped after t1: t1 settled (3 runs), t2 pending
    assert gate.main(["c", "--against", "r", "--sets", "gold"]) == 2
    commands = [c.strip() for c in capsys.readouterr().out.splitlines() if c.startswith("  evals run")]
    assert commands == ["evals run gold --replay c-gold_20261008-110000 --turns G02:t1,G02:t2 --label c-rerun2"]


def test_retrieval_and_latency_need_a_declared_first_turn_set(mini, env):
    from evals import cli

    for name in ("history", "gold"):
        write(
            env,
            "r-%s_20261008-100000" % name,
            [row("H1")] if name == "history" else [row("G01"), row("G02"), row("G03")],
        )
        write(
            env,
            "c-%s_20261008-110000" % name,
            [row("H1")] if name == "history" else [row("G01"), row("G02"), row("G03")],
        )
    assert cli.main(["gate", "c", "--against", "r", "--sets", "history"]) == 64
    assert cli.main(["gate", "c", "--against", "r", "--sets", "gold"]) == 0
    assert cli.main(["gate", "c", "--against", "r", "--sets", "gold,history"]) == 0
