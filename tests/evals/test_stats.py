"""evals stats: noise floor and TTFT spread of back-to-back runs."""

import json

import pytest
from evals import stats as bs

SECTIONS = ["https://x/%d|0|h%d" % (i, i) for i in range(14)]


def row(cid, turn=1, sections=(), passed=True, ttft=7.0, errors=()):
    return {
        "id": cid,
        "turn": turn,
        "first_token_s": ttft,
        "sources": [{"url": "u", "title": "", "internal": False, "section": s} for s in sections],
        "checks": {"auto_pass": passed, "errors": list(errors), "failed": [] if passed else ["contains_any"]},
    }


def write_run(tmp_path, name, rows):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(folder)


def test_identical_runs_agree_fully_with_no_spread():
    rows = [row("G01", sections=SECTIONS), row("G02", sections=SECTIONS[:5], ttft=8.0)]
    stats = bs.summarise({"gold": bs.compare_pair(rows, rows)})
    assert stats["noise_floor"] == 1.0 and stats["noise_floor_meets_97"] is True
    assert stats["ttft_spread"] == 0.0 and stats["ttft_margin"] == 0.3
    assert stats["sets"]["gold"]["flips"] == [] and stats["sets"]["gold"]["identical"] == 2


def test_one_changed_section_scores_thirteen_of_fourteen():
    pair = bs.compare_pair([row("G01", sections=SECTIONS)], [row("G01", sections=SECTIONS[:13] + ["other|0|h"])])
    assert pair["scores"] == [13 / 14]
    assert pair["differing"] == [{"id": "G01", "overlap": round(13 / 14, 4)}]


def test_order_does_not_matter():
    pair = bs.compare_pair([row("G01", sections=SECTIONS)], [row("G01", sections=list(reversed(SECTIONS)))])
    assert pair["scores"] == [1.0]


def test_questions_without_retrieval_in_both_runs_are_counted_not_scored():
    pair = bs.compare_pair([row("S01"), row("S02", sections=SECTIONS)], [row("S01"), row("S02", sections=SECTIONS)])
    assert pair["no_retrieval"] == ["S01"] and pair["scores"] == [1.0]


def test_retrieval_in_one_run_only_scores_zero():
    assert bs.compare_pair([row("S01", sections=SECTIONS)], [row("S01")])["scores"] == [0.0]


def test_only_first_turns_are_scored_but_every_turn_counts_for_flips():
    a = [row("S01", sections=SECTIONS), row("S01", turn=2, sections=SECTIONS[:3], passed=True)]
    b = [row("S01", sections=SECTIONS), row("S01", turn=2, sections=["x|0|y"], passed=False)]
    pair = bs.compare_pair(a, b)
    assert pair["scores"] == [1.0] and pair["flips"] == ["S01 t2"]


def test_error_events_are_counted_per_run():
    pair = bs.compare_pair([row("G01", errors=["Fehler"])], [row("G01")])
    assert pair["errors"] == [1, 0]


def test_ttft_medians_pool_the_first_turns_of_every_set():
    gold = bs.compare_pair([row("G01", ttft=7.0), row("G02", ttft=9.0)], [row("G01", ttft=7.5), row("G02", ttft=9.5)])
    general = bs.compare_pair(
        [row("S01", ttft=8.0), row("S01", turn=2, ttft=1.0)], [row("S01", ttft=8.4), row("S01", turn=2, ttft=1.0)]
    )
    stats = bs.summarise({"gold": gold, "general": general})
    assert stats["ttft_median"] == [8.0, 8.4]
    assert stats["ttft_spread"] == pytest.approx(0.4) and stats["ttft_margin"] == pytest.approx(0.4)


def test_a_noise_floor_below_97_percent_is_flagged():
    a = [row("G%02d" % i, sections=SECTIONS) for i in range(10)]
    b = [
        row("G%02d" % i, sections=SECTIONS if i else SECTIONS[:7] + ["y%d|0|z" % j for j in range(7)])
        for i in range(10)
    ]
    stats = bs.summarise({"gold": bs.compare_pair(a, b)})
    assert stats["noise_floor"] == pytest.approx(0.95) and stats["noise_floor_meets_97"] is False


def test_runs_recorded_without_section_identities_are_refused(tmp_path):
    old = write_run(
        tmp_path,
        "old",
        [
            {
                "id": "G01",
                "turn": 1,
                "first_token_s": 7.0,
                "sources": [{"url": "u", "title": "", "internal": False}],
                "checks": {"auto_pass": True, "errors": [], "failed": []},
            }
        ],
    )
    with pytest.raises(SystemExit, match="no section identity"):
        bs.load_rows(old)


def test_the_command_line_writes_the_statistics(tmp_path, capsys):
    rows = [row("G01", sections=SECTIONS)]
    a, b = write_run(tmp_path, "a", rows), write_run(tmp_path, "b", rows)
    out = tmp_path / "stats.json"
    assert bs.main(["--pair", "gold", a, b, "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["noise_floor"] == 1.0
    assert "noise floor 1.0000 (meets 97 %: True)" in capsys.readouterr().out


def test_questions_in_only_one_run_are_reported_as_missing(tmp_path, capsys):
    a = [row("G01", sections=SECTIONS), row("G02", sections=SECTIONS)]
    b = [row("G01", sections=SECTIONS), row("G03", sections=SECTIONS)]
    pair = bs.compare_pair(a, b)
    assert pair["missing"] == ["G02 (run 1 only)", "G03 (run 2 only)"] and pair["scores"] == [1.0]
    assert bs.summarise({"gold": pair})["sets"]["gold"]["missing"] == pair["missing"]
    assert bs.main(["--pair", "gold", write_run(tmp_path, "a", a), write_run(tmp_path, "b", b)]) == 0
    assert "missing: G02 (run 1 only), G03 (run 2 only)" in capsys.readouterr().out
