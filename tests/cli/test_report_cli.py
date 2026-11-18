import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rag_tuda import cli, doctor as d, tiers

BARE = re.compile(r"qwen3:(4b|30b)(?![-\w])")
CASES = {
    "m1max64": "T3",
    "m2_16": "T1",
    "m3pro36": "T2",
    "rtx4090": "T2",
    "a100x2": "T3",
    "h100x4": "T3+",
    "cpu32": "T1",
    "wsl2": "T2",
    "intelmac": "T1",
}


@pytest.mark.parametrize("name,tier", CASES.items())
def test_scenario_tiers(scenario, name, tier):
    f = scenario(name)
    assert tiers.pick_tier(f).name == tier
    text = d.report(f, tiers.pick_tier(f))
    assert f"Tier: {tier}" in text
    assert not BARE.search(text)


def test_report_content_t1_mac(scenario):
    f = scenario("m2_16")
    t = tiers.pick_tier(f)
    text = d.report(f, t)
    assert "qwen3:4b-instruct-2507-q4_K_M" in text
    assert "snowflake-arctic-embed-m-v2.0" in text and "bge-reranker-v2-m3" in text
    assert "Download total: 6.5 GB" in text
    assert "Mode: Mac native" in text
    assert "10.4 GB" in text  # 16 * 0.65
    assert "personal document Q&A" in text
    assert "Missing: Ollama is not installed: run `brew install ollama`." in text
    assert "Next: fix the missing items" in text


def test_report_t2_and_t3_tags(scenario):
    assert "qwen3:14b" in d.report(scenario("m3pro36"), tiers.pick_tier(scenario("m3pro36")))
    assert "qwen3:30b-a3b-instruct-2507-q4_K_M" in d.report(scenario("m1max64"), tiers.pick_tier(scenario("m1max64")))


def test_compose_modes_and_wsl_hint(scenario):
    assert d.runtime_mode(scenario("intelmac")).startswith("compose")
    assert d.runtime_mode(scenario("rtx4090")).startswith("compose")
    assert d.runtime_mode(scenario("m1max64")).startswith("Mac native")
    win = scenario("cpu32", system="windows", machine="AMD64", meminfo=None, win_mem_bytes=34359738368)
    assert "WSL2" in d.runtime_mode(win)
    assert "WSL2" not in d.runtime_mode(scenario("wsl2"))


def test_low_ram_warns_but_stays_t1(scenario):
    f = scenario("cpu32", meminfo="MemTotal:  8000000 kB\n")
    text = d.report(f, tiers.pick_tier(f))
    assert tiers.pick_tier(f).name == "T1"
    assert "Warning" in text and "16 GB" in text


def test_t3_plus_marked_untested(scenario):
    f = scenario("h100x4")
    assert "untested" in d.report(f, tiers.pick_tier(f))


def test_exit_codes(scenario):
    assert d.problems(scenario("rtx4090")) == []
    assert d.problems(scenario("m1max64")) == []
    p = d.problems(scenario("rtx4090", docker=None))
    assert len(p) == 1 and "Docker" in p[0] and "Docker Engine" in p[0]
    p = d.problems(scenario("m1max64", docker=None))
    assert "Colima" in p[0] and "OrbStack" in p[0] and "Docker Desktop" in p[0]
    down = d.problems(scenario("docker_down", ollama_cli=None))
    assert "not running" in down[0]
    # Apple silicon: native mode needs Ollama on the host; the hint depends on Homebrew
    p = d.problems(scenario("m2_16"))
    assert len(p) == 1 and "brew install ollama" in p[0] and "brew.sh" not in p[0]
    p = d.problems(scenario("m2_16", brew=False))
    assert "Ollama" in p[0] and "brew" in p[0]
    # Linux/compose runs Ollama itself, so it is not a prerequisite
    assert d.problems(scenario("cpu32")) == []


def test_report_names_missing_prerequisite(scenario):
    f = scenario("rtx4090", docker=None)
    text = d.report(f, tiers.pick_tier(f))
    assert "Missing" in text and "Docker Engine" in text


def test_json_round_trip(scenario):
    f = scenario("m1max64")
    data = json.loads(d.report_json(f, tiers.pick_tier(f)))
    assert data["tier"]["name"] == "T3"
    assert data["tier"]["ollama_tag"] == "qwen3:30b-a3b-instruct-2507-q4_K_M"
    assert data["usable_gb"] == pytest.approx(41.6)
    assert data["download_total_gb"] == pytest.approx(23.0)
    assert data["mode"].startswith("Mac native")
    assert data["ok"] is True and data["missing"] == []
    assert data["facts"]["apple_chip"] == "Apple M1 Max"
    assert data["facts"]["docker"]["runtime_hint"] == "Colima"
    assert [m["name"] for m in data["fixed_models"]][1] == "BAAI/bge-reranker-v2-m3"
    bad = json.loads(d.report_json(scenario("rtx4090", docker=None), tiers.TIERS[1]))
    assert bad["ok"] is False and bad["missing"]


def test_cli_exit_code_and_json(monkeypatch, capsys, scenario):
    monkeypatch.setattr(d, "gather", lambda: scenario("rtx4090"))
    assert cli.main(["doctor"]) == 0
    assert "Tier: T2" in capsys.readouterr().out
    assert cli.main(["doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tier"]["name"] == "T2"
    monkeypatch.setattr(d, "gather", lambda: scenario("rtx4090", docker=None))
    assert cli.main(["doctor"]) == 1


def test_cli_rejects_other_subcommands():
    with pytest.raises(SystemExit):
        cli.main(["up"])


def test_gather_on_this_machine_writes_nothing():
    root = Path(__file__).resolve().parents[2]
    skip = {".venv", ".pytest_cache", "__pycache__"}

    def snap():
        return {
            str(p): p.stat().st_mtime_ns
            for p in root.rglob("*")
            if p.is_file() and not skip & set(p.relative_to(root).parts)
        }

    before = snap()
    f = d.gather()
    text = d.report(f, tiers.pick_tier(f))
    d.report_json(f, tiers.pick_tier(f))
    assert text and f.os
    assert snap() == before


def test_console_script_runs():
    script = Path(sys.executable).parent / "rag"  # installed from the root project's [project.scripts]
    assert script.is_file()
    out = subprocess.run([script, "doctor", "--json"], capture_output=True, text=True)
    assert out.returncode in (0, 1)
    json.loads(out.stdout)
