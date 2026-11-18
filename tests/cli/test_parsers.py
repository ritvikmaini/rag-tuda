from pathlib import Path

import pytest

from rag_tuda import doctor as d

FIX = Path(__file__).parent / "fixtures"


def read(scn, f):
    return (FIX / scn / f).read_text()


def test_sysctl():
    assert d.parse_sysctl(read("m1max64", "sysctl.txt")) == (64.0, "Apple M1 Max", 10)
    assert d.parse_sysctl(read("intelmac", "sysctl.txt"))[1].startswith("Intel")
    assert d.parse_sysctl("garbage") is None
    assert d.parse_sysctl("") is None


def test_nvidia():
    assert d.parse_nvidia(read("rtx4090", "nvidia.txt")) == [("NVIDIA GeForce RTX 4090", 24.564)]
    assert len(d.parse_nvidia(read("h100x4", "nvidia.txt"))) == 4
    assert d.parse_nvidia("") == []
    assert d.parse_nvidia("No devices were found\nbad, line, x\nGPU, [N/A]\n") == []


def test_docker_variants():
    assert d.parse_docker(read("m1max64", "docker.json")) == d.Docker(True, True, "27.3.1", "Colima")
    assert d.parse_docker(read("m2_16", "docker.json")).runtime_hint == "OrbStack"
    assert d.parse_docker(read("m3pro36", "docker.json")).runtime_hint == "Docker Desktop"
    assert d.parse_docker(read("rtx4090", "docker.json")).runtime_hint == "Docker Engine"


def test_docker_down_absent_garbage():
    down = d.parse_docker(read("docker_down", "docker.json"))
    assert down.installed and not down.ok and down.server_version == ""
    assert d.parse_docker(None) == d.Docker(False, False, "", "")
    assert d.parse_docker("not json").installed is False


def test_ollama():
    assert d.parse_ollama_cli(read("m1max64", "ollama_cli.txt")) == "0.12.9"
    assert (
        d.parse_ollama_cli(
            "Warning: could not connect to a running Ollama instance\nWarning: client version is 0.12.9\n"
        )
        == "0.12.9"
    )
    assert d.parse_ollama_cli("") is None
    assert d.parse_ollama_api('{"version":"0.12.9"}') == "0.12.9"
    assert d.parse_ollama_api("<html>") is None
    assert d.parse_ollama_api(None) is None


def test_meminfo_and_wsl_and_cpuinfo():
    assert d.parse_meminfo(read("cpu32", "meminfo.txt")) == pytest.approx(32768000 / 1024 / 1024)
    assert d.parse_meminfo("nothing") is None
    assert d.is_wsl_text(read("wsl2", "proc_version.txt"))
    assert not d.is_wsl_text(read("cpu32", "proc_version.txt"))
    assert d.is_wsl_text(None) is False
    assert d.parse_cpuinfo(read("rtx4090", "cpuinfo.txt")) == "AMD Ryzen 9 7950X 16-Core Processor"
    assert d.parse_cpuinfo("") == ""


def test_assemble_scenarios(scenario):
    m = scenario("m1max64")
    assert (m.os, m.arch, m.apple_chip, m.unified_gb, m.ram_gb) == ("darwin", "arm64", "Apple M1 Max", 64.0, 64.0)
    assert m.cores == 10 and m.docker.ok and m.ollama.installed and m.ollama.reachable
    w = scenario("wsl2")
    assert w.is_wsl and w.nvidia == [("NVIDIA GeForce RTX 4070", 12.282)]
    i = scenario("intelmac")
    assert i.apple_chip == "" and i.unified_gb == 0
    c = scenario("cpu32")
    assert c.nvidia == [] and not c.ollama.installed
    assert c.cpu_name == "Intel(R) Core(TM) i5-12400" and c.ram_gb == pytest.approx(31.25, abs=0.01)


def test_missing_tools_are_not_crashes(scenario):
    f = scenario("m1max64", docker=None, ollama_cli=None, ollama_api=None, nvidia=None)
    assert not f.docker.installed and not f.ollama.installed and f.nvidia == []


def test_windows_memory(scenario):
    f = scenario("cpu32", system="windows", machine="AMD64", meminfo=None, win_mem_bytes=34359738368)
    assert f.ram_gb == 32.0 and not f.is_wsl
