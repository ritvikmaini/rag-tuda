from pathlib import Path

import pytest

from rag_tuda import doctor

FIX = Path(__file__).parent / "fixtures"
FILES = {
    "sysctl": "sysctl.txt",
    "nvidia": "nvidia.txt",
    "docker": "docker.json",
    "ollama_cli": "ollama_cli.txt",
    "ollama_api": "ollama_api.json",
    "meminfo": "meminfo.txt",
    "proc_version": "proc_version.txt",
    "cpuinfo": "cpuinfo.txt",
}


def load(name: str, **overrides) -> doctor.Facts:
    d = FIX / name
    system, machine = (d / "platform.txt").read_text().split()
    texts = {k: (d / f).read_text() for k, f in FILES.items() if (d / f).exists()}
    args = dict(system=system, machine=machine, cores=8, disk_free_gb=500.0, brew=True, **texts)
    args.update(overrides)
    return doctor.assemble(**args)


@pytest.fixture
def scenario():
    return load
