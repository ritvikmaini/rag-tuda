"""Production traps: the indexer gets no GPU and the backend defaults to one worker everywhere."""

import os
import re
import shutil
import subprocess

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def text(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def load(name):
    return yaml.safe_load(text(name))


def test_the_indexer_has_no_gpu():
    indexer = load("deploy/compose.yaml")["services"]["indexer"]
    assert indexer.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices") is None
    assert [e for e in indexer.get("environment", []) if e.startswith("NVIDIA")] == []


def test_the_backend_defaults_to_one_worker_everywhere():
    assert "UVICORN_WORKERS=${UVICORN_WORKERS:-1}" in load("deploy/compose.yaml")["services"]["backend"]["environment"]
    assert re.search(r"^UVICORN_WORKERS=1$", text(".env_example"), re.M)
    assert 'workers = int(os.environ.get("UVICORN_WORKERS", "1"))' in text("apps/assistant_api/gunicorn_conf.py")


needs_docker = pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0,
    reason="needs the docker CLI with the compose plugin (no daemon)",
)


@needs_docker
def test_the_rendered_gpu_config_gives_indexer_no_gpu_and_the_base_config():
    result = subprocess.run(
        ["docker", "compose", "-f", "deploy/compose.yaml", "--profile", "build", "config"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    services = yaml.safe_load(result.stdout)["services"]
    indexer = services["indexer"]
    assert indexer.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices") is None
    env_keys = (
        indexer["environment"].keys()
        if isinstance(indexer["environment"], dict)
        else [e.split("=")[0] for e in indexer["environment"]]
    )
    assert [k for k in env_keys if k.startswith("NVIDIA")] == []
    assert not [v for v in indexer["volumes"] if v.get("target", "").startswith("/app/config")]
    assert not [v for v in services["backend"]["volumes"] if v.get("target", "").startswith("/app/config")]
