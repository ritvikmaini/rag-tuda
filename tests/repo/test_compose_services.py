"""Compose services: the unused Streamlit demo is gone; the evals job."""

import os
import re
import shutil
import subprocess

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def services(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return yaml.safe_load(fh)["services"]


def test_no_compose_service_builds_or_runs_the_streamlit_demo():
    for name in ("deploy/compose.yaml",):
        found = services(name)
        assert "frontend" not in found, name
        assert [s for s, spec in found.items() if "frontend" in str((spec or {}).get("build", ""))] == [], name


def test_the_evals_job_is_a_profile_service_that_starts_nothing_else():
    evals = services("deploy/compose.yaml")["evals"]
    assert evals["profiles"] == ["evals"]
    assert "depends_on" not in evals and "ports" not in evals and "restart" not in evals
    assert evals["build"] == {
        "context": "..",
        "dockerfile": "apps/evals/Dockerfile",
        "args": {"PROFILE": "example"},
    }
    assert "networks" not in evals  # the project's default network, with the backend
    assert evals["volumes"] == ["../data/eval_runs:/app/data/eval_runs:ro"]
    assert evals["secrets"] == ["backend_token"]
    assert "OPENAI_API_KEY=not-needed" in evals["environment"]


def test_the_backend_token_reaches_the_evals_job_as_a_file_not_a_variable():
    with open(os.path.join(ROOT, "deploy/compose.yaml"), encoding="utf-8") as fh:
        compose = yaml.safe_load(fh)
    assert compose["secrets"]["backend_token"] == {"environment": "BACKEND_REFRESH_TOKEN"}
    assert [e for e in compose["services"]["evals"].get("environment", []) if "TOKEN" in e] == []


@pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0,
    reason="needs the docker CLI with the compose plugin (no daemon)",
)
def test_compose_renders_the_evals_job_with_every_secret_set(tmp_path):
    names = set()
    for name in ("deploy/compose.yaml",):
        with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
            names |= set(re.findall(r"\$\{([A-Z0-9_]+):\?", fh.read()))
    env = tmp_path / "full.env"
    env.write_text("".join("%s=x\n" % k for k in sorted(names)), encoding="utf-8")
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(env),
            "-f",
            "deploy/compose.yaml",
            "--profile",
            "evals",
            "--profile",
            "migrate",
            "config",
            "--quiet",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_the_job_reads_runs_through_a_read_only_mount_and_writes_inside_its_container():
    # the job's non-root user cannot write a bind mount owned by the host user; every command writes into the
    # container's own out_dir, and the result is copied out with docker cp
    from evals.settings import EvalsSection

    (volume,) = services("deploy/compose.yaml")["evals"]["volumes"]
    source, target, mode = volume.split(":")
    section = EvalsSection()
    assert (source, mode) == ("../data/eval_runs", "ro") and section.runs_dir == target
    assert not (section.out_dir.rstrip("/") + "/").startswith(target.rstrip("/") + "/")


def test_the_documented_index_build_command_runs_the_indexer_cli():
    """`run --rm indexer build` (README, compose.yaml) only works when the image's entrypoint is the `indexer` command; the
    harness runs the same words in the same service."""
    dockerfile = open(os.path.join(ROOT, "apps/indexer/Dockerfile"), encoding="utf-8").read()
    assert 'ENTRYPOINT ["indexer"]\nCMD ["--help"]' in dockerfile
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert "--profile build run --rm indexer build`" in readme
    assert services("deploy/compose.yaml")["indexer"]["profiles"] == ["build"]
    harness = open(os.path.join(ROOT, "packages/rag_testing/src/rag_testing/harness.py"), encoding="utf-8").read()
    assert '"--no-deps", "indexer"]' in harness and '[*command, "build"]' in harness
