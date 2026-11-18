"""The deploy smoke against the harness: `evals smoke` in the real evals image passes
on a correct stack, so a deploy can rely on the smoke."""

import os
import subprocess
from pathlib import Path

from rag_testing import harness

COMPOSE = Path(__file__).resolve().parents[2] / "deploy" / "compose.ci.yaml"


def test_the_deploy_smoke_passes_against_the_harness(ready: None, llm: harness.FakeLLM) -> None:
    llm.script(harness.OUT_OF_SCOPE)  # E03 "Erzähl mir einen Witz." is the smoke's direct reply
    smoke = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE),
            "--profile",
            "evals",
            "run",
            "--rm",
            "--no-deps",
            "--name",
            os.environ.get("COMPOSE_PROJECT_NAME", "rag-ci") + "-evals",  # its own per worktree (just harness)
            "evals",
            "smoke",
            "--label",
            "harness",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
    assert "RESULT PASS: 0 of 6 turns failed a check" in smoke.stdout
