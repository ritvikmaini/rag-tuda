"""Two worktrees' `just ci` at once: a linked worktree gets its own id (its name in ASCII
plus 6 hex of its path's hash), which names its compose project (rag-ci-<id>) and image tags (:ci-<id>), and Docker picks its
host ports; the main checkout and GitHub CI (no id) keep project rag-ci, the plain tags and ports 25432, 28000 and 28002."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# GIT_* too: a git hook (the pre-push hook runs `just unit`) exports GIT_DIR and friends, and the fixture's git commands would
# then commit into, add a worktree to and set core.bare on the real repository
CLEAN = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_CI_", "COMPOSE_", "GIT_"))}
HOST_PORTS = {"pgvector": "25432", "backend": "28002", "vllm": "28000"}
IMAGES = {
    "backend": "rag-ci-backend",
    "indexer": "rag-ci-indexer",
    "vllm": "rag-ci-fakes",
    "evals": "rag-tuda-evals",
    "migrate": "rag-ci-backend",
}  # the one-shot runs the backend's worktree tag
NAMES = ("Unit_4C.move", "a.b", "a_b", "a-b", "Ünï code")


def run(*args, cwd=ROOT, **env):
    return subprocess.run(args, cwd=cwd, env={**CLEAN, **env}, capture_output=True, text=True, check=True).stdout


def evaluate(name, cwd, **env):
    return run("just", "--evaluate", name, cwd=cwd, **env)


@pytest.fixture(scope="module")
def checkouts(tmp_path_factory):
    """A main checkout and linked worktrees named like real ones and not (dots, underscores, non-ASCII), all under a path with
    a space, each with this justfile."""
    base = tmp_path_factory.mktemp("with space")
    main = base / "main checkout"
    git = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
    ]
    run(*git, "init", "-q", str(main), cwd=base)
    run(*git, "commit", "-q", "--allow-empty", "-m", "x", cwd=main)
    linked = {}
    for i, name in enumerate(NAMES):
        linked[name] = base / name
        run(*git, "worktree", "add", "-q", "-b", "b%d" % i, str(linked[name]), cwd=main)
    for folder in (main, *linked.values()):
        shutil.copy(ROOT / "justfile", folder / "justfile")
    return main, linked


@pytest.mark.skipif(shutil.which("just") is None, reason="needs just")
def test_the_main_checkout_keeps_the_plain_names_and_each_linked_worktree_gets_its_own(checkouts):
    main, linked = checkouts
    assert [evaluate(v, main) for v in ("ci_id", "RAG_CI_TAG", "ci_project")] == ["", "", "rag-ci"]
    ids = {name: evaluate("ci_id", folder) for name, folder in linked.items()}
    top = os.path.realpath(linked["Unit_4C.move"]).encode()
    assert ids["Unit_4C.move"] == "unit-4c-move-" + hashlib.sha1(b"blob %d\0" % len(top) + top).hexdigest()[:6]
    assert len(set(ids.values())) == len(NAMES)  # a.b, a_b and a-b differ
    assert all(re.fullmatch(r"[a-z0-9-]+-[0-9a-f]{6}", i) for i in ids.values()), ids
    odd = linked["Ünï code"]
    assert {evaluate("ci_id", odd, LC_ALL=locale) for locale in ("C", "en_US.UTF-8")} == {ids["Ünï code"]}  # any locale
    folder, i = linked["Unit_4C.move"], ids["Unit_4C.move"]
    assert (evaluate("RAG_CI_TAG", folder), evaluate("ci_project", folder)) == (":ci-" + i, "rag-ci-" + i)
    assert evaluate("RAG_CI_TAG", main, RAG_CI_ID="x1") == ":ci-x1"  # an explicit id wins
    assert evaluate("RAG_CI_TAG", folder, RAG_CI_ID="") == ""  # an empty one gives the plain names


def compose_config(**env):
    return json.loads(
        run(
            "docker",
            "compose",
            "-f",
            "deploy/compose.ci.yaml",
            "--profile",
            "evals",
            "--profile",
            "migrate",
            "--profile",
            "build",
            "config",
            "--format",
            "json",
            **env,
        )
    )


def published(config, service):
    return [(p["host_ip"], p.get("published", "")) for p in config["services"][service].get("ports", [])]


@pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0,
    reason="needs the docker compose CLI (no daemon)",
)
def test_the_compose_file_keeps_the_plain_names_by_default_and_isolates_a_worktree():
    plain = compose_config()
    assert plain["name"] == "rag-ci"
    assert {s: plain["services"][s]["image"] for s in IMAGES} == {**IMAGES, "evals": "rag-tuda-evals:dev"}
    assert {s: published(plain, s) for s in HOST_PORTS} == {s: [("127.0.0.1", p)] for s, p in HOST_PORTS.items()}
    ports = {"RAG_CI_%s_PORT" % k: "" for k in ("PG", "LLM", "BACKEND")}
    mine = compose_config(COMPOSE_PROJECT_NAME="rag-ci-wt", RAG_CI_TAG=":ci-wt", **ports)
    assert mine["name"] == "rag-ci-wt"
    assert {s: mine["services"][s]["image"] for s in IMAGES} == {s: i + ":ci-wt" for s, i in IMAGES.items()}
    assert {s: published(mine, s) for s in HOST_PORTS} == {s: [("127.0.0.1", "")] for s in HOST_PORTS}  # Docker picks


def test_the_harness_client_reads_the_ports_just_harness_exports():
    show = "from rag_testing import harness as h; print(h.BACKEND, h.LLM, h.DB_PORT)"
    assert run(sys.executable, "-c", show).split() == [
        "http://127.0.0.1:28002",
        "http://127.0.0.1:28000",
        "25432",
    ]
    ports = {
        "RAG_CI_PG_PORT": "5",
        "RAG_CI_LLM_PORT": "1",
        "RAG_CI_BACKEND_PORT": "2",
    }
    assert run(sys.executable, "-c", show, **ports).split() == [
        "http://127.0.0.1:2",
        "http://127.0.0.1:1",
        "5",
    ]
