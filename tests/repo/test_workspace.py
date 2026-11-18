"""The uv workspace: a virtual root of its members, constraints that pin every package the images install, an image that
installs one app and no group, a root build context that lets in only the workspace and the profiles, and hooks that leave
CRLF scripts, fixtures and role files alone."""

import os
import re
import subprocess
import tomllib

import yaml
from packaging.markers import Marker

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LINUX = {
    "sys_platform": "linux",
    "platform_machine": "x86_64",
    "platform_system": "Linux",
    "python_version": "3.12",
    "python_full_version": "3.12.15",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "os_name": "posix",
    "extra": "",
}
# the minor is the policy (a newer patch is taken at each unit's start); the index digest pins the rest
BASE = re.compile(r"^ARG PYTHON_IMAGE=python:3\.12\.\d+-slim-bookworm@sha256:[0-9a-f]{64}$", re.M)


def text(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_on_linux(*packages):
    """name -> version of every package the lock installs on linux x86_64 for the workspace packages (what the images get)."""
    out = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-default-groups",
            "--no-emit-workspace",
            "--no-hashes",
            "--no-header",
            "--no-annotate",
            *(arg for p in packages for arg in ("--package", p)),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = {}
    for line in out.splitlines():
        requirement, _, marker = line.partition(";")
        if (
            requirement.strip()
            and not requirement.startswith("#")
            and (not marker.strip() or Marker(marker.strip()).evaluate(LINUX))
        ):
            name, version = requirement.strip().split("==")
            found[canon(name)] = version.strip()
    return found


def root():
    return tomllib.loads(text("pyproject.toml"))


def test_the_root_is_a_virtual_workspace_of_rag_core_rag_testing_and_evals():
    config = root()
    assert "project" not in config
    members = [
        "packages/rag_core",
        "packages/rag_testing",
        "packages/example_profile",
        "apps/evals",
        "apps/indexer",
        "apps/assistant_api",
    ]
    assert config["tool"]["uv"]["workspace"]["members"] == members
    assert text(".python-version").strip() == "3.12"
    # on 3.12 the resolver would build zstandard's sdist (setuptools<69 against build-constraint-dependencies); wheels cover it
    assert config["tool"]["uv"]["no-build-package"] == ["zstandard"]
    assert config["tool"]["uv"]["override-dependencies"] == ["pillow<14"]  # lifts docling's pillow<13 cap
    for member in members:
        assert tomllib.loads(text(member + "/pyproject.toml"))["project"]["requires-python"] == ">=3.12,<3.13"


def constraint_pins():
    listed = [entry.split("==") for entry in root()["tool"]["uv"]["constraint-dependencies"]]
    return {canon(name): version for name, version in listed}, len(listed)


def test_the_constraints_are_exact_pins_one_per_package():
    entries = root()["tool"]["uv"]["constraint-dependencies"]
    assert [e for e in entries if not re.fullmatch(r"[A-Za-z0-9._-]+==[^=<>!~;\s]+", e)] == []
    pinned, count = constraint_pins()
    assert len(pinned) == count


def test_every_package_the_box_images_install_is_pinned_at_its_installed_version():
    pinned, _ = constraint_pins()
    wanted = {**installed_on_linux("assistant-api"), **installed_on_linux("indexer")}
    assert {name: (version, pinned.get(name)) for name, version in wanted.items() if pinned.get(name) != version} == {}
    assert sorted(set(pinned) - set(wanted)) == []  # no pin for a package nothing installs


def test_no_group_or_testing_package_reaches_a_member():
    for member in (
        "packages/rag_core",
        "packages/example_profile",
        "apps/evals",
        "apps/indexer",
        "apps/assistant_api",
    ):
        deps = tomllib.loads(text(member + "/pyproject.toml"))["project"]["dependencies"]
        assert [d for d in deps if re.match(r"(pytest|hypothesis|ruff|basedpyright|rag-testing)", d)] == []


def test_the_root_build_context_is_an_allowlist():
    lines = [line.strip() for line in text(".dockerignore").splitlines() if line.strip() and not line.startswith("#")]
    assert lines[0] == "*"
    assert [line for line in lines if line.startswith("!")] == [
        "!pyproject.toml",
        "!uv.lock",
        "!packages/",
        "!apps/",
        "!profiles/",
    ]


def test_the_evals_image_installs_one_app_and_no_group_from_a_pinned_base():
    dockerfile = text("apps/evals/Dockerfile")
    assert BASE.search(dockerfile)
    assert dockerfile.count("FROM ${PYTHON_IMAGE}") == 2
    assert "RUN uv sync --frozen --no-default-groups --no-editable --package evals" in dockerfile
    assert re.search(r"^USER evals$", dockerfile, re.M) and 'ENTRYPOINT ["evals"]' in dockerfile
    assert "install -d -o evals -g evals /app/out" in dockerfile  # evals.settings' out_dir, writable by the job's user
    assert "rag_testing" not in dockerfile and "syntax=" not in dockerfile  # a syntax line would pull a frontend image


def test_the_fakes_image_installs_rag_testing_alone_and_no_service_image_installs_it():
    fakes = text("deploy/fakes/Dockerfile")
    # one base line, identical in all four Dockerfiles
    four = [
        BASE.findall(text(d + "/Dockerfile"))
        for d in ("apps/assistant_api", "apps/indexer", "apps/evals", "deploy/fakes")
    ]
    assert all(len(found) == 1 and found == four[0] for found in four)
    assert "RUN uv sync --frozen --no-default-groups --no-editable --package rag-testing" in fakes
    assert re.search(r"^USER fakes$", fakes, re.M) and "syntax=" not in fakes
    for path in ("apps/assistant_api/Dockerfile", "apps/indexer/Dockerfile", "apps/evals/Dockerfile"):
        # the assistant's lock layer copies every member's pyproject so that uv resolves the workspace; only an install counts
        body = re.sub(r"^COPY packages/rag_testing/pyproject\.toml packages/rag_testing/\n", "", text(path), flags=re.M)
        assert "rag_testing" not in body and "rag-testing" not in body, path


def test_ruff_checks_every_python_folder():
    ruff = root()["tool"]["ruff"]
    assert "extend-exclude" not in ruff and "force-exclude" not in ruff
    assert "exclude" not in ruff.get("lint", {})


def test_the_text_hooks_never_touch_byte_exact_fixtures_or_role_files():
    config = yaml.safe_load(text(".pre-commit-config.yaml"))
    hooks = {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}
    for hook in ("end-of-file-fixer", "mixed-line-ending"):
        pattern = re.compile(hooks[hook]["exclude"])
        for path in (
            "tests/contracts/fixtures/answer_shapes.json",
            "deploy/pgvector/roles/rag_app.txt",
        ):
            assert pattern.search(path), (hook, path)
        assert not pattern.search("apps/evals/src/evals/runner.py")
    assert hooks["mixed-line-ending"]["args"] == ["--fix=lf"]
    assert config["default_stages"] == ["pre-commit"] and hooks["just-pre-push"]["stages"] == ["pre-push"]
    assert {"gitleaks", "ruff-check", "ruff-format"} <= set(hooks)


def test_ci_checks_out_without_credentials_and_verifies_every_downloaded_tool():
    # no token left in .git/config for later steps; no `curl | bash`; every download checked by sha256
    ci = yaml.safe_load(text(".github/workflows/ci.yml"))
    steps = [step for job in ci["jobs"].values() for step in job["steps"]]
    checkouts = [step for step in steps if step.get("uses", "").startswith("actions/checkout@")]
    assert len(checkouts) == len(ci["jobs"]) and all(step["with"]["persist-credentials"] is False for step in checkouts)
    runs = "\n".join(step.get("run", "") for step in steps)
    assert re.search(r"\|\s*(bash|sh)\b", runs) is None  # nothing downloaded is piped into a shell
    assert runs.count("curl ") == runs.count("sha256sum -c") > 0
    assert all(re.fullmatch(r"[0-9a-f]{64}", ci["env"][name]) for name in ("JUST_SHA256", "GITLEAKS_SHA256"))


def test_ci_runs_on_pull_requests_into_main_and_by_hand_only():
    # other branches are checked by `just ci` locally; the Actions minutes (2,000 a month) go to pull requests into main (a push
    # run on the merge would repeat the pull request's run) and to runs started by hand
    ci = yaml.safe_load(text(".github/workflows/ci.yml"))
    assert ci[True] == {
        "pull_request": {"branches": ["main"]},  # PyYAML reads `on:` as True
        "workflow_dispatch": None,
    }
