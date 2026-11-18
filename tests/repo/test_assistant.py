"""The assistant's repository rules: no code reads a YAML config file; the stack's and the harness's technical values; the image
installs exactly the locked packages plus the two workspace packages; ruff excepts no member code."""

import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.markers import Marker

ROOT = Path(__file__).resolve().parents[2]
STACK = {  # the stack's values as RAG__ entries
    "RAG__DATABASE__HOST": "pgvector",
    "RAG__DATABASE__PORT": "5432",
    "RAG__MODEL_SERVERS__LLM_URL": "http://vllm:8000/v1",
    "RAG__MODEL_SERVERS__EMBEDDING_URL": "http://embedding_service:8001",
    "RAG__MODEL_SERVERS__RERANKER_URL": "http://reranker_service:8003",
    "RAG__COMPUTE__SLOTS": "1",
    "RAG__COMPUTE__SLOT_WAIT_S": "30",
    "RAG__TIMEOUTS__REQUEST_DEADLINE_S": "120",
    "RAG__TIMEOUTS__REQUEST_IDLE_S": "45",
    "RAG__TIMEOUTS__ANALYSIS_S": "20",
    "RAG__TIMEOUTS__RETRIEVAL_S": "10",
    "RAG__TIMEOUTS__PARENT_LOOKUP_S": "5",
    "RAG__TIMEOUTS__RERANKER_S": "15",
    "RAG__TIMEOUTS__LLM_READ_S": "60",
    "RAG__HTTP__CORS_DOMAIN": "example.org",
    "RAG__TRACING__URL": "",
}
CI_ONLY = {
    "RAG__COMPUTE__SLOT_WAIT_S": "3",
    "RAG__TIMEOUTS__REQUEST_DEADLINE_S": "25",
    "RAG__TIMEOUTS__REQUEST_IDLE_S": "5",
}
LINUX = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "python_version": "3.12",
    "python_full_version": "3.12.15",
    "os_name": "posix",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
}
# rag_core files that carry `# pyright: standard` on line 1: untyped code.
# The set only shrinks.
STANDARD_TYPED = {
    "packages/rag_core/src/rag_core/" + name
    for name in (
        "domain/answer_shapes.py",
        "domain/citations.py",
        "domain/context_budget.py",
        "domain/history.py",
        "domain/parent_rerank.py",
        "domain/query_analysis.py",
        "domain/ranking.py",
        "domain/source_meta.py",
        "adapters/http.py",
    )
}


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def backend(compose: str) -> dict:
    return yaml.safe_load(text(compose))["services"]["backend"]


def environment(service: dict) -> dict[str, str]:
    return dict(entry.split("=", 1) for entry in service["environment"])


def linux_pins(package: str) -> dict[str, str]:
    out = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-default-groups",
            "--no-hashes",
            "--no-emit-workspace",
            "--package",
            package,
            "--format",
            "requirements-txt",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pins: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        requirement, _, marker = line.partition(";")
        if marker.strip() and not Marker(marker.strip()).evaluate(LINUX):
            continue
        name, _, version = requirement.strip().partition("==")
        pins[canon(name.split("[")[0])] = version.strip()
    return pins


def test_no_code_reads_a_yaml_config_file() -> None:
    found = [
        str(p.relative_to(ROOT))
        for folder in ("apps/assistant_api", "packages/rag_core")
        for p in (ROOT / folder).rglob("*.py")
        if re.search(r"config\.gpu|/app/config|config\.yml", p.read_text(encoding="utf-8"))
    ]
    assert found == []


def test_the_stack_environment_holds_the_expected_values() -> None:
    env = environment(backend("deploy/compose.yaml"))
    assert {k: v for k, v in env.items() if k.startswith("RAG__")} == STACK
    assert "RAG_OVERLAY" not in env and "RAG_PROFILE" not in env  # no overlay for the backend; the profile is baked
    assert env["RAG__MODEL_SERVERS__EMBEDDING_URL"] + "/embed" == "http://embedding_service:8001/embed"
    assert env["RAG__MODEL_SERVERS__RERANKER_URL"] + "/v2/rerank" == "http://reranker_service:8003/v2/rerank"
    assert (
        env["RAG__COMPUTE__SLOTS"] == "1" and env["RAG__COMPUTE__SLOT_WAIT_S"] == "30"
    )  # the code default of the wait is 10


def test_the_stack_sets_the_slot_and_timeout_values_the_code_does_not_default() -> None:
    """A wrong default would change answers under load without a test failing: the pipeline slot wait is 30 s in the stack (the
    settings model has no default for it), one slot, request deadline 120 s and idle 45 s, CORS for example.org."""
    env = environment(backend("deploy/compose.yaml"))
    expected = {
        "RAG__COMPUTE__SLOTS": "1",
        "RAG__COMPUTE__SLOT_WAIT_S": "30",
        "RAG__TIMEOUTS__REQUEST_DEADLINE_S": "120",
        "RAG__TIMEOUTS__REQUEST_IDLE_S": "45",
        "RAG__TIMEOUTS__ANALYSIS_S": "20",
        "RAG__TIMEOUTS__RETRIEVAL_S": "10",
        "RAG__TIMEOUTS__PARENT_LOOKUP_S": "5",
        "RAG__TIMEOUTS__RERANKER_S": "15",
        "RAG__TIMEOUTS__LLM_READ_S": "60",
        "RAG__HTTP__CORS_DOMAIN": "example.org",
    }
    assert {key: env[key] for key in expected} == expected


def test_the_ci_environment_differs_only_in_the_three_harness_timeouts() -> None:
    env = {k: v for k, v in environment(backend("deploy/compose.ci.yaml")).items() if k.startswith("RAG__")}
    assert {k: v for k, v in env.items() if STACK.get(k) != v} == CI_ONLY and set(env) == set(STACK)


def test_the_shared_overlay_carries_no_assistant_section() -> None:
    for name in ("deploy/env/ci.yaml", "deploy/env/ci-example.yaml"):
        assert not {"compute", "timeouts", "http", "tracing", "server"} & set(yaml.safe_load(text(name)) or {}), name


@pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1", reason="needs the rag-ci-backend-example image `just images` builds"
)
def test_the_assistant_image_installs_exactly_its_locked_set() -> None:
    script = (
        "from importlib import metadata\n"
        "print('\\n'.join(d.metadata['Name'] + '==' + d.version for d in metadata.distributions()))"
    )
    out = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "python",
            "rag-ci-backend-example" + os.environ.get("RAG_CI_TAG", ""),
            "-c",
            script,
        ],  # per worktree
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    installed = {canon(line.split("==")[0]): line.split("==")[1] for line in out.split() if "==" in line}
    assert installed == linux_pins("assistant-api") | {"rag-core": "0.1.0", "assistant-api": "0.1.0"}


def test_the_dockerfile_copies_every_member_pyproject_before_the_lock_layer() -> None:
    members = tomllib.loads(text("pyproject.toml"))["tool"]["uv"]["workspace"]["members"]
    dockerfile = text("apps/assistant_api/Dockerfile")
    first_sync = dockerfile.index("--no-install-workspace")
    for member in members:
        assert dockerfile.index(f"COPY {member}/pyproject.toml {member}/") < first_sync, member
    assert re.search(r"^ARG PYTHON_IMAGE=python:3\.12\.\d+-slim-bookworm@sha256:[0-9a-f]{64}$", dockerfile, re.M)
    assert (
        "locales" not in dockerfile
        and "de_DE" not in dockerfile
        and "ENV RAG_PROFILE=/app/profile/profile.yaml" in dockerfile
    )
    assert dockerfile.rstrip().splitlines()[-2:] == [
        "ARG GIT_COMMIT",
        "ENV GIT_COMMIT=${GIT_COMMIT}",
    ]  # the indexer's form


def test_the_image_runs_the_app_module_with_its_own_cmd() -> None:
    dockerfile = text("apps/assistant_api/Dockerfile")
    assert 'CMD ["gunicorn", "assistant_api.app:create_app()", "-c", "gunicorn_conf.py"]' in dockerfile
    assert "COPY apps/assistant_api/gunicorn_conf.py /app/" in dockerfile and "fastapi_backend" not in dockerfile


def test_ruff_has_no_exception_for_member_code() -> None:
    ruff = tomllib.loads(text("pyproject.toml"))["tool"]["ruff"]
    assert [
        p
        for p in ruff.get("extend-exclude", []) + ruff.get("lint", {}).get("exclude", [])
        if p.startswith(("apps", "packages", "tests/assistant_api", "tests/rag_core"))
    ] == []
    assert [p for p in ruff.get("lint", {}).get("per-file-ignores", {}) if p.startswith(("apps", "tests/evals"))] == []
    assert "exclude" not in ruff.get("format", {})


def test_the_standard_typed_files_are_the_named_ones() -> None:
    marked = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "packages" / "rag_core").rglob("*.py")
        if p.read_text(encoding="utf-8").startswith("# pyright: standard")
    }
    assert marked == STANDARD_TYPED
