"""The model servers: each its own uv project outside the workspace, whose lock installs on linux exactly its
constraint pins; images built from those locks on digest-pinned Python 3.12 slim bases, with the model identity baked in
and no download at runtime; compose builds them from their folders and checks /readyz only."""

import json
import os
import re
import subprocess
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml
from packaging.markers import Marker

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVERS = {"embedding": ("embedding_service", 8001), "reranker": ("reranker_service", 8003)}
MINOR = {"embedding": "12", "reranker": "12"}
PYTHON_ARG = re.compile(r"^ARG PYTHON_IMAGE=python:(3\.(\d+)\.\d+)-slim-bookworm@sha256:[0-9a-f]{64}$", re.M)
COMMIT = re.compile(r"[0-9a-f]{40}")


def text(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def linux_env(server):
    """The marker environment of the server image: linux x86_64 on the Python of the server's image."""
    version, minor = PYTHON_ARG.search(text("model_servers/%s/Dockerfile" % server)).groups()
    return {
        "sys_platform": "linux",
        "platform_machine": "x86_64",
        "platform_system": "Linux",
        "python_version": "3." + minor,
        "python_full_version": version,
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "os_name": "posix",
        "extra": "",
    }


def linux_set(server):
    """name -> version of every package the lock installs on linux x86_64 on the server's image Python, dev group left out."""
    out = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-header",
            "--no-annotate",
            "--project",
            os.path.join(ROOT, "model_servers", server),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    linux, found = linux_env(server), {}
    for line in out.splitlines():
        requirement, _, marker = line.partition(";")
        if not requirement.strip() or requirement.startswith("#"):
            continue
        if marker.strip() and not Marker(marker.strip()).evaluate(linux):
            continue
        name, version = requirement.strip().split("==")
        found[norm(name)] = version.strip()
    return found


def test_the_model_servers_stay_outside_the_workspace():
    workspace = tomllib.loads(text("pyproject.toml"))["tool"]["uv"]["workspace"]
    assert workspace["exclude"] == ["model_servers/*"]
    assert [m for m in workspace["members"] if m.startswith("model_servers")] == []


@pytest.mark.parametrize("server", sorted(SERVERS))
def test_each_lock_installs_on_linux_exactly_its_constraint_pins(server):
    project = tomllib.loads(text("model_servers/%s/pyproject.toml" % server))
    image = {norm(pin.split("==")[0]): pin.split("==")[1] for pin in project["tool"]["uv"]["constraint-dependencies"]}
    assert linux_set(server) == image
    assert (
        project["project"]["requires-python"] == "==3.%s.*" % MINOR[server]
        and project["tool"]["uv"]["package"] is False
    )
    assert [d for d in project["project"]["dependencies"] if "==" not in d] == []  # every root at its image version
    assert project["dependency-groups"]["dev"] == ["pytest"]


def test_both_servers_run_one_gpu_stack():
    sets = {server: linux_set(server) for server in SERVERS}
    stack = (
        "torch",
        "transformers",
        "sentence-transformers",
        "numpy",
        "tokenizers",
        "triton",
        "huggingface-hub",
        "hf-xet",
    )
    names = {n for found in sets.values() for n in found if n in stack or n.startswith(("cuda-", "nvidia-"))}
    assert names and [n for n in sorted(names) if len({found.get(n) for found in sets.values()}) > 1] == []


def env_of(server):
    """The ENV lines of a server's Dockerfile (one variable per line; single quotes around a JSON value)."""
    found = {}
    for key, value in re.findall(r"^ENV (\w+)=(.*)$", text("model_servers/%s/Dockerfile" % server), re.M):
        found[key] = value[1:-1] if len(value) > 1 and value[0] == value[-1] == "'" else value
    return found


def reference(server):
    return json.loads(text("model_servers/%s/%s_server/reference.json" % (server, server)))


@pytest.mark.parametrize("server", sorted(SERVERS))
def test_each_image_installs_its_lock_on_its_pinned_python_slim_base_and_never_downloads(server):
    dockerfile = text("model_servers/%s/Dockerfile" % server)
    assert PYTHON_ARG.search(dockerfile).group(2) == MINOR[server]
    assert dockerfile.count("FROM ${PYTHON_IMAGE}") == 2 and "FROM ${PYTHON_IMAGE} AS builder" in dockerfile
    assert "nvidia/cuda" not in dockerfile and "update-alternatives" not in dockerfile
    assert not re.search(r"^\s+python3", dockerfile, re.M)  # no apt Python: the image's own interpreter
    assert re.search(
        r"^ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/local/bin/python3\.%s " % MINOR[server], dockerfile, re.M
    )
    assert "RUN pip3 install --no-cache-dir uv==0.12.23" in dockerfile and "RUN uv sync --frozen --no-dev" in dockerfile
    assert "pip install -r" not in dockerfile and "syntax=" not in dockerfile
    env = env_of(server)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert (
        env["NVIDIA_VISIBLE_DEVICES"] == "all" and env["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"
    )  # the CUDA base set them
    port = SERVERS[server][1]
    cmd = 'CMD ["uvicorn", "--factory", "%s_server.main:create_app", "--host", "0.0.0.0", "--port", "%d"]' % (
        server,
        port,
    )
    assert cmd in dockerfile
    assert text("model_servers/%s/.dockerignore" % server).split() == [".venv", "tests", "**/__pycache__"]


def test_the_reranker_runtime_has_no_compiler():
    # the embedding image runs the same torch and triton wheels without one
    runtime = text("model_servers/reranker/Dockerfile").split("FROM ${PYTHON_IMAGE}")[2]
    assert "apt-get" not in runtime  # no package is installed in the runtime stage, so no compiler either


@pytest.mark.parametrize("server", sorted(SERVERS))
def test_each_model_server_runs_on_the_workspace_python(server):
    line = PYTHON_ARG.search(text("model_servers/%s/Dockerfile" % server)).group(0)
    assert line in text("apps/assistant_api/Dockerfile")


def test_the_embedding_image_carries_the_identity_its_reference_was_taken_with():
    env, ref = env_of("embedding"), reference("embedding")
    assert env["EMBEDDING_MODEL"] == ref["model"] == "Snowflake/snowflake-arctic-embed-m-v2.0"
    assert env["EMBEDDING_REVISION"] == ref["revision"] and COMMIT.fullmatch(ref["revision"])
    assert COMMIT.fullmatch(env["EMBEDDING_CODE_REVISION"])
    assert json.loads(env["EMBEDDING_CONFIG_KWARGS"]) == {
        "use_memory_efficient_attention": False,
        "unpad_inputs": False,
    }
    assert len(ref["texts"]) == len(ref["vectors"]) == 3 and ref["texts"][0].startswith("query: ")
    assert [len(v) for v in ref["vectors"]] == [768] * 3


def test_the_reranker_image_carries_the_identity_its_reference_was_taken_with():
    env, ref = env_of("reranker"), reference("reranker")
    assert env["RERANKER_MODEL"] == ref["model"] == "BAAI/bge-reranker-v2-m3"
    assert env["RERANKER_REVISION"] == ref["revision"] and COMMIT.fullmatch(ref["revision"])
    assert ref["query"] and ref["document"] and 0 < ref["score"] < 1


def compose(name):
    return yaml.safe_load(text(name))


def test_compose_builds_the_servers_from_their_folders():
    services = compose("deploy/compose.yaml")["services"]
    assert services["embedding_service"]["build"] == {"context": "../model_servers/embedding"}
    assert services["reranker_service"]["build"] == {"context": "../model_servers/reranker"}


def healthcheck(service):
    test = compose("deploy/compose.yaml")["services"][service]["healthcheck"]["test"]
    assert test[:3] == ["CMD", "python", "-c"], service
    return test[3]


@pytest.mark.parametrize("service, port", [("embedding_service", 8001), ("reranker_service", 8003)])
@pytest.mark.parametrize("readyz, healthy", [(200, True), (503, False), (404, False), (500, False)])
def test_the_healthcheck_reads_readyz_only(service, port, readyz, healthy):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response({"/readyz": readyz}.get(self.path, 404))
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code = healthcheck(service).replace("localhost:%d" % port, "127.0.0.1:%d" % server.server_address[1])
        assert code != healthcheck(service)  # the snippet names the service's own port
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30)
    finally:
        server.shutdown()
    assert (result.returncode == 0) is healthy


def test_a_server_that_does_not_answer_is_unhealthy():
    code = healthcheck("embedding_service").replace("localhost:8001", "127.0.0.1:9")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30).returncode != 0


def test_just_runs_the_model_servers_tests_offline_and_checks_their_locks():
    justfile = text("justfile")
    assert re.search(
        r'^ci: lint types imports unit cli db-tests contracts model-servers images \(harness "example"\) \(harness "tu"\)$',
        justfile,
        re.M,
    )
    for server in SERVERS:
        assert "uv lock --check --project model_servers/%s" % server in justfile
        assert "cd model_servers/%s && HF_HUB_OFFLINE=1 uv run --frozen pytest -q" % server in justfile
