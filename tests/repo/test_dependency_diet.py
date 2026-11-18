"""Dependency diet: the packages nothing uses stay out of both images, every package the code imports is installed, and
Langfuse keeps the packages its handler needs."""

import ast
import importlib.metadata
import os
import re
import subprocess
import sys
import tomllib

import pytest
import yaml
from packaging.markers import Marker

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# the folders and files whose code runs in each image (the probe runner has its own image)
CODE = {
    "backend": ("apps/assistant_api", "packages/rag_core"),
    "indexer": (
        "apps/indexer",
        "packages/rag_core/src/rag_core/settings",
        "packages/rag_core/src/rag_core/domain/sources.py",
        "packages/rag_core/src/rag_core/domain/text.py",
        "packages/rag_core/src/rag_core/ports/sources.py",
        "packages/rag_core/src/rag_core/adapters/db.py",
        "packages/rag_core/src/rag_core/adapters/embedding_http.py",
        "packages/rag_core/src/rag_core/adapters/http.py",
    ),
}
# the indexer's requirements are its members' dependencies; each image installs its members' part of uv.lock
INDEXER_PROJECTS = ("apps/indexer/pyproject.toml",)
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
# packages dropped from the images, including pydantic-settings (backend), langchain and python-dotenv (indexer);
# langchain stays in the backend: the Langfuse handler imports it (2.x langfuse.callback, 4.x langfuse.langchain; see the
# langfuse test below)
DROPPED = {
    "backend": {
        "aiofiles",
        "azure-core",
        "azure-storage-blob",
        "cachetools",
        "cohere",
        "ipython",
        "langchain-community",
        "langchain-huggingface",
        "nltk",
        "opentelemetry-api",
        "opentelemetry-exporter-otlp",
        "opentelemetry-sdk",
        "pydantic-settings",
        "python-dotenv",
        "rank-bm25",
        "redis",
        "sentence-transformers",
        "torch",
    },
    "indexer": {
        "beautifulsoup4",
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "langchain-openai",
        "pandas",
        "python-dotenv",
        "rank-bm25",
        "sentence-transformers",
        "tqdm",
    },
}


def text(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line):
    return canon(re.split(r"[\[<>=!~; ]", line, maxsplit=1)[0])


def linux_lock(*packages):
    """name -> version of every package the workspace lock installs for the workspace packages on linux x86_64 (what the images get)."""
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


def backend_requirements():
    """What the assistant app declares (its [project] dependencies; the workspace package rag-core, a member, left out)."""
    declared = tomllib.loads(text("apps/assistant_api/pyproject.toml"))["project"]["dependencies"]
    return {requirement_name(d) for d in declared} - {"rag-core"}


def requirements(service):
    if service == "backend":
        return backend_requirements()
    return {
        requirement_name(d) for path in INDEXER_PROJECTS for d in tomllib.loads(text(path))["project"]["dependencies"]
    }


def installed(service):
    if service == "backend":
        return set(linux_lock("assistant-api"))
    return set(linux_lock("indexer")) | {"rag-core", "indexer"}  # the workspace members install as packages too


def test_the_indexer_requires_no_dropped_package():
    assert sorted(requirements("indexer") & DROPPED["indexer"]) == []


@pytest.mark.parametrize("service", sorted(CODE))
def test_every_requirement_has_its_pin(service):
    assert sorted(requirements(service) - installed(service)) == []


def imported_modules(paths):
    """The local module names and the full dotted modules the code under the paths (folders or files) imports."""
    local, modules = set(), set()
    for path in paths:
        full = os.path.join(ROOT, path)
        walk = [(os.path.dirname(full), [], [os.path.basename(full)])] if os.path.isfile(full) else os.walk(full)
        for dirpath, dirnames, files in walk:
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            local.update(dirnames)
            for name in files:
                if not name.endswith(".py"):
                    continue
                local.add(name[:-3])
                with open(os.path.join(dirpath, name), encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                        modules.add(node.module)
    return local, modules


def third_party_imports(paths):
    local, modules = imported_modules(paths)
    return sorted({m.split(".")[0] for m in modules} - local - set(sys.stdlib_module_names))


def test_the_command_line_imports_only_the_standard_library():
    assert set(third_party_imports(["src/rag_tuda"])) - {"rag_tuda"} == set()


@pytest.mark.parametrize("service", sorted(CODE))
def test_every_package_the_code_imports_is_still_installed(service):
    found = installed(service)
    known = importlib.metadata.packages_distributions()  # import name -> distributions, from the test venv
    # a package the test venv lacks (fastapi, langfuse, docling, ...) is published under its import name
    missing = [n for n in third_party_imports(CODE[service]) if not {canon(d) for d in known.get(n, [n])} & found]
    assert missing == []


def test_no_dropped_package_is_required_again():
    # what left the backend's requirements stays out; a change that needs one of these
    # packages names the reason here (OpenTelemetry returns only as the Langfuse SDK's own dependencies)
    assert sorted(backend_requirements() & DROPPED["backend"]) == []


def test_the_backend_keeps_langchain_while_the_langfuse_handler_needs_it():
    # langfuse 2.x's langfuse.callback and 4.x's langfuse.langchain import langchain at import time but declare it only as
    # an extra, so no metadata closure keeps it
    users = sorted(
        m
        for m in imported_modules(CODE["backend"])[1]
        if m.split(".")[:2] in (["langfuse", "callback"], ["langfuse", "langchain"])
    )
    assert users  # the backend traces through the LangChain handler
    assert "langchain" in backend_requirements(), users


def test_the_backend_carries_the_langfuse_sdks_opentelemetry_and_never_switches_it_off():
    # Langfuse SDK 4.x sends its traces as OpenTelemetry spans: OTEL_SDK_DISABLED=true in the
    # backend's environment would switch tracing off without an error, and the SDK needs its OpenTelemetry packages, which
    # are not requirements (they come in as the SDK's dependencies)
    backend = yaml.safe_load(text("deploy/compose.yaml"))["services"]["backend"]
    assert [e for e in backend["environment"] if e.split("=", 1)[0].startswith("OTEL_")] == []
    installed = linux_lock("assistant-api")
    assert installed["langfuse"].startswith("4.")
    assert {"opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-http"} <= set(installed)
