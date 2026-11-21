"""Compose services: the unused Streamlit demo is gone; the evals job; the user stack (loopback ports, pinned images, CPU base,
NVIDIA and vLLM overlays)."""

import json
import os
import re
import shutil
import subprocess

import pytest
import yaml

from rag_core.settings.card import effective_window, load_card

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = "deploy/compose.yaml"
NVIDIA = "deploy/compose.nvidia.yaml"
VLLM = "deploy/compose.vllm.yaml"
NATIVE = "deploy/compose.native.yaml"
COMPOSE_FILES = (
    BASE,
    NVIDIA,
    VLLM,
    NATIVE,
    "deploy/compose.ci.yaml",
    "deploy/compose.ci.example.yaml",
    "deploy/compose.ci.tu.yaml",
)
HAS_COMPOSE = (
    shutil.which("docker") is not None
    and subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0
)
needs_compose = pytest.mark.skipif(not HAS_COMPOSE, reason="needs the docker CLI with the compose plugin (no daemon)")


class ComposeLoader(yaml.SafeLoader):
    """safe_load that reads compose's `!reset` (an overlay removing a key) as null."""


ComposeLoader.add_constructor("!reset", lambda loader, node: None)


def services(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return yaml.load(fh, ComposeLoader)["services"]


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
        "args": {"PROFILE": "${RAG_PROFILE:-example}"},
    }
    assert "networks" not in evals  # the project's default network, with the backend
    assert evals["volumes"] == ["../data/eval_runs:/app/data/eval_runs:ro"]
    assert evals["secrets"] == ["backend_token"]
    assert "OPENAI_API_KEY=not-needed" in evals["environment"]


def test_every_container_that_reads_models_mounts_the_cache_the_cli_checked():
    """`rag up --mode compose` writes RAG_HF_CACHE (the Hugging Face home of the cache it checks) into compose.env."""
    mount = "${RAG_HF_CACHE:-~/.cache/huggingface}:/root/.cache/huggingface"
    for name, service in (
        ("deploy/compose.yaml", "embedding_service"),
        ("deploy/compose.yaml", "reranker_service"),
        ("deploy/compose.yaml", "indexer"),
        ("deploy/compose.vllm.yaml", "vllm"),
    ):
        assert mount in services(name)[service]["volumes"], (name, service)
        assert "~/.cache/huggingface:/root" not in str(services(name)[service]["volumes"]), (name, service)


def test_the_harness_keeps_the_containers_data_in_a_named_volume_not_a_host_folder():
    """The backend and indexer run as root: files they create in a bind-mounted host folder could not be removed by the
    user who starts the next harness leg (Linux). `down -v` removes the volume."""
    found = services("deploy/compose.ci.yaml")
    for name in ("backend", "indexer"):
        mounts = [v for v in found[name]["volumes"] if v.endswith(":/app/data")]
        assert [m.split(":")[0] for m in mounts] == ["ci-data"], name
    with open(os.path.join(ROOT, "deploy/compose.ci.yaml"), encoding="utf-8") as fh:
        assert "ci-data" in yaml.safe_load(fh)["volumes"]


def test_the_backend_token_reaches_the_evals_job_as_a_file_not_a_variable():
    with open(os.path.join(ROOT, "deploy/compose.yaml"), encoding="utf-8") as fh:
        compose = yaml.safe_load(fh)
    assert compose["secrets"]["backend_token"] == {"environment": "BACKEND_REFRESH_TOKEN"}
    assert [e for e in compose["services"]["evals"].get("environment", []) if "TOKEN" in e] == []


def render(*files, args=(), env=None):
    """docker compose config for the files, as the project rag-tuda; no daemon is needed."""
    command = ["docker", "compose", "-p", "rag-tuda"]
    for name in files:
        command += ["-f", name]
    return subprocess.run(
        [*command, *args, "config", "--format", "json"], cwd=ROOT, capture_output=True, text=True, timeout=60, env=env
    )


@needs_compose
@pytest.mark.parametrize("overlays", [(), (NVIDIA,), (VLLM,), (NVIDIA, VLLM)])
def test_compose_renders_every_stack_combination_with_every_secret_set(tmp_path, overlays):
    names = set()
    for name in (BASE, *overlays):
        with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
            names |= set(re.findall(r"\$\{([A-Z0-9_]+):\?", fh.read()))
    env = tmp_path / "full.env"
    env.write_text("".join("%s=x\n" % k for k in sorted(names)), encoding="utf-8")
    result = render(
        BASE,
        *overlays,
        args=("--env-file", str(env), "--profile", "evals", "--profile", "migrate", "--profile", "build"),
    )
    assert result.returncode == 0, result.stderr


@needs_compose
def test_the_stack_takes_the_card_the_window_and_the_secret_folder_from_the_env_file(tmp_path):
    env = tmp_path / "chosen.env"
    env.write_text("RAG__CARD=m-qwen3.6-35b-a3b\nOLLAMA_CONTEXT_LENGTH=16000\nRAG_ROLES_DIR=/roles\n", encoding="utf-8")
    result = render(BASE, args=("--env-file", str(env)))
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["services"]["backend"]["environment"]["RAG__CARD"] == "m-qwen3.6-35b-a3b"
    assert rendered["services"]["ollama"]["environment"]["OLLAMA_CONTEXT_LENGTH"] == "16000"
    files = {name: spec["file"] for name, spec in rendered["secrets"].items() if "file" in spec}
    assert files["pg_vector_password"] == "/roles/pg_vector_password.txt"
    assert files["rag_migrator_password"] == "/roles/rag_migrator_password.txt"
    assert files["rag_app_user"].endswith("deploy/pgvector/roles/rag_app.txt")  # the role names are tracked


@needs_compose
def test_the_indexer_reads_its_profile_and_overlay_and_the_images_are_built_with_the_profile(tmp_path):
    env = tmp_path / "chosen.env"
    env.write_text("RAG_PROFILE=other\nRAG_INDEXER_OVERLAY=/rag/overlay.yaml\n", encoding="utf-8")
    result = render(BASE, args=("--env-file", str(env), "--profile", "build"))
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)["services"]
    indexer = rendered["indexer"]
    mounts = {v["target"]: v["source"] for v in indexer["volumes"]}
    assert mounts["/app/overlay.yaml"] == "/rag/overlay.yaml"
    assert mounts["/app/profiles/other"].endswith("/profiles/other")
    assert indexer["environment"]["RAG_OVERLAY"] == "/app/overlay.yaml"
    assert indexer["environment"]["RAG__CRAWL__CONTACT"] is None  # nobody set it: the bare user agent
    assert indexer["build"]["args"]["PROFILE"] == rendered["backend"]["build"]["args"]["PROFILE"] == "other"


def test_the_compose_indexer_keeps_the_crawl_under_the_rag_folder_that_down_purge_deletes():
    # the same folders as the native stack (runtime.indexer_paths): pages, PDFs, extracted text and snapshots in .rag/
    from indexer.settings import PathsSection

    indexer = services(BASE)["indexer"]
    mounts = dict(v.split(":")[::-1][:2][::-1] for v in indexer["volumes"] if v.startswith("../.rag/"))
    crawl = PathsSection().crawl_dir
    assert mounts == {"../.rag/crawl": crawl, "../.rag/snapshots": os.path.join(os.path.dirname(crawl), "snapshots")}
    env = dict(e.split("=", 1) for e in indexer["environment"] if "=" in e)
    assert env["RAG__PATHS__PDF_DIR"] == f"{crawl}/pdfs" and env["RAG__PATHS__EXTRACTED_DIR"] == f"{crawl}/extracted"


def test_the_default_overlay_names_the_example_profiles_law_texts():
    with open(os.path.join(ROOT, "deploy/env/compose-indexer.yaml"), encoding="utf-8") as fh:
        overlay = yaml.safe_load(fh)
    assert overlay == {"connectors": {"gesetze": {"dir": "/app/profiles/example/documents"}}}
    assert os.path.isdir(os.path.join(ROOT, "profiles/example/documents"))


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
    # sources.yaml and roles.yaml sit beside the profile file, so the image carries the whole profile folder
    assert "COPY profiles/${PROFILE}/ /app/profile/\nENV RAG_PROFILE=/app/profile/profile.yaml" in dockerfile
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert "--profile build run --rm indexer build`" in readme
    assert services("deploy/compose.yaml")["indexer"]["profiles"] == ["build"]
    harness = open(os.path.join(ROOT, "packages/rag_testing/src/rag_testing/harness.py"), encoding="utf-8").read()
    assert '"--no-deps", "indexer"]' in harness and '[*command, "build", *flags]' in harness


def all_services(name):
    return {service: spec for service, spec in services(name).items() if spec is not None}  # `!reset` entries are null


def test_every_published_port_is_bound_to_loopback_and_the_stack_publishes_only_the_backend():
    for name in COMPOSE_FILES:
        for service, spec in all_services(name).items():
            for port in spec.get("ports", []):
                assert str(port).startswith("127.0.0.1:"), (name, service, port)
    published = {s: spec["ports"] for s, spec in services(BASE).items() if "ports" in spec}
    assert published == {"backend": ["127.0.0.1:8700:8000"]}
    for overlay in (NVIDIA, VLLM):
        assert [s for s, spec in all_services(overlay).items() if "ports" in spec] == [], overlay


def test_every_stack_service_is_built_here_or_pinned_by_digest():
    for name in (BASE, VLLM):
        for service, spec in all_services(name).items():
            if "build" in spec or (name != BASE and service in services(BASE)):  # an overlay only adjusts these
                continue
            image = spec.get("image", "")
            if service == "migrate":  # a one-off of the backend image the stack builds
                assert image == "rag-tuda-backend:latest"
                continue
            assert re.fullmatch(r"[\w./-]+:[\w.-]+@sha256:[0-9a-f]{64}", image), (name, service, image)


def test_the_base_stack_runs_on_the_cpu_with_ollama():
    found = services(BASE)
    assert "vllm" not in found and "ollama" in found
    assert "ports" not in found["ollama"] and "ollama-models:/root/.ollama" in found["ollama"]["volumes"]
    assert "ollama" in found["backend"]["depends_on"] and "vllm" not in found["backend"]["depends_on"]
    for service, spec in found.items():
        assert "devices" not in str(spec.get("deploy", {})) and "gpu" not in str(spec.get("deploy", {})), service
        assert not [e for e in spec.get("environment", []) if "cuda" in e.lower() or "RAG_DEVICE" in e], service


def default_of(entry):
    """The default of a `${NAME:-default}` entry value."""
    return re.fullmatch(r"\$\{\w+:-([^}]*)\}", entry).group(1)


def test_ollama_context_length_is_the_default_cards_window():
    from_stack = {e.split("=", 1)[0]: e.split("=", 1)[1] for e in services(BASE)["backend"]["environment"] if "=" in e}
    card = load_card(default_of(from_stack["RAG__CARD"]), os.path.join(ROOT, "models", "cards"))
    ollama = dict(e.split("=", 1) for e in services(BASE)["ollama"]["environment"])
    assert int(default_of(ollama["OLLAMA_CONTEXT_LENGTH"])) == effective_window(card)
    assert from_stack["RAG__LLM_URL"] == "http://ollama:11434/v1"
    assert card.answer.provider == "ollama"


def test_the_native_overlay_publishes_postgres_alone_on_loopback():
    found = services(NATIVE)
    assert set(found) == {"pgvector"} and found["pgvector"] == {"ports": ["127.0.0.1:55432:5432"]}


def test_the_nvidia_overlay_adds_the_gpus_and_the_device_to_the_model_servers_only():
    found = services(NVIDIA)
    assert set(found) == {"embedding_service", "reranker_service", "ollama"}
    for service in ("embedding_service", "reranker_service"):
        assert "RAG_DEVICE=cuda" in found[service]["environment"], service
    for service, spec in found.items():
        (device,) = spec["deploy"]["resources"]["reservations"]["devices"]
        assert device == {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}, service


@needs_compose
def test_the_vllm_overlay_replaces_ollama_with_the_cards_server():
    for files in ((BASE, VLLM), (BASE, NVIDIA, VLLM)):
        result = render(*files)
        assert result.returncode == 0, result.stderr
        rendered = json.loads(result.stdout)["services"]
        assert "ollama" not in rendered, files
        assert set(rendered["backend"]["depends_on"]) == {"pgvector", "vllm"}, files
        env = rendered["backend"]["environment"]
        card = load_card(env["RAG__CARD"], os.path.join(ROOT, "models", "cards"))
        assert card.answer.provider == "vllm" and env["RAG__LLM_URL"] == "http://vllm:8000/v1"
        command = " ".join(rendered["vllm"]["command"]) + " "
        assert f"--model {card.answer.model} " in command and "ports" not in rendered["vllm"]
        assert f"--max-model-len {card.answer.context_window} " in command
        assert {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}.items() <= rendered["vllm"]["environment"].items(), (
            files
        )
    assert "vllm" not in json.loads(render(BASE).stdout)["services"]


@needs_compose
def test_the_indexer_gets_the_users_crawl_contact_from_the_shell_that_runs_compose_and_the_file_holds_none(tmp_path):
    env = {**os.environ, "RAG__CRAWL__CONTACT": "me@example.org"}
    rendered = json.loads(render(BASE, args=("--profile", "build"), env=env).stdout)["services"]
    assert rendered["indexer"]["environment"]["RAG__CRAWL__CONTACT"] == "me@example.org"
    assert "RAG__CRAWL__CONTACT=" not in open(os.path.join(ROOT, BASE), encoding="utf-8").read()
