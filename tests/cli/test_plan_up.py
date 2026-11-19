"""plan_up: the steps per state of the machine. The plan is a pure function; nothing here starts a process."""

import json
from pathlib import Path

import pytest

from rag_tuda import cards, stack, state

S4B = cards.named("s-qwen3-4b-instruct-2507")
M35 = cards.named("m-qwen3.6-35b-a3b")
GROQ = cards.named("l-groq-gpt-oss-120b")
XL = cards.named("xl-vllm-example")
TAG = "qwen3:4b-instruct-2507-q4_K_M"
DERIVED = "rag-s-qwen3-4b-instruct-2507"
ALL_HF = frozenset(pin.key for pin in stack.PINNED.values())
RUNNING = frozenset({"embedding", "reranker", "backend"})


@pytest.fixture
def current(tmp_path):
    return state.ensure(tmp_path)


def facts(tmp_path, card=S4B, **kwargs) -> stack.Facts:
    return stack.Facts(root=str(tmp_path), card=card, **kwargs)


def args(card=S4B, **kwargs) -> stack.Args:
    return stack.Args(card["name"], **kwargs)


def settled(tmp_path, card=S4B, **kwargs) -> stack.Facts:
    """Everything present and running."""
    derived = f"rag-{card['name']}"
    window = {"num_ctx": str(cards.window(card)), "temperature": "0.1", "top_p": "0.9"}
    values = {
        "ollama_up": True,
        "ollama_models": frozenset({card["answer"]["model"], f"{derived}:latest"}),
        "ollama_derived": {derived: window},
        "hf": ALL_HF,
        "alive": RUNNING,
        "ready": RUNNING,
        "postgres": True,
        "index_built": True,
    }
    return facts(tmp_path, card, **{**values, **kwargs})


def names(steps) -> list[str]:
    return [f"{s.kind}:{s.name}" for s in steps if s.kind != "say"]


def settle(current, a: stack.Args) -> state.State:
    """The state after a first successful up."""
    data = {**current.data, **stack._signature(a), "first_run_done": True}
    return state.State(current.dir, current.token, current.passwords, data)


def test_up_from_nothing_runs_the_steps_in_order(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, ollama_up=True), args())
    assert names(steps) == [
        "run:download-embedding",
        "run:download-reranker",
        f"run:pull-{TAG}",
        "write:compose-env",
        "run:postgres",
        "spawn:embedding",
        "wait:embedding",
        "spawn:reranker",
        "wait:reranker",
        "run:migrate",
        "write:overlay",
        "run:index-build",
        "request:derived-model",
        "spawn:backend",
        "wait:backend",
        "request:warm-up",
        "write:state",
    ]


def test_up_is_idempotent(tmp_path, current):
    a = args()
    steps = stack.plan_up(settle(current, a), settled(tmp_path), a)
    assert names(steps) == []
    assert [s.text for s in steps] == [
        "postgres: already running",
        "embedding: already running",
        "reranker: already running",
        "backend: already running",
    ]


def test_a_second_up_changes_nothing_after_the_first_one_recorded_itself(tmp_path, current):
    a = args()
    first = stack.plan_up(current, facts(tmp_path, ollama_up=True), a)
    recorded = json.loads(next(s.text for s in first if s.name == "state"))
    assert recorded["card"] == S4B["name"] and recorded["first_run_done"] is True
    again = state.State(current.dir, current.token, current.passwords, recorded)
    assert names(stack.plan_up(again, settled(tmp_path), a)) == []


def test_up_plan_warms_the_answer_model(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, ollama_up=True), args())
    order = names(steps)
    assert order.index("request:warm-up") > order.index("wait:backend")
    (warm,) = [s for s in steps if s.name == "warm-up"]
    body = json.loads(warm.body)
    assert warm.url == "http://127.0.0.1:11434/v1/chat/completions"
    assert body["model"] == DERIVED and body["max_tokens"] == 1 and body["stream"] is False


def test_a_hosted_answer_is_not_warmed_and_its_analysis_model_is_made_with_its_own_window(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, GROQ, ollama_up=True, groq_key=True), args(GROQ))
    assert "warm-up" not in [s.name for s in steps]
    assert [s.name for s in steps if s.name == "derived-model"] == []  # no Ollama answer model to derive
    (create,) = [s for s in steps if s.name == "analysis-model"]
    assert json.loads(create.body)["model"] == "rag-l-groq-gpt-oss-120b-analysis"
    assert json.loads(create.body)["parameters"]["num_ctx"] == 8000
    order = names(steps)
    assert order.index("request:analysis-model") < order.index("spawn:backend")


def test_a_hosted_card_is_refused_without_groq_api_key(tmp_path, current):
    with pytest.raises(stack.StackError, match=r"GROQ_API_KEY in the environment.*passed to the backend only"):
        stack.plan_up(current, facts(tmp_path, GROQ, ollama_up=True), args(GROQ))


def test_the_key_reaches_the_backend_spawn_only_by_name(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, GROQ, ollama_up=True, groq_key=True), args(GROQ))
    keyed = [s for s in steps if "GROQ_API_KEY" in s.inherit]  # the index build inherits the crawl contact, not the key
    assert [s.name for s in keyed] == ["backend"] and keyed[0].kind == "spawn"
    backend = next(s for s in steps if s.kind == "spawn" and s.name == "backend")
    assert backend.env["RAG__LLM_ANALYSIS_MODEL"] == "rag-l-groq-gpt-oss-120b-analysis"
    local = stack.plan_up(current, facts(tmp_path, ollama_up=True), args())
    assert not [s for s in local if "GROQ_API_KEY" in s.inherit]


def test_a_remote_card_without_an_analysis_block_is_refused_before_anything_is_planned(tmp_path, current):
    bare = {k: v for k, v in GROQ.items() if k != "analysis"}
    with pytest.raises(ValueError, match=r"\[analysis\] block"):
        stack.plan_up(current, facts(tmp_path, bare, ollama_up=True, groq_key=True), args(bare))


def test_the_derived_model_carries_the_cards_window_and_is_made_from_its_model(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, ollama_up=True), args())
    (create,) = [s for s in steps if s.name == "derived-model"]
    assert create.url == "http://127.0.0.1:11434/api/create"
    assert json.loads(create.body) == {
        "model": DERIVED,
        "from": TAG,
        "parameters": {"num_ctx": 8000, "temperature": 0.1, "top_p": 0.9},
        "stream": False,
    }
    big = stack.plan_up(current, facts(tmp_path, M35, ollama_up=True), args(M35))
    assert json.loads(next(s.body for s in big if s.name == "derived-model"))["parameters"]["num_ctx"] == 16000


def test_a_derived_model_with_another_window_is_made_again(tmp_path, current):
    a = args()
    stale = settled(tmp_path, ollama_derived={DERIVED: {"num_ctx": "4096", "temperature": "0.1", "top_p": "0.9"}})
    steps = stack.plan_up(settle(current, a), stale, a)
    assert names(steps) == ["request:derived-model", "request:warm-up"]  # the loaded runner is stale: load it again


def test_a_changed_base_model_makes_the_derived_model_again(tmp_path, current):
    plain, other = args(), args(model="mistral-small3.2:24b")
    present = settled(tmp_path, ollama_models=frozenset({TAG, "mistral-small3.2:24b", f"{DERIVED}:latest"}))
    steps = stack.plan_up(settle(current, plain), present, other)
    assert names(steps) == [
        "stop:backend",
        "run:migrate",
        "request:derived-model",
        "spawn:backend",
        "wait:backend",
        "request:warm-up",
        "write:state",
    ]
    (create,) = [s for s in steps if s.name == "derived-model"]
    assert json.loads(create.body)["from"] == "mistral-small3.2:24b"
    assert names(stack.plan_up(settle(current, other), present, other)) == []  # the same base again: nothing


def test_downloads_that_are_present_are_not_planned(tmp_path, current):
    present = facts(tmp_path, ollama_up=True, ollama_models=frozenset({TAG}), hf=ALL_HF)
    assert not [s for s in stack.plan_up(current, present, args()) if s.is_download]
    only_tag = facts(tmp_path, ollama_up=True, ollama_models=frozenset({TAG}))
    assert [s.name for s in stack.plan_up(current, only_tag, args()) if s.is_download] == [
        "download-embedding",
        "download-reranker",
    ]


def test_a_download_names_its_host_its_size_and_its_pinned_revision(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, ollama_up=True), args())
    by_name = {s.name: s for s in steps}
    embedding = by_name["download-embedding"]
    assert embedding.host == "huggingface.co" and embedding.gb == pytest.approx(1.2)
    assert embedding.argv[-4:] == (
        "download",
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "--revision",
        stack.PINNED["embedding"].revision,
    )
    pull = by_name[f"pull-{TAG}"]
    assert pull.host == "registry.ollama.ai" and pull.gb == 2.5 and pull.argv == ("ollama", "pull", TAG)
    assert "huggingface.co" in embedding.text and "registry.ollama.ai" in pull.text
    listing = stack.confirm_text(steps) or ""
    assert "total about 6 GB" in listing and TAG in listing
    assert stack.confirm_text([s for s in steps if not s.is_download]) is None


def test_ollama_must_be_running_and_says_how_to_start_it(tmp_path, current):
    with pytest.raises(stack.StackError, match=r"Ollama is not running at http://127.0.0.1:11434.*ollama serve"):
        stack.plan_up(current, facts(tmp_path), args())
    # a card with no Ollama model needs no Ollama
    steps = stack.plan_up(current, facts(tmp_path, XL), args(XL, llm_url="http://gpu.lan:8000/v1"))
    assert "derived-model" not in [s.name for s in steps]


def test_docker_must_be_running(tmp_path, current):
    with pytest.raises(stack.StackError, match="Docker is not running"):
        stack.plan_up(current, facts(tmp_path, docker=False, ollama_up=True), args())


def test_the_plan_does_not_touch_the_machine(tmp_path, current):
    before = sorted(p for p in tmp_path.rglob("*"))
    one = stack.plan_up(current, facts(tmp_path, ollama_up=True), args())
    assert one == stack.plan_up(current, facts(tmp_path, ollama_up=True), args())
    assert sorted(p for p in tmp_path.rglob("*")) == before


def test_each_process_gets_its_own_secret_folder_and_the_six_folders(tmp_path, current):
    steps = {s.name: s for s in stack.plan_up(current, facts(tmp_path, ollama_up=True), args()) if s.kind != "wait"}
    assert steps["migrate"].env["RAG_SECRETS_DIR"] == str(current.dir / "secrets" / "migrate")
    index = steps["index-build"]
    assert index.env["RAG_SECRETS_DIR"] == str(current.dir / "secrets" / "indexer")
    assert index.argv == ("uv", "run", "--frozen", "--package", "indexer", "indexer", "build")
    assert {k for k in index.env if k.startswith("RAG__PATHS__")} == {
        f"RAG__PATHS__{k}"
        for k in ("CRAWL_DIR", "PDF_DIR", "EXTRACTED_DIR", "STATE_DIR", "MANIFEST_DIR", "SKIPPED_PDFS_LOG")
    }
    assert index.env["RAG_OVERLAY"] == str(current.dir / "overlay.yaml")
    assert json.loads(steps["overlay"].text)["connectors"]["gesetze"]["dir"] == f"{tmp_path}/profiles/example/documents"
    backend = steps["backend"]
    assert backend.env["RAG_SECRETS_DIR"] == str(current.dir / "secrets" / "backend")
    assert backend.env["RAG__LLM_MODEL"] == DERIVED and backend.env["RAG_CARDS_DIR"] == f"{tmp_path}/models/cards"
    assert backend.env["GUNICORN_BIND"] == "127.0.0.1:8700"


def test_the_model_servers_run_loopback_only_from_their_own_projects_offline(tmp_path, current):
    steps = {s.name: s for s in stack.plan_up(current, facts(tmp_path, ollama_up=True), args()) if s.kind == "spawn"}
    for name, port in (("embedding", "8701"), ("reranker", "8702")):
        step = steps[name]
        assert step.cwd == f"{tmp_path}/model_servers/{name}" and step.path == f"{current.dir}/pids/{name}.pid"
        assert step.argv[-4:] == ("--host", "127.0.0.1", "--port", port)
        assert step.env["HF_HUB_OFFLINE"] == "1"
    assert steps["embedding"].env["EMBEDDING_REVISION"] == stack.PINNED["embedding"].revision
    assert steps["reranker"].env["RERANKER_REVISION"] == stack.PINNED["reranker"].revision


def test_postgres_comes_from_the_compose_files_on_the_loopback_port(tmp_path, current):
    steps = {s.name: s for s in stack.plan_up(current, facts(tmp_path, ollama_up=True), args())}
    argv = steps["postgres"].argv
    assert argv[:6] == ("docker", "compose", "-p", "rag-tuda", "--env-file", f"{current.dir}/compose.env")
    assert f"{tmp_path}/deploy/compose.native.yaml" in argv and argv[-4:] == ("up", "-d", "--wait", "pgvector")
    env = dict(line.split("=", 1) for line in steps["compose-env"].text.splitlines())
    assert (
        env["RAG_ROLES_DIR"] == str(current.dir / "secrets" / "roles") and env["BACKEND_REFRESH_TOKEN"] == current.token
    )


def test_no_secret_is_on_a_command_line_or_in_a_process_environment(tmp_path, current):
    secret_values = {current.token, *current.passwords.values()}
    for step in stack.plan_up(current, facts(tmp_path, ollama_up=True), args()):
        if step.kind == "write":
            continue
        shown = " ".join([*step.argv, *step.env.values(), step.url, step.body])
        assert not any(value in shown for value in secret_values), step.name


def test_a_second_card_restarts_the_backend_and_keeps_the_rest(tmp_path, current):
    first = args(S4B)
    second = args(M35)
    steps = stack.plan_up(settle(current, first), settled(tmp_path, M35, ollama_derived={}), second)
    order = names(steps)
    assert order[0] == "stop:backend" and "spawn:backend" in order and "request:derived-model" in order
    assert "spawn:embedding" not in order and "run:postgres" not in order


def test_a_model_override_makes_the_derived_model_from_it_and_pulls_it(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, ollama_up=True), args(model="mistral-small3.2:24b"))
    assert "pull-mistral-small3.2:24b" in [s.name for s in steps]
    (create,) = [s for s in steps if s.name == "derived-model"]
    assert json.loads(create.body)["from"] == "mistral-small3.2:24b"


def test_a_foreign_server_gets_no_ollama_steps(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path), args(llm_url="http://gpu.lan:8080/v1"))
    assert not [s for s in steps if s.name.startswith(("pull-", "derived"))]
    backend = next(s for s in steps if s.name == "backend" and s.kind == "spawn")
    assert backend.env["RAG__LLM_URL"] == "http://gpu.lan:8080/v1" and "RAG__LLM_MODEL" not in backend.env


def test_a_crawling_profile_has_no_folder_source_and_so_no_overlay_file(tmp_path, current):
    assert stack.FOLDER_SOURCES["tu-informatik"] == {}
    for profile, written in (("example", True), ("tu-informatik", False)):
        steps = stack.plan_up(current, facts(tmp_path, ollama_up=True, crawl_contact=True), args(profile=profile))
        assert any(s.name == "overlay" for s in steps) is written, profile
        index = next(s for s in steps if s.name == "index-build")
        assert ("RAG_OVERLAY" in index.env) is written, profile


def test_the_crawl_contact_hint_is_printed_when_a_crawling_profile_has_none(tmp_path, current):
    hint = lambda steps: [s.text for s in steps if s.kind == "say" and "RAG__CRAWL__CONTACT" in s.text]  # noqa: E731
    assert not hint(stack.plan_up(current, facts(tmp_path, ollama_up=True), args(profile="example")))
    assert hint(stack.plan_up(current, facts(tmp_path, ollama_up=True), args(profile="tu-informatik")))
    assert not hint(
        stack.plan_up(current, facts(tmp_path, ollama_up=True, crawl_contact=True), args(profile="tu-informatik"))
    )


def test_both_index_builds_may_inherit_the_crawl_contact_and_hold_it_in_no_plan(tmp_path, current):
    for mode in ("native", "compose"):
        steps = stack.plan_up(current, facts(tmp_path, ollama_up=True, hf=ALL_HF), args(mode=mode))
        index = next(s for s in steps if s.name == "index-build")
        assert index.inherit == ("RAG__CRAWL__CONTACT",), mode
        assert not [s for s in steps if "RAG__CRAWL__CONTACT" in (s.env.keys() | {s.text if s.kind == "write" else ""})]


# --- compose ---------------------------------------------------------------------------------------------------------------


def test_compose_builds_then_migrates_then_ups(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, hf=ALL_HF), args(mode="compose"))
    assert names(steps) == [
        "write:indexer-overlay",
        "write:compose-env",
        "run:build",
        "run:postgres",
        "run:migrate",
        "run:ollama",
        f"run:pull-{TAG}",
        "write:state-models",
        "run:index-build",
        "run:up",
        "run:warm-up",
        "write:state",
    ]
    by_name = {s.name: s for s in steps}
    assert by_name["build"].argv[-1] == "build" and "--profile" in by_name["migrate"].argv
    assert by_name["index-build"].argv[-6:] == ("--profile", "build", "run", "--rm", "indexer", "build")
    assert by_name[f"pull-{TAG}"].argv[-6:] == ("exec", "-T", "ollama", "ollama", "pull", TAG)


def test_compose_warms_the_answer_model_from_the_backend_container(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, hf=ALL_HF), args(mode="compose"))
    warm = next(s for s in steps if s.name == "warm-up")
    assert warm.argv[-6:-1] == ("exec", "-T", "backend", "python", "-c")
    code = warm.argv[-1]
    assert "http://ollama:11434/v1/chat/completions" in code and TAG in code and "max_tokens" in code
    assert names(steps).index("run:warm-up") > names(steps).index("run:up")
    compile(code, "warm-up", "exec")  # the one-liner the container runs is valid Python


def test_compose_takes_the_profile_from_the_env_file_and_writes_the_indexers_overlay(tmp_path, current):
    steps = {
        s.name: s for s in stack.plan_up(current, facts(tmp_path, hf=ALL_HF), args(mode="compose", profile="example"))
    }
    env = dict(line.split("=", 1) for line in steps["compose-env"].text.splitlines())
    assert env["RAG_PROFILE"] == "example" and env["RAG_INDEXER_OVERLAY"] == f"{current.dir}/indexer-overlay.yaml"
    assert steps["indexer-overlay"].path == env["RAG_INDEXER_OVERLAY"]
    assert json.loads(steps["indexer-overlay"].text) == {
        "connectors": {"gesetze": {"dir": "/app/profiles/example/documents"}}
    }
    other = {s.name: s for s in stack.plan_up(current, facts(tmp_path, hf=ALL_HF), args(mode="compose", profile="tu"))}
    assert "RAG_PROFILE=tu" in other["compose-env"].text and json.loads(other["indexer-overlay"].text) == {}


def test_compose_builds_again_when_the_profile_changed(tmp_path, current):
    example = args(mode="compose")
    images = facts(tmp_path, hf=ALL_HF, compose_images=True)
    assert "run:build" not in names(stack.plan_up(settle(current, example), images, example))
    other = args(mode="compose", profile="tu")
    assert "run:build" in names(stack.plan_up(settle(current, example), images, other))


def test_compose_writes_the_card_and_the_window_for_the_env_file(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, M35, hf=ALL_HF), args(M35, mode="compose"))
    env = dict(line.split("=", 1) for line in next(s.text for s in steps if s.name == "compose-env").splitlines())
    assert (env["RAG__CARD"], env["OLLAMA_CONTEXT_LENGTH"]) == ("m-qwen3.6-35b-a3b", "16000")
    assert next(s for s in steps if s.name == "build").argv[:6] == (
        "docker",
        "compose",
        "-p",
        "rag-tuda",
        "--env-file",
        f"{current.dir}/compose.env",
    )


def test_compose_overlays_follow_the_gpu_and_the_card(tmp_path, current):
    plain = next(
        s for s in stack.plan_up(current, facts(tmp_path, hf=ALL_HF), args(mode="compose")) if s.name == "build"
    )
    assert [a for a in plain.argv if a.endswith(".yaml")] == [f"{tmp_path}/deploy/compose.yaml"]
    gpu = facts(tmp_path, XL, hf=ALL_HF, nvidia=True, compose_images=True)
    argv = next(s for s in stack.plan_up(current, gpu, args(XL, mode="compose")) if s.name == "up").argv
    files = [a for a in argv if a.endswith(".yaml")]
    assert [Path(f).name for f in files] == ["compose.yaml", "compose.nvidia.yaml", "compose.vllm.yaml"]  # vllm last
    assert "pull" not in " ".join(names(stack.plan_up(current, gpu, args(XL, mode="compose"))))


def test_compose_up_again_changes_nothing(tmp_path, current):
    a = args(mode="compose")
    first = stack.plan_up(current, facts(tmp_path, hf=ALL_HF), a)
    recorded = json.loads(next(s.text for s in first if s.name == "state"))
    assert recorded["compose_models"] == [TAG]
    again = state.State(current.dir, current.token, current.passwords, recorded)
    done = facts(tmp_path, hf=ALL_HF, ready=frozenset({"backend"}), compose_images=True, index_built=True)
    steps = stack.plan_up(again, done, a)
    assert [s for s in steps if s.kind not in ("say", "write")] == []
    assert "state" not in [s.name for s in steps]  # the writes that remain are of files that are the same already


def test_compose_skips_what_is_present(tmp_path, current):
    present = facts(tmp_path, hf=ALL_HF, compose_images=True, index_built=True, compose_ollama_models=frozenset({TAG}))
    assert names(stack.plan_up(current, present, args(mode="compose")))[2:] == [
        "run:postgres",
        "run:migrate",
        "run:ollama",
        "run:up",
        "run:warm-up",
        "write:state",
    ]


def test_compose_does_not_pull_again_after_down_because_the_state_file_remembers(tmp_path, current):
    a = args(mode="compose")
    remembered = state.State(current.dir, current.token, current.passwords, {**current.data, "compose_models": [TAG]})
    gone = facts(tmp_path, hf=ALL_HF, compose_images=True, index_built=True)  # the container is down: no models listed
    assert not [s for s in stack.plan_up(remembered, gone, a) if s.name.startswith("pull-")]
    assert [s.name for s in stack.plan_up(current, gone, a) if s.name.startswith("pull-")] == [f"pull-{TAG}"]


def test_compose_rejects_what_it_cannot_do(tmp_path, current):
    with pytest.raises(ValueError, match="native"):
        stack.plan_up(current, facts(tmp_path), args(mode="compose", llm_url="http://x/v1"))
    with pytest.raises(ValueError, match="hosted"):
        stack.plan_up(current, facts(tmp_path, GROQ), args(GROQ, mode="compose"))


def test_a_server_the_user_runs_is_noted_as_not_having_its_window_set(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path), args(llm_url="http://gpu.lan:8080/v1"))
    (note,) = [s.text for s in steps if s.kind == "say" and "num_ctx" in s.text]
    assert "http://gpu.lan:8080/v1" in note and "8000" in note
    assert not [s for s in stack.plan_up(current, facts(tmp_path, ollama_up=True), args()) if "num_ctx" in s.text]


def test_a_changed_profile_builds_the_index_again(tmp_path, current):
    for mode in ("native", "compose"):
        before = args(mode=mode)
        extra = {"compose_images": True} if mode == "compose" else {}
        done = settled(tmp_path, **extra) if mode == "native" else facts(tmp_path, hf=ALL_HF, index_built=True, **extra)
        assert "run:index-build" not in names(stack.plan_up(settle(current, before), done, before))
        after = args(mode=mode, profile="tu-informatik")
        assert "run:index-build" in names(stack.plan_up(settle(current, before), done, after))


def test_compose_builds_the_indexer_and_evals_images_with_the_others(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, hf=ALL_HF), args(mode="compose"))
    (build,) = [s for s in steps if s.name == "build"]
    assert build.argv[-5:] == ("--profile", "build", "--profile", "evals", "build")


def test_the_list_of_a_running_compose_ollama_beats_the_recorded_models(tmp_path, current):
    a = args(mode="compose")
    remembered = state.State(current.dir, current.token, current.passwords, {**current.data, "compose_models": [TAG]})
    removed = facts(tmp_path, hf=ALL_HF, compose_images=True, index_built=True, compose_ollama_up=True)
    assert [s.name for s in stack.plan_up(remembered, removed, a) if s.name.startswith("pull-")] == [f"pull-{TAG}"]


def test_a_model_server_without_its_environment_says_the_first_start_installs_it(tmp_path, current):
    steps = stack.plan_up(current, facts(tmp_path, ollama_up=True, no_venv=frozenset({"embedding"})), args())
    text = {s.name: s.text for s in steps if s.kind == "spawn"}
    assert "installs its Python environment" in text["embedding"]
    assert "installs" not in text["reranker"]
