"""The executor, the facts and the pins behind plan_up, and the `up` and `down` commands, with a fake machine."""

import json
import os
import stat
import sys
import tomllib
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from rag_tuda import cards, cli, doctor, state, stack

ROOT = Path(__file__).resolve().parents[2]
S4B = cards.named("s-qwen3-4b-instruct-2507")
posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")


class FakeSystem(stack.System):
    """Records what would run; HTTP answers from a table of status codes (default 200)."""

    def __init__(self, environ=None, answers=None, codes=None, default=(200, "{}")):
        super().__init__(environ if environ is not None else {})
        self.answers, self.codes, self.default = answers or {}, codes or {}, default
        self.ran, self.popened, self.requests, self.live, self.now = [], [], [], set(), 0.0
        self.sleep = lambda seconds: setattr(self, "now", self.now + seconds)
        self.clock = lambda: self.now

    def run(self, argv, cwd, env, timeout_s=0, inherit=()):
        self.ran.append((argv, cwd, self.child_env(env, inherit)))
        return self.codes.get(argv[0], 0)

    def popen(self, argv, cwd, env, log, inherit=()):
        self.popened.append((argv, cwd, self.child_env(env, inherit)))
        self.live.add(4242)
        return 4242

    def start_time(self, pid):
        return "Wed 18 Nov 12:00:00 2026"

    def http(self, method, url, body, timeout_s):
        self.requests.append((method, url, body))
        answer = self.answers.get(url, self.default)
        return answer() if callable(answer) else answer

    def alive(self, pid):
        return pid in self.live


def test_every_model_server_module_is_the_dockerfiles():
    for name, module in stack.MODULES.items():
        assert f'"{module}"' in (ROOT / "model_servers" / name / "Dockerfile").read_text(encoding="utf-8")


def test_the_download_tool_is_the_huggingface_hub_the_model_servers_lock():
    for name in stack.MODULES:
        lock = tomllib.loads((ROOT / "model_servers" / name / "uv.lock").read_text(encoding="utf-8"))
        locked = [p["version"] for p in lock["package"] if p["name"] == "huggingface-hub"]
        assert locked == [stack.HF_HUB_VERSION], name


def test_a_child_does_not_inherit_rag_variables_the_settings_loader_would_refuse():
    system = stack.System(
        {"PATH": "/bin", "RAG__CRAWL__CONTACT": "me@example.org", "RAG_PROFILE": "x", "RAG_DEVICE": "mps"}
    )
    assert system.child_env({"RAG__CARD": "c"}) == {"PATH": "/bin", "RAG_DEVICE": "mps", "RAG__CARD": "c"}


def test_the_provider_key_reaches_only_the_process_that_names_it():
    system = stack.System({"PATH": "/bin", "GROQ_API_KEY": "gsk-secret", "OPENAI_API_KEY": "sk-x"})
    assert "GROQ_API_KEY" not in system.child_env({"A": "1"})
    assert system.child_env({"A": "1"}, ("GROQ_API_KEY",))["GROQ_API_KEY"] == "gsk-secret"
    assert "GROQ_API_KEY" not in stack.System({"PATH": "/bin"}).child_env({}, ("GROQ_API_KEY",))  # nothing to inherit


def test_a_spawn_hands_its_named_variables_to_the_process_and_a_run_none(tmp_path):
    current = state.ensure(tmp_path)
    system = FakeSystem(environ={"GROQ_API_KEY": "gsk-secret"})
    spawn = stack.Step(
        "spawn",
        name="backend",
        argv=("uv",),
        path=f"{current.dir}/pids/b.pid",
        log=f"{current.dir}/logs/b.log",
        inherit=("GROQ_API_KEY",),
    )
    stack.execute([spawn, stack.Step("run", name="x", argv=("uv",))], system, lambda line: None)
    assert system.popened[0][2]["GROQ_API_KEY"] == "gsk-secret" and "GROQ_API_KEY" not in system.ran[0][2]


def test_write_replaces_the_content_only_when_it_differs(tmp_path):
    path = tmp_path / ".rag" / "compose.env"
    step = stack.Step("write", path=str(path), text="A=1\n")
    stack.execute([step], FakeSystem(), lambda line: None)
    assert path.read_text() == "A=1\n"
    path.chmod(0o644)
    mtime = path.stat().st_mtime_ns
    stack.execute([step], FakeSystem(), lambda line: None)
    assert path.stat().st_mtime_ns == mtime  # unchanged content is not written again
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700


@posix
def test_spawn_writes_the_pid_file_and_the_log_with_private_modes(tmp_path):
    current = state.ensure(tmp_path)
    system = FakeSystem()
    step = stack.Step(
        "spawn",
        name="backend",
        argv=("uv", "run"),
        cwd=str(tmp_path),
        env={"A": "1"},
        path=f"{current.dir}/pids/backend.pid",
        log=f"{current.dir}/logs/backend.log",
    )
    stack.execute([step], system, lambda line: None)
    assert system.popened[0][0] == ("uv", "run") and system.popened[0][2]["A"] == "1"
    assert (current.dir / "pids" / "backend.pid").read_text() == "4242\nWed 18 Nov 12:00:00 2026\n"
    for path in (current.dir / "pids" / "backend.pid", current.dir / "logs" / "backend.log"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def wait_step(tmp_path, **kwargs):
    return stack.Step(
        "wait",
        name="embedding",
        url="http://127.0.0.1:8701/readyz",
        path=str(tmp_path / "e.pid"),
        log="e.log",
        timeout_s=kwargs.get("timeout_s", 5),
    )


def test_wait_returns_when_the_service_answers(tmp_path):
    codes = iter([(0, "refused"), (503, ""), (200, "{}")])
    system = FakeSystem(answers={"http://127.0.0.1:8701/readyz": lambda: next(codes)})
    stack.execute([wait_step(tmp_path)], system, lambda line: None)
    assert system.now == 2  # one second between probes


def test_wait_gives_up_after_its_timeout_and_names_the_log(tmp_path):
    system = FakeSystem(answers={"http://127.0.0.1:8701/readyz": (503, "")})
    with pytest.raises(stack.StackError, match=r"embedding was not ready after 5 s; see e\.log"):
        stack.execute([wait_step(tmp_path)], system, lambda line: None)


def test_wait_stops_early_when_the_process_died(tmp_path):
    (tmp_path / "e.pid").write_text("4242\nuv\n")
    system = FakeSystem(answers={"http://127.0.0.1:8701/readyz": (0, "refused")})
    with pytest.raises(stack.StackError, match=r"embedding stopped while starting; see e\.log"):
        stack.execute([wait_step(tmp_path)], system, lambda line: None)
    assert system.now == 0


def test_a_failed_download_fails_loudly_and_names_the_host():
    step = stack.Step("run", name="download-embedding", argv=("uv", "run"), host="huggingface.co")
    with pytest.raises(
        stack.StackError, match=r"the download from huggingface\.co failed \(exit 1\).*may block huggingface\.co"
    ):
        stack.execute([step], FakeSystem(codes={"uv": 1}), lambda line: None)
    with pytest.raises(stack.StackError, match=r"migrate: failed \(exit 3\)"):
        stack.execute(
            [stack.Step("run", name="migrate", argv=("uv", "run"))], FakeSystem(codes={"uv": 3}), lambda line: None
        )


def test_a_failed_request_names_the_url_and_status():
    step = stack.Step("request", name="derived-model", url="http://127.0.0.1:11434/api/create", body="{}")
    with pytest.raises(stack.StackError, match=r"derived-model: http://127.0.0.1:11434/api/create answered 500"):
        stack.execute([step], FakeSystem(answers={step.url: (500, "boom")}), lambda line: None)
    with pytest.raises(stack.StackError, match="answered nothing"):
        stack.execute([step], FakeSystem(answers={step.url: (0, "refused")}), lambda line: None)


def test_the_steps_run_in_order_and_the_first_failure_stops_the_rest():
    system = FakeSystem(codes={"second": 1})
    steps = [stack.Step("run", name=n, argv=(n,)) for n in ("first", "second", "third")]
    with pytest.raises(stack.StackError):
        stack.execute(steps, system, lambda line: None)
    assert [argv[0] for argv, _, _ in system.ran] == ["first", "second"]


def test_ollama_tags_and_parameters_are_parsed():
    assert stack.parse_tags('{"models": [{"name": "a:1"}, {"name": "rag-x:latest"}]}') == {"a:1", "rag-x:latest"}
    assert stack.parse_tags("garbage") == frozenset()
    shown = json.dumps(
        {"parameters": 'num_ctx                        8000\ntemperature                    0.1\nstop   "<|x|>"'}
    )
    assert stack.parse_parameters(shown) == {"num_ctx": "8000", "temperature": "0.1", "stop": '"<|x|>"'}
    assert stack.parse_parameters("{}") == {}
    assert stack.has_model(frozenset({"rag-x:latest"}), "rag-x") and not stack.has_model(frozenset({"a:1"}), "a")
    assert stack.same_parameters({"num_ctx": "8000", "temperature": "0.10"}, {"num_ctx": 8000, "temperature": 0.1})
    assert not stack.same_parameters({"num_ctx": "4096"}, {"num_ctx": 8000}) and not stack.same_parameters(
        {}, {"num_ctx": 1}
    )


def test_a_list_valued_parameter_is_compared_by_its_lines():
    shown = json.dumps({"parameters": 'num_ctx 8000\nstop   "<|a|>"\nstop   "<|b|>"'})
    have = stack.parse_parameters(shown)
    assert stack.same_parameters(have, {"num_ctx": 8000, "stop": ["<|b|>", "<|a|>"]})
    assert not stack.same_parameters(have, {"stop": ["<|a|>"]})


def test_the_hugging_face_cache_follows_the_environment(tmp_path):
    assert stack.hf_cache({"HF_HUB_CACHE": "/a"}) == Path("/a")
    assert stack.hf_cache({"HF_HOME": "/h"}) == Path("/h/hub")
    assert stack.hf_cache({}) == Path.home() / ".cache" / "huggingface" / "hub"


def test_compose_mounts_the_home_of_the_cache_it_checks(tmp_path, scenario):
    assert stack.hf_home({"HF_HOME": "/h"}) == Path("/h")
    assert stack.hf_home({"HF_HUB_CACHE": "/h/hub"}) == Path("/h")
    with pytest.raises(stack.StackError, match="HF_HUB_CACHE .*must end in /hub"):
        stack.hf_home({"HF_HUB_CACHE": "/elsewhere"})
    hardware = machine(scenario)
    hardware.docker.ok = True
    current = state.ensure(tmp_path)
    system = FakeSystem(environ={"HF_HOME": str(tmp_path / "hf")}, default=(0, "refused"))
    system.output = docker_output
    found = stack.gather(tmp_path, current, S4B, hardware, stack.Args(S4B["name"], mode="compose"), system)
    assert found.hf_home == str(tmp_path / "hf")
    native = stack.gather(tmp_path, current, S4B, hardware, stack.Args(S4B["name"]), system)
    assert native.hf_home == ""


def snapshot_of(cache, pin, weights=True):
    folder = cache / f"models--{pin.repo.replace('/', '--')}" / "snapshots" / pin.revision
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text("{}")
    if weights:
        (folder / "model.safetensors").write_text("w")
    return folder


def docker_output(argv):
    """What Docker answers on a machine with a running Postgres and its volume made long ago."""
    if "volume" in argv:
        return "2020-01-01T00:00:00Z\n"
    return "abc\n" if "pgvector" in argv else None


def machine(scenario):
    return scenario("m2_16")


def test_gather_reads_the_machine_and_plans_nothing_for_a_settled_one(tmp_path, scenario):
    hardware = machine(scenario)
    hardware.docker.ok = True
    current = state.ensure(tmp_path)
    cache = tmp_path / "hf" / "hub"
    for pin in stack.PINNED.values():
        snapshot_of(cache, pin)
    (current.dir / "state").mkdir()
    (current.dir / "state" / "corpus_build_stats.json").write_text('{"profile": "example"}')
    live = FakeSystem(environ={"HF_HOME": str(tmp_path / "hf"), "RAG__CRAWL__CONTACT": "me@example.org"})
    for name in ("embedding", "reranker", "backend"):
        (current.dir / "pids" / f"{name}.pid").write_text("4242\nWed 18 Nov 12:00:00 2026\n")
    live.live.add(4242)
    derived = "rag-s-qwen3-4b-instruct-2507"
    live.answers = {
        "http://127.0.0.1:11434/api/version": (200, '{"version": "0.40.12"}'),
        "http://127.0.0.1:11434/api/tags": (
            200,
            json.dumps({"models": [{"name": "qwen3:4b-instruct-2507-q4_K_M"}, {"name": f"{derived}:latest"}]}),
        ),
        "http://127.0.0.1:11434/api/show": (
            200,
            json.dumps({"parameters": "num_ctx 8000\ntemperature 0.1\ntop_p 0.9"}),
        ),
    }
    live.output = docker_output
    a = stack.Args(S4B["name"])
    found = stack.gather(tmp_path, current, S4B, hardware, a, live)
    assert found.ollama_up and found.hf == {pin.key for pin in stack.PINNED.values()}
    assert found.alive == found.ready == {"embedding", "reranker", "backend"}
    assert found.index_built and found.postgres and found.crawl_contact
    assert found.ollama_derived[derived]["num_ctx"] == "8000"
    settled = state.State(
        current.dir, current.token, current.passwords, {**current.data, **stack._signature(a), "first_run_done": True}
    )
    assert [s for s in stack.plan_up(settled, found, a) if s.kind != "say"] == []


def test_compose_after_down_plans_no_pull_through_the_real_gather_path(tmp_path, scenario):
    """The container is gone, `ollama list` fails, and the state file remembers what the volume holds."""
    hardware = machine(scenario)
    hardware.docker.ok = True
    current = state.ensure(tmp_path)
    a = stack.Args(S4B["name"], "compose")
    cache = tmp_path / "hf" / "hub"
    for pin in stack.PINNED.values():
        snapshot_of(cache, pin)
    system = FakeSystem(environ={"HF_HOME": str(tmp_path / "hf")}, default=(0, "refused"))
    system.output = lambda argv: None  # no container, no images, no models
    before = stack.gather(tmp_path, current, S4B, hardware, a, system)
    assert before.compose_ollama_models == frozenset()
    pulls = lambda facts, st: [s.name for s in stack.plan_up(st, facts, a) if s.name.startswith("pull-")]  # noqa: E731
    assert pulls(before, current) == ["pull-qwen3:4b-instruct-2507-q4_K_M"]  # never pulled: the first up downloads
    pulled = state.State(
        current.dir,
        current.token,
        current.passwords,
        {**current.data, "compose_models": ["qwen3:4b-instruct-2507-q4_K_M"]},
    )
    assert pulls(before, pulled) == []


def test_a_partly_downloaded_snapshot_is_not_present(tmp_path, scenario):
    hardware = machine(scenario)
    hardware.docker.ok = True
    current = state.ensure(tmp_path)
    cache = tmp_path / "hf" / "hub"
    pins = list(stack.PINNED.values())
    snapshot_of(cache, pins[0])
    snapshot_of(cache, pins[1], weights=False)  # the config came, the weights did not
    system = FakeSystem(environ={"HF_HOME": str(tmp_path / "hf")}, default=(0, "refused"))
    system.output = lambda argv: None
    found = stack.gather(tmp_path, current, S4B, hardware, stack.Args(S4B["name"]), system)
    assert found.hf == {pins[0].key}


def test_gather_names_the_model_servers_without_an_environment(tmp_path, scenario):
    hardware = machine(scenario)
    hardware.docker.ok = True
    current = state.ensure(tmp_path)
    (tmp_path / "model_servers" / "reranker" / ".venv").mkdir(parents=True)
    system = FakeSystem(environ={"HF_HOME": str(tmp_path)}, default=(0, "refused"))
    system.output = lambda argv: None
    found = stack.gather(tmp_path, current, S4B, hardware, stack.Args(S4B["name"]), system)
    assert found.no_venv == {"embedding"}


def test_a_pid_file_of_another_boot_does_not_count_as_alive(tmp_path, scenario):
    hardware = machine(scenario)
    hardware.docker.ok = True
    current = state.ensure(tmp_path)
    (current.dir / "pids" / "embedding.pid").write_text("4242\nMon  1 Jan 00:00:00 2024\n")
    system = FakeSystem(environ={"HF_HOME": str(tmp_path)}, default=(0, "refused"))
    system.output = lambda argv: None
    system.live.add(4242)  # some process has the pid now; FakeSystem gives it another start time than the file records
    found = stack.gather(tmp_path, current, S4B, hardware, stack.Args(S4B["name"]), system)
    assert "embedding" not in found.alive


def test_a_settled_hosted_card_plans_nothing_through_gather(tmp_path, scenario):
    hardware = machine(scenario)
    hardware.docker.ok = True
    current = state.ensure(tmp_path)
    groq = cards.named("l-groq-gpt-oss-120b")
    a = stack.Args(groq["name"])
    cache = tmp_path / "hf" / "hub"
    for pin in stack.PINNED.values():
        snapshot_of(cache, pin)
    (current.dir / "state").mkdir()
    (current.dir / "state" / "corpus_build_stats.json").write_text('{"profile": "example"}')
    for name in ("embedding", "reranker", "backend"):
        (current.dir / "pids" / f"{name}.pid").write_text("4242\nWed 18 Nov 12:00:00 2026\n")
    analysis = "rag-l-groq-gpt-oss-120b-analysis"
    system = FakeSystem(environ={"HF_HOME": str(tmp_path / "hf"), "GROQ_API_KEY": "gsk-test"})
    system.live.add(4242)
    system.answers = {
        "http://127.0.0.1:11434/api/tags": (
            200,
            json.dumps({"models": [{"name": "qwen3:4b-instruct-2507-q4_K_M"}, {"name": f"{analysis}:latest"}]}),
        ),
        "http://127.0.0.1:11434/api/show": (
            200,
            json.dumps({"parameters": "num_ctx 8000\ntemperature 0.1\ntop_p 0.9"}),
        ),
    }
    system.output = docker_output
    found = stack.gather(tmp_path, current, groq, hardware, a, system)
    assert found.groq_key and found.ollama_derived[analysis]["num_ctx"] == "8000"
    settled = state.State(
        current.dir, current.token, current.passwords, {**current.data, **stack._signature(a), "first_run_done": True}
    )
    assert [s for s in stack.plan_up(settled, found, a) if s.kind != "say"] == []


def test_gather_on_a_bare_machine_finds_nothing_present(tmp_path, scenario):
    hardware = machine(scenario)
    hardware.docker.ok = True
    current = state.ensure(tmp_path)
    bare = FakeSystem(environ={"HF_HOME": str(tmp_path / "hf")}, default=(0, "refused"))
    bare.output = lambda argv: None
    found = stack.gather(tmp_path, current, S4B, hardware, stack.Args(S4B["name"]), bare)
    assert not (found.ollama_up or found.hf or found.alive or found.ready or found.postgres or found.index_built)


# --- the commands -----------------------------------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path, monkeypatch, scenario):
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy" / "compose.yaml").write_text("name: rag-tuda\n")
    for profile in ("example", "tu-informatik"):
        (tmp_path / "profiles" / profile).mkdir(parents=True)
        (tmp_path / "profiles" / profile / "profile.yaml").write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    hardware = scenario("m2_16")
    hardware.docker.ok = True
    monkeypatch.setattr(doctor, "gather", lambda: hardware)
    return tmp_path


def settled_facts(root, **kwargs):
    present = frozenset(pin.key for pin in stack.PINNED.values())
    values = {"ollama_up": True, "ollama_models": frozenset({S4B["answer"]["model"]}), "hf": present} | kwargs
    return lambda *a, **k: stack.Facts(root=str(root), card=S4B, **values)


def test_up_prints_the_link_when_the_steps_have_run(repo, monkeypatch, capsys):
    system = FakeSystem()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(stack, "gather", settled_facts(repo))
    assert cli.main(["up", "--yes"]) == 0
    out = capsys.readouterr().out
    token = (repo / ".rag" / "token").read_text()
    assert f"ready: http://127.0.0.1:8700/#{token}" in out
    assert out.count(token) == 1 and "BACKEND_REFRESH_TOKEN" not in out  # a written file's content is not printed
    assert (
        "postgres: starting on 127.0.0.1:55432" in out
        and "loading the answer model rag-s-qwen3-4b-instruct-2507" in out
    )
    assert [a[0] for a, _, _ in system.ran][:1] == ["docker"]
    assert (repo / ".rag" / "compose.env").is_file() and json.loads((repo / ".rag" / "state.json").read_text())[
        "card"
    ] == S4B["name"]


def test_up_lists_downloads_and_needs_a_yes(repo, monkeypatch, capsys):
    system = FakeSystem()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(stack, "gather", settled_facts(repo, hf=frozenset(), ollama_models=frozenset()))
    assert cli.main(["up"]) == 1  # no terminal, no --yes
    captured = capsys.readouterr()
    assert "This will download:" in captured.out and "total about 6 GB" in captured.out
    assert "EasyOCR" in captured.out  # a bare `rag up` is the tu-informatik profile
    assert "downloads need your yes" in captured.err and system.ran == []
    assert cli.main(["up", "--yes"]) == 0 and system.ran


def test_up_stops_with_the_failing_steps_message(repo, monkeypatch, capsys):
    monkeypatch.setattr(stack, "System", lambda: FakeSystem(codes={"docker": 1}))
    monkeypatch.setattr(stack, "gather", settled_facts(repo))
    assert cli.main(["up", "--yes"]) == 1
    captured = capsys.readouterr()
    assert "postgres: failed (exit 1)" in captured.err and "ready:" not in captured.out


def test_up_without_ollama_names_the_hint_and_starts_nothing(repo, monkeypatch, capsys):
    system = FakeSystem()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(stack, "gather", settled_facts(repo, ollama_up=False))
    assert cli.main(["up", "--yes"]) == 1
    assert "Ollama is not running" in capsys.readouterr().err and system.ran == [] and system.popened == []


def test_up_wants_the_repository_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["up", "--yes"]) == 1
    assert "root of the repository" in capsys.readouterr().err and not (tmp_path / ".rag").exists()


def test_up_rejects_an_unknown_card_with_its_name(repo, capsys):
    assert cli.main(["up", "--yes", "--card", "no-such-card"]) == 1
    assert "no-such-card" in capsys.readouterr().err


def test_down_never_creates_state_and_says_so(repo, monkeypatch, capsys):
    monkeypatch.setattr(stack, "System", lambda: FakeSystem())
    assert cli.main(["down"]) == 0
    assert "no .rag/ here" in capsys.readouterr().out and not (repo / ".rag").exists()


def test_down_wants_the_repository_root_too(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["down"]) == 1
    assert "root of the repository" in capsys.readouterr().err and not (tmp_path / ".rag").exists()


def test_up_refuses_a_hosted_card_without_the_key_and_starts_nothing(repo, monkeypatch, capsys):
    system = FakeSystem()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(
        stack,
        "gather",
        lambda *a, **k: stack.Facts(root=str(repo), card=cards.named("l-groq-gpt-oss-120b"), ollama_up=True),
    )
    assert cli.main(["up", "--yes", "--card", "l-groq-gpt-oss-120b"]) == 1
    assert "GROQ_API_KEY" in capsys.readouterr().err and system.ran == [] and system.popened == []


def test_down_through_the_command_is_safe_twice(repo, monkeypatch, capsys):
    system = FakeSystem()
    monkeypatch.setattr(stack, "System", lambda: system)
    state.ensure(repo)
    assert cli.main(["down"]) == 0 and cli.main(["down"]) == 0
    assert capsys.readouterr().out.count("backend: not running") == 2


def test_purge_needs_a_yes(repo, monkeypatch, capsys):
    monkeypatch.setattr(stack, "System", lambda: FakeSystem())
    state.ensure(repo)
    (repo / ".rag" / "crawl").mkdir()
    assert cli.main(["down", "--purge"]) == 1 and (repo / ".rag" / "crawl").is_dir()
    assert cli.main(["down", "--purge", "--yes"]) == 0 and not (repo / ".rag" / "crawl").exists()


def test_the_real_system_runs_commands_and_talks_to_the_loopback(monkeypatch):
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/ok" else 503)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")  # a proxy must not be used for the loopback
    system = stack.System()
    try:
        assert system.http("GET", f"{base}/ok", None, 5) == (200, '{"ok": true}')
        assert system.http("GET", f"{base}/no", None, 5)[0] == 503
        assert system.http("POST", f"{base}/echo", '{"a": 1}', 5) == (200, '{"a": 1}')
    finally:
        server.shutdown()
    assert system.http("GET", f"{base}/ok", None, 1)[0] == 0  # nothing listens any more
    assert system.run(("true",), "", {}) == 0 and system.run(("false",), "", {}) == 1
    assert system.output(("echo", "hi")) == "hi\n" and system.output(("false",)) is None
    with pytest.raises(stack.StackError, match="cannot be started"):
        system.run(("no-such-command-here",), "", {})


@posix
def test_a_run_gets_the_crawl_contact_only_when_the_step_names_it_and_the_user_has_it():
    import os

    def seen(system, expected, **inherit):
        """Whether the child's RAG__CRAWL__CONTACT is `expected` ('unset': absent)."""
        test = '[ "${RAG__CRAWL__CONTACT-unset}" = "$0" ]'
        return system.run(("sh", "-c", test, expected), "", {}, **inherit) == 0

    mine = stack.System({"PATH": os.environ["PATH"], "RAG__CRAWL__CONTACT": "me@example.org"})
    assert seen(mine, "unset") and seen(mine, "me@example.org", inherit=stack.CRAWL_CONTACT)
    assert seen(stack.System({"PATH": os.environ["PATH"]}), "unset", inherit=stack.CRAWL_CONTACT)


@posix
def test_the_probes_do_not_get_the_provider_key():
    import os

    system = stack.System({"PATH": os.environ["PATH"], "GROQ_API_KEY": "gsk-secret"})
    assert "GROQ_API_KEY" not in (system.output(("env",)) or "")


def test_up_takes_the_mode_of_the_hardware_unless_told(repo, monkeypatch, scenario):
    seen = []
    monkeypatch.setattr(stack, "System", lambda: FakeSystem())
    monkeypatch.setattr(stack, "gather", settled_facts(repo))
    monkeypatch.setattr(stack, "plan_up", lambda current, facts, a: seen.append(a.mode) or [])
    linux = scenario("rtx4090")
    linux.docker.ok = True
    monkeypatch.setattr(doctor, "gather", lambda: linux)
    assert (
        cli.main(["up", "--card", S4B["name"]]) == 0
        and cli.main(["up", "--card", S4B["name"], "--mode", "native"]) == 0
    )
    assert seen == ["compose", "native"]


def test_purge_removes_crawl_dirs(repo, monkeypatch, capsys):
    monkeypatch.setattr(stack, "System", lambda: FakeSystem())
    current = state.ensure(repo)
    for folder in ("crawl/www.example.org", "crawl/pdfs", "snapshots"):
        (current.dir / folder).mkdir(parents=True)
        (current.dir / folder / "page.json").write_text("{}")
    (current.dir / "state").mkdir()
    (current.dir / "state" / "corpus_build_stats.json").write_text("{}")
    assert cli.main(["down", "--purge", "--yes"]) == 0
    assert not (current.dir / "crawl").exists() and not (current.dir / "snapshots").exists()
    assert (current.dir / "state" / "corpus_build_stats.json").is_file() and (current.dir / "token").is_file()
    assert cli.main(["down", "--purge", "--yes"]) == 0  # nothing left to delete is not an error


def test_up_defaults_to_the_tu_informatik_profile(repo, monkeypatch):
    system = FakeSystem()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(stack, "gather", settled_facts(repo))
    seen = []
    plan_up = stack.plan_up
    monkeypatch.setattr(
        stack, "plan_up", lambda current, facts, up: seen.append(up.profile) or plan_up(current, facts, up)
    )
    assert cli.main(["up", "--yes"]) == 0 and cli.main(["up", "--yes", "--profile", "example"]) == 0
    assert seen == ["tu-informatik", "example"]


def _spy_profile(monkeypatch):
    seen = []
    plan_up = stack.plan_up
    monkeypatch.setattr(
        stack, "plan_up", lambda current, facts, up: (seen.append(up.profile), plan_up(current, facts, up))[1]
    )
    return seen


def _record_profile(repo, profile):
    (repo / ".rag").mkdir(exist_ok=True)
    (repo / ".rag" / "state.json").write_text(json.dumps({"profile": profile, "first_run_done": True}))


@pytest.mark.parametrize(
    ("recorded", "flags", "expected"),
    [
        (None, [], "tu-informatik"),
        ("example", [], "example"),
        ("example", ["--profile", "tu-informatik"], "tu-informatik"),
    ],
    ids=["first run", "recorded example", "explicit flag wins"],
)
def test_up_takes_the_profile_from_the_flag_then_the_recorded_one_then_the_tu_profile(
    repo, monkeypatch, recorded, flags, expected
):
    monkeypatch.setattr(stack, "System", lambda: FakeSystem())
    monkeypatch.setattr(stack, "gather", settled_facts(repo, index_built=True))
    seen = _spy_profile(monkeypatch)
    if recorded:
        _record_profile(repo, recorded)
    assert cli.main(["up", "--yes", *flags]) == 0
    assert seen == [expected]
    assert json.loads((repo / ".rag" / "state.json").read_text())["profile"] == expected


def write_stats(path, profile="example", age_s=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"profile": profile, "parents": 3}))
    if age_s:
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime - age_s))
    return path


def test_the_index_counts_as_built_for_the_profile_whose_stats_and_volume_say_so(tmp_path):
    stats = write_stats(tmp_path / "state" / "corpus_build_stats.json", "tu-informatik")
    volume = "2020-01-01T00:00:00Z"
    assert stack.index_built(stats, "tu-informatik", volume)
    assert not stack.index_built(stats, "example", volume)  # the other profile's index is not this one's
    assert not stack.index_built(tmp_path / "none.json", "example", volume)
    assert not stack.index_built(stats, "tu-informatik", None)  # no volume: the database is gone
    stats.write_text("{}")  # stats written before they named a profile
    assert not stack.index_built(stats, "tu-informatik", volume)
    stats.write_text("{broken")
    assert not stack.index_built(stats, "tu-informatik", volume)


def test_a_volume_made_after_the_stats_means_the_database_was_lost(tmp_path):
    stats = write_stats(tmp_path / "corpus_build_stats.json", age_s=3600)
    written = datetime.fromtimestamp(stats.stat().st_mtime, UTC)
    older, newer = written - timedelta(minutes=5), written + timedelta(minutes=5)
    assert stack.index_built(stats, "example", older.isoformat())
    assert not stack.index_built(stats, "example", newer.isoformat())
    assert not stack.index_built(stats, "example", newer.astimezone(timezone(timedelta(hours=-5))).isoformat())
    assert stack.index_built(stats, "example", "no time")  # an unreadable creation time is trusted


def test_gather_reads_the_index_for_the_profile_asked_for_and_the_volume(tmp_path, scenario):
    hardware = machine(scenario)
    hardware.docker.ok = True
    current = state.ensure(tmp_path)
    write_stats(current.dir / "state" / "corpus_build_stats.json", "tu-informatik")
    system = FakeSystem(environ={"HF_HOME": str(tmp_path)}, default=(0, "refused"))
    asked = []
    system.output = lambda argv: asked.append(argv) or docker_output(argv)

    def built(profile):
        return stack.gather(
            tmp_path, current, S4B, hardware, stack.Args(S4B["name"], profile=profile), system
        ).index_built

    assert built("tu-informatik") and not built("example")
    volume = next(a for a in asked if "volume" in a)
    assert volume == ("docker", "volume", "inspect", "rag-tuda_db-data", "--format", "{{.CreatedAt}}")
    system.output = lambda argv: None if "volume" in argv else docker_output(argv)  # `docker volume rm`, a Colima reset
    assert not built("tu-informatik")


@posix
@pytest.mark.skipif(sys.platform != "win32" and os.geteuid() == 0, reason="root can delete anything")
def test_a_purge_that_leaves_files_behind_names_the_path_and_the_remedy(repo, monkeypatch, capsys):
    """On Linux the compose indexer writes the crawl as root: the user's delete fails and must not report success."""
    monkeypatch.setattr(stack, "System", lambda: FakeSystem())
    current = state.ensure(repo)
    locked = current.dir / "crawl" / "www.example.org"
    locked.mkdir(parents=True)
    (locked / "page.json").write_text("{}")
    locked.chmod(0o500)  # a folder its owner cannot unlink from
    try:
        assert cli.main(["down", "--purge", "--yes"]) == 1
        err = capsys.readouterr().err
        assert str(current.dir / "crawl") in err and f"sudo rm -rf {current.dir / 'crawl'}" in err
        assert (locked / "page.json").exists()
    finally:
        locked.chmod(0o700)
    assert cli.main(["down", "--purge", "--yes"]) == 0 and not (current.dir / "crawl").exists()


def test_up_refuses_a_profile_that_does_not_exist_before_any_step(repo, monkeypatch, capsys):
    """A typo must not stop the running backend or start a crawl: the plan is never made."""
    system = FakeSystem()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(stack, "plan_up", lambda *a: pytest.fail("no plan for an unknown profile"))
    assert cli.main(["up", "--yes", "--profile", "exmaple"]) == 1
    err = capsys.readouterr().err
    assert "'exmaple'" in err and "profiles/exmaple/profile.yaml" in err and "example, tu-informatik" in err
    assert system.ran == [] and system.popened == []
