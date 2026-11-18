"""The executor, the facts and the pins behind plan_up, and the `up` and `down` commands, with a fake machine."""

import json
import re
import stat
import sys
from pathlib import Path

import pytest
import yaml

from rag_tuda import cards, classes, cli, doctor, state, stack

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

    def run(self, argv, cwd, env, timeout_s=0):
        self.ran.append((argv, cwd, self.child_env(env)))
        return self.codes.get(argv[0], 0)

    def popen(self, argv, cwd, env, log):
        self.popened.append((argv, cwd, self.child_env(env)))
        self.live.add(4242)
        return 4242

    def http(self, method, url, body, timeout_s):
        self.requests.append((method, url, body))
        answer = self.answers.get(url, self.default)
        return answer() if callable(answer) else answer

    def alive(self, pid):
        return pid in self.live


def test_the_pins_are_the_ones_the_profile_and_the_images_pin():
    profile = yaml.safe_load((ROOT / "profiles" / "example" / "profile.yaml").read_text(encoding="utf-8"))["retrieval"]
    embedding, reranker = stack.PINNED["embedding"], stack.PINNED["reranker"]
    assert (embedding.repo, embedding.revision) == (profile["embedding"]["model"], profile["embedding"]["revision"])
    assert (reranker.repo, reranker.revision) == (profile["rerank"]["model"], profile["rerank"]["revision"])
    for name, pin in stack.PINNED.items():
        dockerfile = (ROOT / "model_servers" / name / "Dockerfile").read_text(encoding="utf-8")
        baked = dict(re.findall(r"^ENV (\w+)=(.+)$", dockerfile, re.M))
        for key, value in pin.env.items():
            assert baked[key].strip("'") == value, key
    assert {m.name for m in classes.FIXED_MODELS} >= {embedding.repo, reranker.repo}


def test_every_model_server_module_is_the_dockerfiles():
    for name, module in stack.MODULES.items():
        assert f'"{module}"' in (ROOT / "model_servers" / name / "Dockerfile").read_text(encoding="utf-8")


def test_a_child_does_not_inherit_rag_variables_the_settings_loader_would_refuse():
    system = stack.System(
        {"PATH": "/bin", "RAG__CRAWL__CONTACT": "me@example.org", "RAG_PROFILE": "x", "RAG_DEVICE": "mps"}
    )
    assert system.child_env({"RAG__CARD": "c"}) == {"PATH": "/bin", "RAG_DEVICE": "mps", "RAG__CARD": "c"}


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
    assert (current.dir / "pids" / "backend.pid").read_text() == "4242\nuv\n"
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


def test_the_hugging_face_cache_follows_the_environment(tmp_path):
    assert stack.hf_cache({"HF_HUB_CACHE": "/a"}) == Path("/a")
    assert stack.hf_cache({"HF_HOME": "/h"}) == Path("/h/hub")
    assert stack.hf_cache({}) == Path.home() / ".cache" / "huggingface" / "hub"


def machine(scenario):
    return scenario("m2_16")


def test_gather_reads_the_machine_and_plans_nothing_for_a_settled_one(tmp_path, scenario):
    hardware = machine(scenario)
    hardware.docker.ok = True
    current = state.ensure(tmp_path)
    cache = tmp_path / "hf" / "hub"
    for pin in stack.PINNED.values():
        (cache / f"models--{pin.repo.replace('/', '--')}" / "snapshots" / pin.revision).mkdir(parents=True)
        (cache / f"models--{pin.repo.replace('/', '--')}" / "snapshots" / pin.revision / "config.json").write_text("{}")
    (current.dir / "state").mkdir()
    (current.dir / "state" / "corpus_build_stats.json").write_text("{}")
    live = FakeSystem(environ={"HF_HOME": str(tmp_path / "hf"), "RAG__CRAWL__CONTACT": "me@example.org"})
    for name in ("embedding", "reranker", "backend"):
        (current.dir / "pids" / f"{name}.pid").write_text("4242\nuv\n")
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
    live.output = lambda argv: "abc\n" if "pgvector" in argv else None
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


def test_down_through_the_command_is_safe_twice(repo, monkeypatch, capsys):
    system = FakeSystem()
    monkeypatch.setattr(stack, "System", lambda: system)
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
