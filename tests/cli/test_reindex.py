"""reindex and rollback: an index job on the running stack, then a backend restart (it caches the index handles and the
full-text term statistics per load)."""

import json

import pytest

from rag_tuda import cards, classes, cli, doctor, stack, state

S4B = cards.named("s-qwen3-4b-instruct-2507")
GROQ = cards.named("l-groq-gpt-oss-120b")
ALL_HF = frozenset(pin.key for pin in stack.PINNED.values())
RUNNING = frozenset({"embedding", "reranker", "backend"})


def first_run(tmp_path, a: stack.Args) -> state.State:
    current = state.ensure(tmp_path)
    return state.State(
        current.dir, current.token, current.passwords, {**current.data, **stack._signature(a), "first_run_done": True}
    )


def running(tmp_path, card=S4B, **kwargs) -> stack.Facts:
    values = {"docker": True, "hf": ALL_HF, "alive": RUNNING, "ready": RUNNING, "postgres": True, "index_built": True}
    return stack.Facts(root=str(tmp_path), card=card, **{**values, **kwargs})


def names(steps) -> list[str]:
    return [f"{s.kind}:{s.name}" for s in steps if s.kind != "say"]


def plan(tmp_path, a=None, facts=None, **kwargs):
    a = a or stack.Args(S4B["name"])
    return stack.plan_reindex(first_run(tmp_path, a), facts or running(tmp_path), a, **kwargs)


def test_reindex_restarts_backend(tmp_path):
    steps = plan(tmp_path)
    assert names(steps) == [
        "write:overlay",
        "run:index-build",
        "stop:backend",
        "spawn:backend",
        "wait:backend",
    ]
    build = next(s for s in steps if s.name == "index-build")
    assert build.argv[-2:] == ("indexer", "build") and build.inherit == stack.CRAWL_CONTACT
    stop = next(s for s in steps if s.kind == "stop")
    assert stop.path == f"{tmp_path}/.rag/pids/backend.pid"


def test_the_full_set_and_resume_reach_the_indexer(tmp_path):
    a = stack.Args(S4B["name"], profile="tu-informatik")
    steps = plan(tmp_path, a, full=True, resume=True)
    build = next(s for s in steps if s.name == "index-build")
    assert build.argv[-5:] == ("indexer", "build", "--sources", "full", "--resume")
    assert "full" in next(s.text for s in steps if s.kind == "say" and "source set" in s.text)


def test_a_stopped_backend_is_left_stopped(tmp_path):
    steps = plan(
        tmp_path, facts=running(tmp_path, alive=frozenset({"embedding", "reranker"}), ready=frozenset({"embedding"}))
    )
    assert names(steps) == ["write:overlay", "run:index-build"]
    assert "rag up" in steps[-1].text


def test_the_restarted_backend_keeps_its_settings(tmp_path):
    a = stack.Args(S4B["name"], profile="tu-informatik")
    steps = plan(tmp_path, a)
    spawn = next(s for s in steps if s.kind == "spawn")
    assert spawn.env["RAG__CARD"] == S4B["name"] and spawn.env["RAG_PROFILE"].endswith("tu-informatik/profile.yaml")
    assert "write:overlay" not in names(steps)  # the TU profile has no folder source


def test_a_hosted_card_restarts_with_its_key_by_name(tmp_path):
    a = stack.Args(GROQ["name"])
    with pytest.raises(stack.StackError, match="GROQ_API_KEY"):
        plan(tmp_path, a, running(tmp_path, GROQ))
    steps = plan(tmp_path, a, running(tmp_path, GROQ, groq_key=True))
    assert next(s for s in steps if s.kind == "spawn").inherit == ("GROQ_API_KEY",)


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        ({"postgres": False}, "rag up"),
        ({"ready": frozenset({"backend"})}, "embedding"),
        ({"docker": False}, "Docker"),
    ],
)
def test_reindex_needs_the_stack_up_and_says_so(tmp_path, facts, message):
    with pytest.raises(stack.StackError, match=message):
        plan(tmp_path, facts=running(tmp_path, **facts))


def test_reindex_before_the_first_up_says_to_run_up(tmp_path):
    a = stack.Args(S4B["name"])
    with pytest.raises(stack.StackError, match="rag up"):
        stack.plan_reindex(state.ensure(tmp_path), running(tmp_path), a)


def test_compose_runs_the_indexer_job_then_restarts_the_backend_container(tmp_path):
    a = stack.Args(S4B["name"], "compose")
    steps = plan(tmp_path, a, running(tmp_path, ready=frozenset({"backend"})), full=True, resume=True)
    assert names(steps) == ["run:index-build", "run:restart-backend", "wait:backend"]
    build, restart, wait = steps
    assert build.argv[-9:] == ("--profile", "build", "run", "--rm", "indexer", "build", "--sources", "full", "--resume")
    assert restart.argv[-2:] == ("restart", "backend") and wait.url.endswith(":8700/readyz")


def test_rollback_is_the_same_job_with_another_command(tmp_path):
    a = stack.Args(S4B["name"])
    steps = stack.plan_rollback(first_run(tmp_path, a), running(tmp_path), a)
    assert names(steps) == ["write:overlay", "run:rollback", "stop:backend", "spawn:backend", "wait:backend"]
    assert next(s for s in steps if s.name == "rollback").argv[-2:] == ("indexer", "rollback")
    assert next(s for s in steps if s.name == "rollback").inherit == ()


def test_rollback_across_profiles_is_refused(tmp_path):
    """The index was last built for another profile than the one the stack runs: its previous generation is not this one's."""
    a = stack.Args(S4B["name"], profile="tu-informatik")
    other = first_run(tmp_path, stack.Args(S4B["name"]))
    current = state.State(other.dir, other.token, other.passwords, {**other.data, "index_profile": "example"})
    with pytest.raises(stack.StackError, match="holds the example profile, not tu-informatik.*rag up --profile"):
        stack.plan_rollback(current, running(tmp_path), a)
    same = state.State(other.dir, other.token, other.passwords, {**other.data, "index_profile": "tu-informatik"})
    assert "run:rollback" in names(stack.plan_rollback(same, running(tmp_path), a))


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


class Recorder(stack.System):
    def __init__(self):
        super().__init__({})
        self.ran = []

    def run(self, argv, cwd, env, timeout_s=0, inherit=()):
        self.ran.append(argv)
        return 0

    def popen(self, argv, cwd, env, log, inherit=()):
        self.ran.append(argv)
        return 4242

    def start_time(self, pid):
        return "now"

    def http(self, method, url, body, timeout_s):
        return 200, "{}"

    def alive(self, pid):
        return False


def settled_state(repo, **data):
    current = state.ensure(repo)
    (repo / ".rag" / "state.json").write_text(
        json.dumps({**current.data, **stack._signature(stack.Args(S4B["name"])), "first_run_done": True, **data})
    )


@pytest.mark.parametrize("command", [["reindex", "--full", "--resume"], ["rollback"]])
def test_the_commands_run_the_job_and_restart_the_backend(repo, monkeypatch, capsys, command):
    system = Recorder()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(stack, "gather", lambda *a, **k: running(repo))
    settled_state(repo)
    assert cli.main(command) == 0
    out = capsys.readouterr().out
    job = [a for a in system.ran if "indexer" in a][0]
    assert job[-1] == ("--resume" if command[0] == "reindex" else "rollback")
    assert "backend: restarting" in out and any("gunicorn" in a for a in system.ran[-1])


def test_the_commands_need_the_repository_root_and_a_first_up(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["reindex"]) == 1 and "root of the repository" in capsys.readouterr().err


def test_reindex_without_a_first_up_says_so(repo, capsys):
    assert cli.main(["reindex"]) == 1
    assert "rag up" in capsys.readouterr().err and not (repo / ".rag").exists()


def test_the_crawl_estimates_are_the_specs():
    assert "300 pages" in classes.CRAWL_SETS["core"] and "2,000 pages" in classes.CRAWL_SETS["full"]


def test_registered_folders_are_not_indexed_by_the_compose_indexer_and_it_says_so(repo, monkeypatch, capsys):
    from rag_tuda import sources

    (repo / "docs").mkdir()
    sources.register(repo, repo / "docs")
    settled_state(repo, mode="compose")
    system = Recorder()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(stack, "gather", lambda *a, **k: running(repo, ready=frozenset({"backend"})))
    assert cli.main(["reindex"]) == 0
    assert "compose indexer does not mount" in capsys.readouterr().err


def test_a_native_reindex_hands_the_indexer_the_sources_file(tmp_path):
    build = next(s for s in plan(tmp_path) if s.name == "index-build")
    assert build.env["RAG__PATHS__USER_SOURCES"] == f"{tmp_path}/.rag/sources.json"
