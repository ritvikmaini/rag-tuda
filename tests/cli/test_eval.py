"""rag eval: the flags become the evals job against the running stack; --card runs the job on that card, bringing the
stack onto it first."""

import json
import os

import pytest

from rag_tuda import cards, cli, doctor, eval as rag_eval, stack, state

S4B = cards.named("s-qwen3-4b-instruct-2507")
S35 = cards.named("s-gemma4-e4b")
GROQ = cards.named("l-groq-gpt-oss-120b")
ALL_HF = frozenset(pin.key for pin in stack.PINNED.values())
RUNNING = frozenset({"embedding", "reranker", "backend"})


def facts(tmp_path, card=S4B, **kwargs) -> stack.Facts:
    values = {"docker": True, "hf": ALL_HF, "alive": RUNNING, "ready": RUNNING, "postgres": True, "index_built": True}
    return stack.Facts(root=str(tmp_path), card=card, **{**values, **kwargs})


def recorded(tmp_path, a: stack.Args) -> state.State:
    current = state.ensure(tmp_path)
    return state.State(
        current.dir, current.token, current.passwords, {**current.data, **stack._signature(a), "first_run_done": True}
    )


def job(tmp_path, a=None, f=None, **flags):
    a = a or stack.Args(S4B["name"])
    options = {"retrieval_only": False, "ablation": False, **flags}
    (step,) = rag_eval.plan(recorded(tmp_path, a), f or facts(tmp_path), a, **options)
    return step


def test_eval_smoke_maps_to_the_evals_runner(tmp_path):
    step = job(tmp_path)
    assert step.argv[:5] == ("uv", "run", "--frozen", "--package", "evals")
    assert step.argv[5:] == ("evals", "smoke", "--label", S4B["name"])
    env = step.env
    assert env["RAG__EVALS__ASSISTANT_URL"] == "http://127.0.0.1:8700"
    assert env["RAG_SECRETS_DIR"] == f"{tmp_path}/.rag/secrets/backend"  # the token is read from its file, never passed
    assert env["RAG__EVALS__PROFILE_DIR"] == f"{tmp_path}/profiles/example"
    assert env["RAG__EVALS__OUT_DIR"].startswith(f"{tmp_path}/.rag/")
    assert "token" not in " ".join(step.argv).lower()


def test_retrieval_only_runs_the_assistants_module(tmp_path):
    step = job(tmp_path, retrieval_only=True)
    assert step.argv[3:] == ("--package", "assistant-api", "python", "-m", "assistant_api.retrieval_eval")
    assert step.env["RAG__CARD"] == S4B["name"] and step.env["RAG_SECRETS_DIR"].endswith("secrets/backend")
    assert job(tmp_path, retrieval_only=True, ablation=True).argv[-2:] == ("assistant_api.retrieval_eval", "--ablation")


def test_a_hosted_card_gets_its_key_by_name_for_the_analysis_only_run(tmp_path):
    a = stack.Args(GROQ["name"])
    step = job(tmp_path, a, facts(tmp_path, GROQ, groq_key=True), retrieval_only=True)
    assert step.inherit == ("GROQ_API_KEY",) and "gsk" not in json.dumps(step.env)


def test_a_hosted_card_without_its_key_is_refused_before_anything_runs(tmp_path):
    with pytest.raises(stack.StackError, match="GROQ_API_KEY"):
        job(tmp_path, stack.Args(GROQ["name"]), facts(tmp_path, GROQ), retrieval_only=True)


def test_the_run_needs_the_stack_it_measures(tmp_path):
    with pytest.raises(stack.StackError, match="rag up"):
        job(tmp_path, f=facts(tmp_path, ready=frozenset({"embedding", "reranker"})))
    with pytest.raises(stack.StackError, match="rag up"):
        job(tmp_path, f=facts(tmp_path, ready=frozenset({"backend"})), retrieval_only=True)


def test_compose_runs_the_smoke_in_the_evals_container_and_refuses_the_retrieval_run(tmp_path):
    a = stack.Args(S4B["name"], "compose")
    out = f"{tmp_path}/.rag/eval_runs"
    folder, step = rag_eval.plan(recorded(tmp_path, a), facts(tmp_path), a, retrieval_only=False, ablation=False)
    assert folder.argv == ("mkdir", "-p", out)  # made by this user: Docker would create it as root
    user = f"{os.getuid()}:{os.getgid()}"
    assert step.argv[step.argv.index("run") :] == (
        "run", "--rm", "--user", user, "-v", f"{out}:{out}", "-e", f"RAG__EVALS__OUT_DIR={out}",
        "evals", "smoke", "--label", S4B["name"],
    )  # fmt: skip
    with pytest.raises(stack.StackError, match="native"):
        job(tmp_path, a, retrieval_only=True)


# --- the command --------------------------------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path, monkeypatch, scenario):
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy" / "compose.yaml").write_text("name: rag-tuda\n")
    monkeypatch.chdir(tmp_path)
    hardware = scenario("m2_16")
    hardware.docker.ok = True
    monkeypatch.setattr(doctor, "gather", lambda: hardware)
    current = state.ensure(tmp_path)
    data = {**current.data, **stack._signature(stack.Args(S4B["name"])), "first_run_done": True}
    (tmp_path / ".rag" / "state.json").write_text(json.dumps(data))
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


def test_the_checks_of_the_run_come_before_the_download_confirmation(repo, monkeypatch, capsys):
    """A card switch needs a pull and so a yes; a retrieval-only run in compose mode can never happen, so no yes is asked."""
    data = json.loads((repo / ".rag" / "state.json").read_text())
    (repo / ".rag" / "state.json").write_text(json.dumps({**data, "mode": "compose"}))
    system = Recorder()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(stack, "gather", lambda root, current, card, hardware, a, *r: facts(repo, card, ollama_up=True))
    asked = []
    monkeypatch.setattr(cli, "_confirm", lambda steps, yes, profile: asked.append(steps) or True)
    assert cli.main(["eval", "--card", S35["name"], "--retrieval-only"]) == 1
    assert "native stack" in capsys.readouterr().err and asked == [] and system.ran == []


def test_the_command_runs_the_job_on_the_recorded_card(repo, monkeypatch, capsys):
    system = Recorder()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(stack, "gather", lambda root, current, card, hardware, a, *r: facts(repo, card))
    assert cli.main(["eval", "--retrieval-only", "--ablation"]) == 0
    assert [a[-1] for a in system.ran] == ["--ablation"]


def test_a_card_that_is_not_running_is_brought_up_before_the_job(repo, monkeypatch, capsys):
    system = Recorder()
    monkeypatch.setattr(stack, "System", lambda: system)
    monkeypatch.setattr(stack, "gather", lambda root, current, card, hardware, a, *r: facts(repo, card, ollama_up=True))
    assert cli.main(["eval", "--card", S35["name"], "--yes"]) == 0
    assert "restarting with the new settings" in capsys.readouterr().out
    assert json.loads((repo / ".rag" / "state.json").read_text())["card"] == S35["name"]
    assert system.ran[-1][-2:] == ("--label", S35["name"])


def test_the_command_refuses_what_it_cannot_run(repo, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["eval", "--ablation"])  # the ablation is a retrieval run
    assert exc.value.code == 2
    assert cli.main(["eval", "--card", "nope"]) == 1 and "nope" in capsys.readouterr().err
    (repo / ".rag" / "state.json").unlink()
    assert cli.main(["eval"]) == 1 and "rag up" in capsys.readouterr().err
