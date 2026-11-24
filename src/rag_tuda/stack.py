"""`rag up` and `rag down`: what to do (`plan_up`, `plan_down`: pure functions from the state, the facts about the machine and
the arguments to a list of steps) and the executor that runs the steps.

Native: Postgres alone in Docker on 127.0.0.1:55432 (deploy/compose.native.yaml), then the migrations, the embedding and
reranker servers, the first index build and the backend as uv processes with pid files and logs, and the Ollama setup of the
card. Compose: the compose files of deploy/ with the env file `.rag/compose.env`. A step whose work is done is not planned:
a second `up` plans only lines saying so."""

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from rag_tuda import classes, runtime, state as state_module
from rag_tuda.state import State

PROJECT = "rag-tuda"
HF_HOST = "huggingface.co"
OLLAMA_HOST = "registry.ollama.ai"
CRAWL_CONTACT = (
    "RAG__CRAWL__CONTACT",
)  # the user's own contact for the crawler's user agent: passed to the index build, never stored
KEYS = ("GROQ_API_KEY",)  # provider keys: only a step that names one gets it
KEEP_RAG_ENV = ("RAG_DEVICE",)  # the one RAG_ variable a child inherits; every other is set by the plan
DB_VOLUME = f"{PROJECT}_db-data"  # the compose project's Postgres volume
STATS_FILE = "corpus_build_stats.json"  # the indexer's record of its last published build
READY_WAIT_S = {"embedding": 900, "reranker": 900, "backend": 180}
SERVICES = {"embedding": runtime.PORTS["embedding"], "reranker": runtime.PORTS["reranker"]}
# the profiles whose folder sources read a folder of the profile (source name -> folder in the profile); the TU profile
# crawls the web and has none, so its index build needs no overlay file
FOLDER_SOURCES = {"example": {"gesetze": "documents"}, "tu-informatik": {}}
# the huggingface_hub the model servers lock (tests/cli/test_stack.py compares it with their uv.lock files); a download runs
# this release through `uv tool run`, so no model server project environment (torch) is installed on the host
HF_HUB_VERSION = "0.36.2"
HF_ENV = {"HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}


@dataclass(frozen=True)
class Pinned:
    """A Hugging Face model at the revision the profile and the model server's image pin (tests/cli/test_stack.py)."""

    repo: str
    revision: str
    env: Mapping[str, str]  # what the model server reads, besides the offline switches

    @property
    def key(self) -> str:
        return f"{self.repo}@{self.revision}"


PINNED = {
    "embedding": Pinned(
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "95c2741480856aa9666782eb4afe11959938017f",
        {
            "EMBEDDING_MODEL": "Snowflake/snowflake-arctic-embed-m-v2.0",
            "EMBEDDING_REVISION": "95c2741480856aa9666782eb4afe11959938017f",
            "EMBEDDING_CODE_REVISION": "95c2741480856aa9666782eb4afe11959938017f",
            "EMBEDDING_CONFIG_KWARGS": '{"use_memory_efficient_attention": false, "unpad_inputs": false}',
        },
    ),
    "reranker": Pinned(
        "BAAI/bge-reranker-v2-m3",
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        {"RERANKER_MODEL": "BAAI/bge-reranker-v2-m3", "RERANKER_REVISION": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"},
    ),
}
MODULES = {"embedding": "embedding_server.main:create_app", "reranker": "reranker_server.main:create_app"}


class StackError(Exception):
    """A step failed, or the plan cannot be made; the message says what to do."""


@dataclass(frozen=True)
class Step:
    kind: str  # say | write | run | spawn | wait | request | stop | remove
    name: str = ""
    text: str = field(default="", repr=False)  # the line to print; for a write the content (it may hold a secret)
    argv: tuple[str, ...] = ()
    cwd: str = ""
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    path: str = ""  # write, remove: the target; spawn, wait, stop: the pid file
    log: str = ""
    url: str = ""
    body: str = ""
    timeout_s: float = 0
    gb: float = 0  # a download's size, when known
    host: str = ""  # a download's server: named when it fails
    inherit: tuple[
        str, ...
    ] = ()  # variables of our own environment the process gets (a key, never copied into the plan)

    @property
    def is_download(self) -> bool:
        return bool(self.host)


@dataclass(frozen=True)
class Args:
    card: str
    mode: str = "native"
    profile: str = "example"
    model: str | None = None
    llm_url: str | None = None


@dataclass(frozen=True)
class Facts:
    """What the machine looks like; `gather` reads it, `plan_up` only looks."""

    root: str
    card: dict
    docker: bool = True
    nvidia: bool = False
    ollama_up: bool = False
    ollama_models: frozenset[str] = frozenset()
    ollama_derived: Mapping[str, Mapping[str, str]] = field(default_factory=dict)  # name -> parameters it has
    hf: frozenset[str] = frozenset()  # "repo@revision" snapshots in the Hugging Face cache
    hf_home: str = ""  # compose mode: the Hugging Face home the containers mount (RAG_HF_CACHE)
    alive: frozenset[str] = frozenset()  # native processes with a live pid
    ready: frozenset[str] = frozenset()  # services answering /readyz
    postgres: bool = False
    index_built: bool = False
    compose_images: bool = False
    compose_ollama_models: frozenset[str] = frozenset()
    compose_ollama_up: bool = False  # the compose Ollama container answers `ollama list`: its list is the truth
    no_venv: frozenset[str] = (
        frozenset()
    )  # model servers whose project has no environment yet: the first start installs it
    crawl_contact: bool = False
    groq_key: bool = False


def has_model(models: frozenset[str], tag: str) -> bool:
    return tag in models or (":" not in tag and f"{tag}:latest" in models)


def same_parameters(have: Mapping[str, str], want: Mapping[str, object]) -> bool:
    def equal(a: str, b: object) -> bool:
        if isinstance(b, list):  # a repeated parameter (stop): one quoted line per value
            return sorted(line.strip().strip('"') for line in a.splitlines()) == sorted(map(str, b))
        try:
            return float(a) == float(b)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return a == str(b)

    return all(k in have and equal(have[k], v) for k, v in want.items())


def uv_argv(*args: str) -> tuple[str, ...]:
    return ("uv", "run", "--frozen", *args)


def _compose(root: str, rag: str, *files: str) -> tuple[str, ...]:
    argv = ["docker", "compose", "-p", PROJECT, "--env-file", f"{rag}/compose.env"]
    for name in ("compose.yaml", *files):
        argv += ["-f", f"{root}/deploy/{name}"]
    return tuple(argv)


def _download(name: str, pin: Pinned, root: str) -> Step:
    gb = next((m.gb for m in classes.FIXED_MODELS if m.name == pin.repo), 0.0)
    return Step(
        "run",
        name=f"download-{name}",
        text=f"downloading {pin.repo} ({gb:g} GB) from {HF_HOST}",
        argv=(
            *("uv", "tool", "run", "--from", f"huggingface_hub=={HF_HUB_VERSION}"),
            *("huggingface-cli", "download", pin.repo, "--revision", pin.revision),
        ),
        cwd=root,
        env=HF_ENV,
        gb=gb,
        host=HF_HOST,
    )


def _pull(tag: str, gb: float, api: str | None, compose: tuple[str, ...] | None) -> Step:
    if compose:
        argv, env = (*compose, "exec", "-T", "ollama", "ollama", "pull", tag), {}
    else:
        argv, env = ("ollama", "pull", tag), {"OLLAMA_HOST": api or ""}
    return Step(
        "run",
        name=f"pull-{tag}",
        text=f"downloading {tag}" + (f" ({gb:g} GB)" if gb else "") + f" from {OLLAMA_HOST}",
        argv=argv,
        env=env,
        gb=gb,
        host=OLLAMA_HOST,
    )


def _signature(args: Args) -> dict:
    return {"mode": args.mode, "card": args.card, "profile": args.profile, "model": args.model, "llm_url": args.llm_url}


def _index_recorded(state: State, args: Args) -> Step:
    """Right after a build: which profile the index holds, so a later step that fails does not lose it."""
    text = json.dumps({**state.data, "index_profile": args.profile}, indent=2) + "\n"
    return Step("write", name="state-index", path=f"{state.dir}/state.json", text=text)


def _record(state: State, args: Args, llm: runtime.Llm, built: bool) -> list[Step]:
    new = {**state.data, **_signature(args), "first_run_done": True}
    if built:
        new["index_profile"] = args.profile
    if args.mode == "compose":  # what the compose Ollama volume holds, for the next up after a down
        new["compose_models"] = sorted({*state.data.get("compose_models", ()), *llm.pulls})
    if new == state.data:
        return []
    return [Step("write", name="state", path=f"{state.dir}/state.json", text=json.dumps(new, indent=2) + "\n")]


def plan_up(state: State, facts: Facts, args: Args) -> list[Step]:
    """The steps that bring the stack up from what the machine looks like; ValueError or StackError when it cannot."""
    llm = runtime.resolve(facts.card, args.mode, model=args.model, llm_url=args.llm_url)
    if not facts.docker:
        raise StackError(
            "Docker is not running: start Docker Desktop, OrbStack or Colima (`colima start`), then run `rag up` again"
        )
    if llm.ollama_api and not facts.ollama_up:
        raise StackError(
            f"Ollama is not running at {llm.ollama_api}: start it (the Ollama app, or `ollama serve`), then run `rag up` again"
        )
    if llm.provider in runtime.REMOTE and not facts.groq_key:
        raise StackError(
            f"{llm.card}: its answers come from Groq, which needs GROQ_API_KEY in the environment "
            "(export it in this shell, then run `rag up` again; it is passed to the backend only and never stored)"
        )
    steps: list[Step] = []
    if args.llm_url:
        steps.append(
            Step(
                "say",
                text=f"note: {args.llm_url} is a server you run: rag does not set its context window (num_ctx); "
                f"the card's budget needs {llm.window} tokens, so set it there (Ollama: OLLAMA_CONTEXT_LENGTH)",
            )
        )
    if not facts.crawl_contact and args.profile != "example":
        steps += _CONTACT_NOTE
    plan = _plan_native if args.mode == "native" else _plan_compose
    steps += plan(state, facts, args, llm)
    return steps + _record(state, args, llm, any(s.name == "index-build" for s in steps))


def _profile_changed(state: State, args: Args) -> bool:
    """A later `up` names another profile: the index and the images of the first profile are not the new one's."""
    return bool(state.data.get("first_run_done")) and state.data.get("profile") != args.profile


def _plan_native(state: State, facts: Facts, args: Args, llm: runtime.Llm) -> list[Step]:
    root, rag = facts.root, str(state.dir)
    steps: list[Step] = [_download(name, pin, root) for name, pin in PINNED.items() if pin.key not in facts.hf]
    for tag in llm.pulls:
        if not has_model(facts.ollama_models, tag):
            # the card states the size of its own answer model only
            gb = (facts.card.get("download_gb") or 0) if tag == facts.card["answer"]["model"] else 0
            steps.append(_pull(tag, gb, llm.ollama_api, None))
    if facts.postgres:
        steps.append(Step("say", text="postgres: already running"))
    else:
        env_file = runtime.compose_env_text(
            llm, token=state.token, roles_dir=str(state.secrets_dir("roles")), profile=args.profile
        )
        port = runtime.PORTS["postgres"]
        steps += [
            Step("write", name="compose-env", path=f"{rag}/compose.env", text=env_file),
            Step(
                "run",
                name="postgres",
                text=f"postgres: starting on {runtime.LOOPBACK}:{port}",
                argv=(*_compose(root, rag, "compose.native.yaml"), "up", "-d", "--wait", "pgvector"),
                cwd=root,
                timeout_s=300,
            ),
        ]
    for name, port in SERVICES.items():
        env = {**runtime.PRIVACY, **PINNED[name].env}
        steps += _service(
            name,
            root,
            rag,
            port,
            env,
            alive=name in facts.alive,
            ready=name in facts.ready,
            first_start=name in facts.no_venv,
        )
    restart = "backend" in facts.alive and _signature(args) != {k: state.data.get(k) for k in _signature(args)}
    backend_up = "backend" in facts.ready and not restart
    if restart:
        steps.append(
            Step(
                "stop", name="backend", text="backend: restarting with the new settings", path=f"{rag}/pids/backend.pid"
            )
        )
    if not backend_up:
        steps.append(
            Step(
                "run",
                name="migrate",
                text="migrating the database",
                argv=uv_argv("--package", "assistant-api", "python", "-m", "assistant_api.migrate"),
                cwd=root,
                env=runtime.migrate_env(str(state.secrets_dir("migrate"))),
            )
        )
    if not facts.index_built:
        steps += _build_index(state, facts, args)
    created = False
    for body, window in (
        (runtime.derived_model_body(llm, facts.card), llm.window),
        (runtime.analysis_model_body(llm, facts.card), llm.analysis_window),
    ):
        if body is None:
            continue
        have = facts.ollama_derived.get(body["model"], {})
        # the derived model records its window and sampling, not its base: a changed base is made again
        base_changed = body["model"] == llm.derived and _base_changed(state, args)
        if base_changed or not same_parameters(have, body["parameters"]):
            created = created or body["model"] == llm.derived
            steps.append(
                Step(
                    "request",
                    name="derived-model" if body["model"] == llm.derived else "analysis-model",
                    text=f"creating the Ollama model {body['model']} (window {window})",
                    url=f"{llm.ollama_api}/api/create",
                    body=json.dumps(body),
                    timeout_s=300,
                )
            )
    steps += _backend(state, facts, args, llm, alive="backend" in facts.alive and not restart, ready=backend_up)
    if llm.warms and (not backend_up or created):
        steps.append(_warm(llm))
    return steps


def _backend(state: State, facts: Facts, args: Args, llm: runtime.Llm, *, alive: bool, ready: bool) -> list[Step]:
    env = runtime.backend_env(llm, root=facts.root, secrets=str(state.secrets_dir("backend")), profile=args.profile)
    return _service(
        "backend",
        facts.root,
        str(state.dir),
        runtime.PORTS["backend"],
        env,
        alive=alive,
        ready=ready,
        inherit=("GROQ_API_KEY",) if llm.provider in runtime.REMOTE else (),
    )


def _base_changed(state: State, args: Args) -> bool:
    return bool(state.data.get("first_run_done")) and state.data.get("model") != args.model


def _build_index(state: State, facts: Facts, args: Args) -> list[Step]:
    text = f"building the index of the {args.profile} profile (indexer build)"
    return [
        *_crawl_estimate(args, "core"),
        *_indexer_job(state, facts, args, "index-build", ("build",), text, CRAWL_CONTACT),
        _index_recorded(state, args),
    ]


def _crawl_estimate(args: Args, which: str) -> list[Step]:
    """The crawl is outbound traffic: its size is announced before it starts (the example profile crawls nothing)."""
    if args.profile == "example":
        return []
    text = f"crawling the {which} source set ({classes.CRAWL_SETS[which]}), about 1 request per second per host"
    return [Step("say", text=text)]


def _indexer_job(
    state: State, facts: Facts, args: Args, name: str, tail: tuple[str, ...], text: str, inherit: tuple[str, ...] = ()
) -> list[Step]:
    """One `indexer` command as a native process: its overlay file (the profile's folder sources) and the run."""
    root, rag = facts.root, str(state.dir)
    steps: list[Step] = []
    overlay = runtime.overlay_text(root, args.profile, FOLDER_SOURCES.get(args.profile, {}))
    if overlay:
        steps.append(Step("write", name="overlay", path=f"{rag}/overlay.yaml", text=overlay))
    env = runtime.indexer_env(
        root=root,
        rag=rag,
        secrets=str(state.secrets_dir("indexer")),
        profile=args.profile,
        overlay=f"{rag}/overlay.yaml" if overlay else None,
    )
    argv = uv_argv("--package", "indexer", "indexer", *tail)
    return [*steps, Step("run", name=name, text=text, argv=argv, cwd=root, env=env, inherit=inherit)]


def _service(
    name: str,
    root: str,
    rag: str,
    port: int,
    env: Mapping[str, str],
    *,
    alive: bool,
    ready: bool,
    inherit: tuple[str, ...] = (),
    first_start: bool = False,
) -> list[Step]:
    """Start a native process unless its pid is alive, then wait until it answers; nothing when it already does."""
    if ready:
        return [Step("say", text=f"{name}: already running")]
    pid, log = f"{rag}/pids/{name}.pid", f"{rag}/logs/{name}.log"
    steps: list[Step] = []
    if not alive:
        if name == "backend":
            gunicorn = ("gunicorn", "assistant_api.app:create_app()", "-c", "apps/assistant_api/gunicorn_conf.py")
            argv, cwd = uv_argv("--package", "assistant-api", *gunicorn), root
        else:
            directory = f"{root}/model_servers/{name}"  # the server's package is found next to its project
            argv = uv_argv(
                "--project",
                directory,
                "uvicorn",
                "--factory",
                MODULES[name],
                "--host",
                runtime.LOOPBACK,
                "--port",
                str(port),
            )
            cwd = directory
        text = f"{name}: starting on {runtime.LOOPBACK}:{port}"
        if first_start:  # uv builds the server's environment (torch and more, several GB from PyPI) before it runs
            text += " (the first start installs its Python environment, which takes minutes and several GB)"
        steps.append(
            Step("spawn", name=name, text=text, argv=argv, cwd=cwd, env=env, path=pid, log=log, inherit=inherit)
        )
    url = f"http://{runtime.LOOPBACK}:{port}/readyz"
    steps.append(
        Step(
            "wait",
            name=name,
            text=f"{name}: waiting until it is ready",
            url=url,
            path=pid,
            log=log,
            timeout_s=READY_WAIT_S[name],
        )
    )
    return steps


def _warm(llm: runtime.Llm) -> Step:
    return Step(
        "request",
        name="warm-up",
        text=f"loading the answer model {llm.answer_model} (a first request)",
        url=f"{llm.answer_url}/chat/completions",
        body=_warm_body(llm),
        timeout_s=900,
    )


def _plan_compose(state: State, facts: Facts, args: Args, llm: runtime.Llm) -> list[Step]:
    root, rag = facts.root, str(state.dir)
    compose = compose_for(facts, rag, llm)
    # the indexer's overlay names the profile's folder sources as the container sees them (/app/profiles/<profile>/...)
    overlay = runtime.overlay_text("/app", args.profile, FOLDER_SOURCES.get(args.profile, {})) or "{}\n"
    env_file = runtime.compose_env_text(
        llm,
        token=state.token,
        roles_dir=str(state.secrets_dir("roles")),
        profile=args.profile,
        overlay=f"{rag}/indexer-overlay.yaml",
        hf_home=facts.hf_home or None,
    )
    steps = [
        Step("write", name="indexer-overlay", path=f"{rag}/indexer-overlay.yaml", text=overlay),
        Step("write", name="compose-env", path=f"{rag}/compose.env", text=env_file),
    ]
    steps += [_download(name, pin, root) for name, pin in PINNED.items() if pin.key not in facts.hf]
    if "backend" in facts.ready and _signature(args) == {k: state.data.get(k) for k in _signature(args)}:
        return [*steps, Step("say", text="the compose stack: already running")]
    profile_changed = _profile_changed(state, args)
    if not facts.compose_images or profile_changed:  # the images carry the profile, the indexer's and the evals' too
        build = (*compose, "--profile", "build", "--profile", "evals", "build")
        steps.append(Step("run", name="build", text="building the images", argv=build, cwd=root))
    steps += [
        Step(
            "run",
            name="postgres",
            text="postgres: starting",
            argv=(*compose, "up", "-d", "--wait", "pgvector"),
            cwd=root,
            timeout_s=300,
        ),
        Step(
            "run",
            name="migrate",
            text="migrating the database",
            argv=(*compose, "--profile", "migrate", "run", "--rm", "migrate"),
            cwd=root,
        ),
    ]
    if llm.provider == "ollama":
        steps.append(
            Step(
                "run",
                name="ollama",
                text="ollama: starting",
                argv=(*compose, "up", "-d", "--wait", "ollama"),
                cwd=root,
                timeout_s=300,
            )
        )
        # the container is gone after `down`, its volume is not: the models pulled before are recorded in the state file
        recorded = frozenset(state.data.get("compose_models", ()))
        have = facts.compose_ollama_models if facts.compose_ollama_up else facts.compose_ollama_models | recorded
        pulls = [
            _pull(tag, facts.card.get("download_gb") or 0, None, compose)
            for tag in llm.pulls
            if not has_model(have, tag)
        ]
        steps += pulls
        if pulls:  # recorded at once, so a later failure does not make the next up pull again
            models = sorted(recorded | set(llm.pulls))
            text = json.dumps({**state.data, "compose_models": models}, indent=2) + "\n"
            steps.append(Step("write", name="state-models", path=f"{state.dir}/state.json", text=text))
    if not facts.index_built:
        steps += [
            Step(
                "run",
                name="index-build",
                text=f"building the index of the {args.profile} profile (indexer build)",
                argv=(*compose, "--profile", "build", "run", "--rm", "indexer", "build"),
                cwd=root,
                inherit=CRAWL_CONTACT,  # compose.yaml hands it on to the indexer container
            ),
            _index_recorded(state, args),
        ]
    steps.append(
        Step(
            "run", name="up", text="starting the stack", argv=(*compose, "up", "-d", "--wait"), cwd=root, timeout_s=900
        )
    )
    if llm.warms:
        steps.append(_warm_in_compose(llm, compose, root))
    return steps


def compose_for(facts: Facts, rag: str, llm: runtime.Llm) -> tuple[str, ...]:
    overlays = (["compose.nvidia.yaml"] if facts.nvidia else []) + (
        ["compose.vllm.yaml"] if llm.provider == "vllm" else []
    )
    return _compose(facts.root, rag, *overlays)


def _warm_body(llm: runtime.Llm) -> str:
    return json.dumps(
        {
            "model": llm.answer_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "stream": False,
        }
    )


def _warm_in_compose(llm: runtime.Llm, compose: tuple[str, ...], root: str) -> Step:
    """The model server has no host port: the backend container sends the request over the compose network."""
    code = (
        "import urllib.request as u; "
        f"u.urlopen(u.Request({llm.answer_url + '/chat/completions'!r}, {_warm_body(llm)!r}.encode(), "
        "{'Content-Type': 'application/json'}), timeout=900).read()"
    )
    return Step(
        "run",
        name="warm-up",
        text=f"loading the answer model {llm.answer_model} (a first request)",
        argv=(*compose, "exec", "-T", "backend", "python", "-c", code),
        cwd=root,
        timeout_s=900,
    )


# --- reindex and rollback ---------------------------------------------------------------------------------------------------


def plan_reindex(state: State, facts: Facts, args: Args, full: bool = False, resume: bool = False) -> list[Step]:
    """A new index generation from a re-crawl and an atomic swap, then a backend restart. `args` is what the last `up`
    recorded; `full` adds the full source set, `resume` continues an interrupted crawl."""
    which = "full" if full else "core"
    tail = ("build", *(("--sources", "full") if full else ()), *(("--resume",) if resume else ()))
    text = f"building a new index generation of the {args.profile} profile (indexer build, {which} source set)"
    note = [] if facts.crawl_contact or args.profile == "example" else _CONTACT_NOTE
    return _index_job(
        state, facts, args, "index-build", tail, text, [*_crawl_estimate(args, which), *note], CRAWL_CONTACT
    )


def plan_rollback(state: State, facts: Facts, args: Args) -> list[Step]:
    """The previous index generation goes live again, then a backend restart. Refused when the index was built for
    another profile than the one the stack runs."""
    built_for = state.data.get("index_profile")
    if built_for and built_for != args.profile:
        raise StackError(
            f"the index holds the {built_for} profile, not {args.profile}: a rollback would not fit "
            f"the {args.profile} stack; run `rag up --profile {args.profile}` to build its index"
        )
    return _index_job(
        state, facts, args, "rollback", ("rollback",), "rolling the index back (indexer rollback)", [], ()
    )


_CONTACT_NOTE = [
    Step("say", text="note: RAG__CRAWL__CONTACT is not set; the crawler's user agent will not carry a contact")
]


def _index_job(
    state: State,
    facts: Facts,
    args: Args,
    name: str,
    tail: tuple[str, ...],
    text: str,
    notes: list[Step],
    inherit: tuple[str, ...],
) -> list[Step]:
    llm = runtime.resolve(facts.card, args.mode, model=args.model, llm_url=args.llm_url)
    if not state.data.get("first_run_done"):
        raise StackError("nothing was started from this folder yet: run `rag up` first")
    if not facts.docker:
        raise StackError(
            "Docker is not running: start Docker Desktop, OrbStack or Colima (`colima start`), then try again"
        )
    if not facts.postgres:
        raise StackError("postgres is not running: run `rag up` first")
    if llm.provider in runtime.REMOTE and not facts.groq_key:
        raise StackError(f"{llm.card}: the backend restarts with GROQ_API_KEY from the environment: export it first")
    if args.mode == "native":
        if tail[0] == "build" and "embedding" not in facts.ready:
            raise StackError("the embedding server is not running: run `rag up` first")
        steps = [*notes, *_indexer_job(state, facts, args, name, tail, text, inherit)]
        if "backend" not in facts.alive:
            return [*steps, Step("say", text="backend: not running; `rag up` starts it on the new index")]
        restart = Step(
            "stop", name="backend", text="backend: restarting on the new index", path=f"{state.dir}/pids/backend.pid"
        )
        return [*steps, restart, *_backend(state, facts, args, llm, alive=False, ready=False)]
    compose = compose_for(facts, str(state.dir), llm)
    job = Step(
        "run",
        name=name,
        text=text,
        argv=(*compose, "--profile", "build", "run", "--rm", "indexer", *tail),
        cwd=facts.root,
        inherit=inherit,
    )
    if "backend" not in facts.ready:
        return [*notes, job, Step("say", text="backend: not running; `rag up` starts it on the new index")]
    return [
        *notes,
        job,
        Step(
            "run",
            name="restart-backend",
            text="backend: restarting on the new index",
            argv=(*compose, "restart", "backend"),
            cwd=facts.root,
            timeout_s=300,
        ),
        Step(
            "wait",
            name="backend",
            text="backend: waiting until it is ready",
            url=f"http://{runtime.LOOPBACK}:{runtime.PORTS['backend']}/readyz",
            log="`docker compose -p rag-tuda logs backend`",
            timeout_s=READY_WAIT_S["backend"],
        ),
    ]


# --- down ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DownFacts:
    root: str
    docker: bool
    pid_files: frozenset[str] = frozenset()  # the native processes that have a pid file


def plan_down(rag: Path, facts: DownFacts, purge: bool = False) -> list[Step]:
    """Stop the native processes (a stale pid file is cleared), then the containers; every step is safe to repeat."""
    rag = str(rag)
    steps: list[Step] = []
    for name in ("backend", "reranker", "embedding"):  # the reverse of the start order
        if name in facts.pid_files:
            steps.append(Step("stop", name=name, text=f"{name}: stopping", path=f"{rag}/pids/{name}.pid"))
        else:
            steps.append(Step("say", text=f"{name}: not running"))
    if facts.docker:
        steps.append(
            Step(
                "run",
                name="containers",
                text="stopping the containers",
                argv=("docker", "compose", "-p", PROJECT, "down", "--remove-orphans"),
                cwd=facts.root,
            )
        )
    else:
        steps.append(Step("say", text="Docker is not running: no containers to stop"))
    if purge:
        steps += [
            Step("remove", name=name, text=f"deleting {rag}/{name}", path=f"{rag}/{name}")
            for name in ("crawl", "snapshots")
        ]
    return steps


# --- the executor ----------------------------------------------------------------------------------------------------------


class System:
    """The machine as the executor sees it: processes, HTTP on the loopback and the clock. Tests replace the methods."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self.environ = os.environ if environ is None else environ
        self.sleep: Callable[[float], None] = time.sleep
        self.clock: Callable[[], float] = time.monotonic

    def child_env(self, extra: Mapping[str, str], inherit: tuple[str, ...] = ()) -> dict[str, str]:
        """The environment of a child: ours without the RAG_ variables (the settings loader refuses keys it does not know) and
        without the provider key, plus what the plan sets and the variables the step names to inherit."""
        base = {k: v for k, v in self.environ.items() if not k.startswith("RAG_") or k in KEEP_RAG_ENV}
        for name in KEYS:
            base.pop(name, None)
        return {**base, **{k: self.environ[k] for k in inherit if k in self.environ}, **extra}

    def run(
        self,
        argv: tuple[str, ...],
        cwd: str,
        env: Mapping[str, str],
        timeout_s: float = 0,
        inherit: tuple[str, ...] = (),
    ) -> int:
        try:
            return subprocess.run(
                argv, cwd=cwd or None, env=self.child_env(env, inherit), timeout=timeout_s or None
            ).returncode
        except OSError as exc:
            raise StackError(f"{argv[0]}: cannot be started ({exc.strerror})") from None
        except subprocess.TimeoutExpired:
            raise StackError(f"{' '.join(argv[:3])} ...: no end after {timeout_s:g} s") from None

    def output(self, argv: tuple[str, ...], extra: Mapping[str, str] | None = None) -> str | None:
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=30, env=self.child_env(extra or {}))
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    def popen(
        self, argv: tuple[str, ...], cwd: str, env: Mapping[str, str], log: IO[bytes], inherit: tuple[str, ...] = ()
    ) -> int:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd or None,
                env=self.child_env(env, inherit),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise StackError(f"{argv[0]}: cannot be started ({exc.strerror})") from None
        return process.pid

    def http(self, method: str, url: str, body: str | None, timeout_s: float) -> tuple[int, str]:
        """(status, text); status 0 when nothing answers. Never through a proxy: every URL here is on the loopback."""
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            url,
            data=body.encode() if body is not None else None,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with opener.open(request, timeout=timeout_s) as response:
                return response.status, response.read(65536).decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(4096).decode("utf-8", "replace")
        except (OSError, ValueError) as exc:
            return 0, str(exc)

    def alive(self, pid: int) -> bool:
        try:
            if os.waitpid(pid, os.WNOHANG)[0]:  # our own child that has ended: reaped
                return False
        except ChildProcessError:
            pass
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def start_time(self, pid: int) -> str:
        """When the process started, as `ps` prints it; a recycled pid has another start time. The text is read in the C
        locale and UTC, so a start time recorded in one environment (a terminal, cron, launchd) matches in another."""
        return (self.output(("ps", "-o", "lstart=", "-p", str(pid)), {"LC_ALL": "C", "TZ": "UTC"}) or "").strip()

    def signal_group(self, pid: int, sig: int) -> None:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass


def read_pid(path: str) -> tuple[int, str] | None:
    """(pid, the process's start time) from a pid file; None when it is missing or unreadable."""
    try:
        pid, started = Path(path).read_text(encoding="utf-8").split("\n")[:2]
        return int(pid), started
    except (OSError, ValueError):
        return None


def execute(steps: list[Step], system: System, out: Callable[[str], None] = print) -> None:
    """Runs the steps in order and stops at the first failure with a StackError naming what to do."""
    for step in steps:
        if step.text and step.kind != "write":  # a write's text is the file's content, which may hold a secret
            out(step.text)
        _STEPS[step.kind](step, system, out)


def _say(step: Step, system: System, out: Callable[[str], None]) -> None:
    return None  # the line was printed already


def _write(step: Step, system: System, out: Callable[[str], None]) -> None:
    path = Path(step.path)
    if path.is_file() and path.read_text(encoding="utf-8") == step.text:
        path.chmod(0o600)
        return
    state_module.write_private(path, step.text)


def _run(step: Step, system: System, out: Callable[[str], None]) -> None:
    code = system.run(step.argv, step.cwd, step.env, step.timeout_s, step.inherit)
    if code != 0 and step.is_download:
        raise StackError(
            f"{step.name}: the download from {step.host} failed (exit {code}); the network may block {step.host}"
        )
    if code != 0:
        raise StackError(f"{step.name}: failed (exit {code}): {' '.join(step.argv[:4])} ...")


def _spawn(step: Step, system: System, out: Callable[[str], None]) -> None:
    with os.fdopen(os.open(step.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab") as log:
        pid = system.popen(step.argv, step.cwd, step.env, log, step.inherit)
    state_module.write_private(Path(step.path), f"{pid}\n{system.start_time(pid)}\n")


def _wait(step: Step, system: System, out: Callable[[str], None]) -> None:
    deadline = system.clock() + step.timeout_s
    while True:
        if (system.http("GET", step.url, None, 3)[0]) == 200:
            return
        if (found := read_pid(step.path)) and not system.alive(found[0]):
            raise StackError(f"{step.name} stopped while starting; see {step.log}")
        if system.clock() > deadline:
            raise StackError(f"{step.name} was not ready after {step.timeout_s:g} s; see {step.log}")
        system.sleep(1)


def _request(step: Step, system: System, out: Callable[[str], None]) -> None:
    status, text = system.http("POST", step.url, step.body, step.timeout_s)
    if 200 <= status < 300:
        return
    where = f" (is {step.host} reachable?)" if step.host else ""
    raise StackError(f"{step.name}: {step.url} answered {status or 'nothing'}{where}: {text[:200]}")


def _stop(step: Step, system: System, out: Callable[[str], None]) -> None:
    found = read_pid(step.path)
    if found and system.alive(found[0]) and (not found[1] or found[1] != system.start_time(found[0])):
        # the process behind the pid may still be ours: it is not signalled, and its file stays as the way to find it
        out(f"{step.name}: pid {found[0]} is running but its start time does not match {step.path}; not stopped")
        out(f"  if it is not ours, delete {step.path}; if it is, stop pid {found[0]} by hand")
        return
    if found and system.alive(found[0]):
        pid = found[0]
        system.signal_group(pid, signal.SIGTERM)
        deadline = system.clock() + 20
        while system.alive(pid) and system.clock() < deadline:
            system.sleep(0.2)
        if system.alive(pid):
            system.signal_group(pid, signal.SIGKILL)
    Path(step.path).unlink(missing_ok=True)


def _remove(step: Step, system: System, out: Callable[[str], None]) -> None:
    path = Path(step.path)
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:
        pass  # what is left is named below
    if path.exists():
        raise StackError(
            f"{path} could not be deleted completely (files a container wrote as root?): "
            f"run `sudo rm -rf {path}`, or `docker run --rm -v {path.parent}:/x alpine rm -rf /x/{path.name}`"
        )


_STEPS: dict[str, Callable[[Step, System, Callable[[str], None]], None]] = {
    "say": _say,
    "write": _write,
    "run": _run,
    "spawn": _spawn,
    "wait": _wait,
    "request": _request,
    "stop": _stop,
    "remove": _remove,
}


# --- the facts --------------------------------------------------------------------------------------------------------------


def hf_cache(environ: Mapping[str, str]) -> Path:
    """The Hugging Face hub cache: HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub."""
    if cache := environ.get("HF_HUB_CACHE"):
        return Path(cache)
    return Path(environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"


def hf_home(environ: Mapping[str, str]) -> Path:
    """The Hugging Face home that holds the hub cache `hf_cache` checks: what the compose containers mount."""
    cache = hf_cache(environ)
    if cache.name != "hub":
        raise StackError(
            f"compose mode mounts the Hugging Face home: HF_HUB_CACHE ({cache}) must end in /hub (or set HF_HOME)"
        )
    return cache.parent


def parse_tags(text: str) -> frozenset[str]:
    try:
        return frozenset(m["name"] for m in json.loads(text)["models"])
    except (ValueError, KeyError, TypeError):
        return frozenset()


def parse_parameters(text: str) -> dict[str, str]:
    """`parameters` of /api/show: one `name value` pair per line."""
    try:
        lines = json.loads(text)["parameters"].splitlines()
    except (ValueError, KeyError, TypeError, AttributeError):
        return {}
    found: dict[str, str] = {}
    for line in lines:
        if len(parts := line.split(None, 1)) == 2:
            found[parts[0]] = f"{found[parts[0]]}\n{parts[1]}" if parts[0] in found else parts[1]
    return found


def _complete(snapshot: Path) -> bool:
    """A snapshot with its config and a weights file. The hub links a file into `snapshots/` only when its download is
    finished, and is_file follows the link, so a partial download shows neither."""
    return (snapshot / "config.json").is_file() and any(
        p.is_file() for pattern in ("*.safetensors", "pytorch_model*.bin") for p in snapshot.glob(pattern)
    )


def gather(root: Path, state: State, card: dict, hardware, args: Args, system: System | None = None) -> Facts:
    """Reads the machine for `plan_up`: processes, containers, Ollama, the Hugging Face cache and the index."""
    system = system or System()
    llm = runtime.resolve(card, args.mode, model=args.model, llm_url=args.llm_url)
    pids = {name: found for name in (*SERVICES, "backend") if (found := read_pid(f"{state.dir}/pids/{name}.pid"))}
    # a pid file of an earlier boot or a recycled pid names another process: the start time must match
    alive = frozenset(
        name for name, (pid, started) in pids.items() if system.alive(pid) and started == system.start_time(pid)
    )
    ports = {**SERVICES, "backend": runtime.PORTS["backend"]}
    ready = frozenset(
        name
        for name, port in ports.items()
        if system.http("GET", f"http://{runtime.LOOPBACK}:{port}/readyz", None, 2)[0] == 200
    )
    ollama_up, models, derived = False, frozenset[str](), {}
    if llm.ollama_api:
        status, text = system.http("GET", f"{llm.ollama_api}/api/version", None, 2)
        ollama_up = status == 200
        if ollama_up:
            models = parse_tags(system.http("GET", f"{llm.ollama_api}/api/tags", None, 5)[1])
            for name in filter(None, (llm.derived, llm.analysis_derived)):
                if has_model(models, name):
                    shown = system.http("POST", f"{llm.ollama_api}/api/show", json.dumps({"model": name}), 10)[1]
                    derived[name] = parse_parameters(shown)
    cache = hf_cache(system.environ)
    hf = frozenset(
        pin.key
        for pin in PINNED.values()
        if _complete(cache / f"models--{pin.repo.replace('/', '--')}" / "snapshots" / pin.revision)
    )
    compose = ("docker", "compose", "-p", PROJECT)
    running = system.output((*compose, "ps", "--status", "running", "-q", "pgvector")) if hardware.docker.ok else None
    images = system.output(("docker", "images", "--format", "{{.Repository}}")) if hardware.docker.ok else None
    names = set((images or "").split())
    ollama_list = (
        system.output((*compose, "exec", "-T", "ollama", "ollama", "list"))
        if args.mode == "compose" and hardware.docker.ok
        else None
    )
    index = state.dir / "state" if args.mode == "native" else root / "data" / "vector_database_contents"
    volume = (
        system.output(("docker", "volume", "inspect", DB_VOLUME, "--format", "{{.CreatedAt}}"))
        if hardware.docker.ok
        else None
    )
    return Facts(
        root=str(root),
        card=card,
        docker=hardware.docker.ok,
        nvidia=bool(hardware.nvidia),
        ollama_up=ollama_up,
        ollama_models=models,
        ollama_derived=derived,
        hf=hf,
        hf_home=str(hf_home(system.environ)) if args.mode == "compose" else "",
        alive=alive,
        ready=ready,
        postgres=bool(running and running.strip()),
        index_built=index_built(index / STATS_FILE, args.profile, volume),
        compose_images={f"{PROJECT}-backend", f"{PROJECT}-embedding_service", f"{PROJECT}-reranker_service"} <= names,
        compose_ollama_models=frozenset(
            line.split()[0] for line in (ollama_list or "").splitlines()[1:] if line.split()
        ),
        compose_ollama_up=ollama_list is not None,
        no_venv=frozenset(name for name in SERVICES if not (root / "model_servers" / name / ".venv").is_dir()),
        crawl_contact=bool(system.environ.get("RAG__CRAWL__CONTACT")),
        groq_key=bool(system.environ.get("GROQ_API_KEY", "").strip()),
    )


def index_built(stats_file: Path, profile: str, volume_created: str | None) -> bool:
    """The index counts as built when the indexer's stats exist and name this profile, and the database volume exists
    and is older than the stats: a volume removed or reset since leaves stats that describe a database that is gone. A
    creation time that cannot be read is trusted."""
    try:
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        written = datetime.fromtimestamp(stats_file.stat().st_mtime, UTC)
    except (OSError, ValueError):
        return False
    if not isinstance(stats, dict) or stats.get("profile") != profile or not volume_created:
        return False
    try:
        created = datetime.fromisoformat(volume_created.strip())
    except ValueError:
        return True
    return created.replace(tzinfo=created.tzinfo or UTC) <= written


def gather_down(root: Path, rag: Path, hardware) -> DownFacts:
    pid_files = frozenset(p.stem for p in (rag / "pids").glob("*.pid"))
    return DownFacts(str(root), bool(hardware.docker.ok), pid_files)


def confirm_text(steps: list[Step], profile: str = "example", mode: str = "native") -> str | None:
    """The downloads of a plan, one per line, with the total; None when there are none. A profile that reads PDFs adds
    the models its indexer fetches by itself; in native mode the total also warns about the servers' Python environments."""
    downloads = [s for s in steps if s.is_download]
    if not downloads:
        return None
    lines = [f"  {s.text}" for s in downloads]
    total = sum(s.gb for s in downloads)
    more = (
        ["  and the docling layout and table models and the EasyOCR weights (fetched by the indexer at the first PDF)"]
        if profile != "example"
        else []
    )
    return "\n".join(
        [
            "This will download:",
            *lines,
            f"  total about {total:g} GB (sizes the cards state"
            + (
                "; the model servers' first start also installs their Python environments)" if mode == "native" else ")"
            ),
            *more,
        ]
    )
