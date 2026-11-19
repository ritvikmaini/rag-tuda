"""The error reference on every log line: ref=<first 8 characters of the turn id> from a ContextVar through a logging filter,
into asyncio.to_thread, langchain_core's run_in_executor (LangGraph's sync nodes) and every line of a request, never the next
request. A raw loop.run_in_executor loses it: the product has none."""

import ast
import asyncio
import contextvars
import io
import logging
import types
from pathlib import Path
from typing import TypedDict

from fastapi.testclient import TestClient
from langchain_core.runnables.config import run_in_executor
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr

from assistant_api.api import chat as chat_route
from assistant_api.api.deps import verify_token
from assistant_api.app import create_app
from assistant_api import runtime
from assistant_api.logref import REF, ClassOnlyFormatter, RefFilter, set_ref
from assistant_api.sse import frames as fr

ROOT = Path(__file__).resolve().parents[2]
TURN = "A1B2C3D4-0000-4000-8000-000000000001"
T1, T2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"


class Collect(logging.Handler):
    """A handler that keeps (ref, message) of every record it receives, with the filter the gunicorn config puts on its own."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.addFilter(RefFilter())
        self.seen: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.seen.append((record.ref, record.getMessage()))  # type: ignore[attr-defined]


def collecting():
    handler, root = Collect(), logging.getLogger()
    root.addHandler(handler)
    old = root.level
    root.setLevel(logging.INFO)
    return handler, lambda: (root.removeHandler(handler), root.setLevel(old))


def log(label: str) -> None:
    logging.getLogger("test.logref").info(label)


def test_a_line_outside_a_request_carries_a_dash() -> None:
    handler, undo = collecting()
    try:
        contextvars.copy_context().run(log, "outside")
    finally:
        undo()
    assert handler.seen == [("-", "outside")]


def test_set_ref_keeps_the_first_8_characters() -> None:
    def inside() -> str:
        set_ref(TURN)
        return REF.get()

    assert contextvars.copy_context().run(inside) == "a1b2c3d4"
    assert REF.get() == "-"


def test_lines_from_worker_threads_carry_the_ref() -> None:
    class State(TypedDict):
        n: int

    def node(state: State) -> State:  # a sync node: LangGraph runs it through langchain_core's run_in_executor
        log("sync node")
        return {"n": state["n"] + 1}

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    compiled = graph.compile()

    async def request() -> None:
        set_ref(TURN)
        log("loop")
        await asyncio.to_thread(log, "to_thread")
        await run_in_executor(None, log, "langchain run_in_executor")
        await compiled.ainvoke({"n": 0})

    handler, undo = collecting()
    try:
        asyncio.run(request())
    finally:
        undo()
    mine = [
        (ref, msg)
        for ref, msg in handler.seen
        if msg in {"loop", "to_thread", "langchain run_in_executor", "sync node"}
    ]
    assert [m for _, m in mine] == ["loop", "to_thread", "langchain run_in_executor", "sync node"]
    assert {ref for ref, _ in mine} == {"a1b2c3d4"}


def test_a_raw_executor_call_loses_the_ref() -> None:
    async def request() -> None:
        set_ref(TURN)
        await asyncio.get_running_loop().run_in_executor(None, log, "raw")

    handler, undo = collecting()
    try:
        asyncio.run(request())
    finally:
        undo()
    assert handler.seen == [("-", "raw")]  # the hazard the next test keeps out of the product


def test_no_raw_executor_on_the_request_path() -> None:
    found = []
    for base in ("apps/assistant_api/src", "packages/rag_core/src"):
        for path in sorted((ROOT / base).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Attribute) and node.attr in {"run_in_executor", "ThreadPoolExecutor"}:
                    found.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.attr}")
                if isinstance(node, ast.Name) and node.id == "ThreadPoolExecutor":
                    found.append(f"{path.relative_to(ROOT)}:{node.lineno} ThreadPoolExecutor")
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "threading"
                    and any(a.name == "Thread" for a in node.names)
                ):
                    found.append(f"{path.relative_to(ROOT)}:{node.lineno} Thread")
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "Thread"
                    and getattr(node.value, "id", "") == "threading"
                ):
                    found.append(f"{path.relative_to(ROOT)}:{node.lineno} threading.Thread")
    assert found == []


def test_every_line_of_a_question_carries_its_ref(monkeypatch) -> None:
    async def pipeline(rt, **kwargs):
        log("pipeline loop " + kwargs["turn"][:2])
        await asyncio.to_thread(log, "pipeline thread " + kwargs["turn"][:2])
        yield fr.Chunk(content="a")

    monkeypatch.setattr(chat_route, "pipeline", pipeline)
    app = create_app()
    app.dependency_overrides[verify_token] = lambda: None
    app.state.runtime = types.SimpleNamespace(
        settings=types.SimpleNamespace(
            card=types.SimpleNamespace(timeouts=types.SimpleNamespace(deadline_s=5, idle_s=5)),
            overlay=types.SimpleNamespace(
                server=types.SimpleNamespace(token=SecretStr("t")),
                http=types.SimpleNamespace(cors_domain="example.org"),
            ),
            profile=types.SimpleNamespace(guard=types.SimpleNamespace(max_question_chars=2000)),
        ),
        handler=None,
        turns=None,
        messages_for=lambda lang: {"guard.error": "Fehler"},
    )
    client = TestClient(app)  # without `with`: no lifespan
    handler, undo = collecting()
    try:
        for turn in (T1, T2):
            handler.seen.clear()
            reply = client.get("/api/chat", params={"question": "Frage", "conversation_id": "c-ref", "turn_id": turn})
            assert reply.status_code == 200
            mine = [(ref, msg) for ref, msg in handler.seen if msg.startswith(("pipeline", "[Turn]"))]
            assert {ref for ref, _ in mine} == {turn[:8]}  # only its own ref: nothing leaks from the request before
            assert [m for _, m in mine if m.startswith("pipeline")] == [
                f"pipeline loop {turn[:2]}",
                f"pipeline thread {turn[:2]}",
            ]
            turn_lines = [msg for _, msg in mine if msg.startswith("[Turn]")]
            assert len(turn_lines) == 1 and turn_lines[0].startswith("[Turn] outcome=completed")
    finally:
        undo()
        app.dependency_overrides.clear()


def test_a_reload_started_by_a_question_logs_ref_dash(monkeypatch) -> None:
    """A task the reload starts (psycopg_pool's workers, retire's close_later) copies the context of the question that
    triggered it; the reload runs under ref=- so those tasks never carry that turn's ref, and the question keeps its own."""
    tasks: list[asyncio.Task] = []

    async def fake_reload(rt) -> None:
        log("reload line")

        async def pool_worker() -> None:
            log("pool task")

        tasks.append(asyncio.create_task(pool_worker()))

    monkeypatch.setattr(runtime, "reload", fake_reload)
    rt = types.SimpleNamespace(reload_lock=asyncio.Lock(), chain=None, last_reload_attempt=float("-inf"))

    async def request() -> None:
        set_ref(TURN)
        await runtime.try_reload(rt, quiet=True)  # type: ignore[arg-type]
        await asyncio.gather(*tasks)
        log("after")

    handler, undo = collecting()
    try:
        asyncio.run(request())
    finally:
        undo()
    assert handler.seen == [("-", "reload line"), ("-", "pool task"), ("a1b2c3d4", "after")]


def test_a_logged_exception_prints_its_class_and_frames_but_never_a_message() -> None:
    """Exception messages carry input values (pydantic input_value=, psycopg DETAIL, a tsquery); every exc_info line passes
    the one formatter on the handler, chained exceptions included."""
    out = io.StringIO()
    handler = logging.StreamHandler(out)
    handler.setFormatter(ClassOnlyFormatter(fmt="%(levelprefix)s %(message)s"))
    logger = logging.getLogger("test.logref.exc")
    logger.propagate = False
    logger.addHandler(handler)
    cause, message = "kanarie-" + "cause", "kanarie-" + "message"  # not in the source line the frame prints
    try:
        try:
            try:
                raise ValueError(cause)
            except ValueError as inner:
                raise RuntimeError(message) from inner
        except RuntimeError:
            logger.error("pipeline failed", exc_info=True)
    finally:
        logger.removeHandler(handler)
    text = out.getvalue()
    assert "kanarie" not in text
    assert "pipeline failed" in text and "test_logref.py" in text and text.rstrip().endswith("RuntimeError")
