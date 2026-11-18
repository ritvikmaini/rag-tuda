"""The assistant's runtime: built once by the lifespan, read by the routes from app.state. Start-up, the
reload on a request that finds no graph and the self-heal share one path: the new index handles and graph first, then the old engines closed
after the request deadline."""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from rag_core.adapters.chat_openai import ChatOpenAIModel, build_chat_openai
from rag_core.adapters.embedding_http import EmbeddingClient
from rag_core.adapters.http import get_http_client
from rag_core.adapters.pg_index import IndexHandles, open_index
from rag_core.adapters.reranker_http import RerankerHttp
from rag_core.adapters.vllm_tokenize import VllmTokenCounter
from rag_core.domain.access import AllowAll
from rag_core.domain.resilience import dispose_engines
from rag_core.services.prompts import PromptService
from rag_core.services.retrieval import RetrievalService

from assistant_api.graph.build import build_graph
from assistant_api.graph.state import GraphDeps
from assistant_api.graph.tracing import make_handler
from assistant_api.logref import REF
from assistant_api.settings import AssistantSettings

RELOAD_RETRY_S = 60.0


@dataclass
class Runtime:
    settings: AssistantSettings
    langfuse: Any | None
    handler: Any | None
    prompts: PromptService
    answer_llm: Any
    analysis: ChatOpenAIModel
    counter: VllmTokenCounter
    embedder: EmbeddingClient
    slot: asyncio.Semaphore
    turns: Any = None  # the TurnStore the lifespan sets (Any: no chat import here, no None check per use)
    generation: int | None = None  # the live index generation the loaded index was read at
    index: IndexHandles | None = None
    chain: Any | None = None
    reload_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    retiring: dict[asyncio.Task[None], tuple[Any, ...]] = field(default_factory=dict)  # old pools waiting to close
    last_reload_attempt: float = float("-inf")

    @property
    def collection(self) -> str:
        return self.settings.profile.collections[0].name

    @property
    def config_hash(self) -> str:
        """The answer-affecting settings of this process (the profile), so a turn names the configuration it ran with."""
        return hashlib.sha256(self.settings.profile.model_dump_json().encode("utf-8")).hexdigest()[:16]

    def run_info(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "index_generation": self.generation,
            "config_hash": self.config_hash,
            "prompts": self.prompts.versions(),
        }

    @property
    def messages(self) -> dict[str, str]:
        """The profile's messages as a flat mapping ("guard.busy", …): the form the guard's code table reads."""
        return {
            f"{group}.{key}": text
            for group, texts in self.settings.messages.model_dump(exclude={"labels"}).items()
            for key, text in texts.items()
        }


def build_runtime(settings: AssistantSettings, langfuse: Any | None, authenticated: bool) -> Runtime:
    profile, overlay = settings.profile, settings.overlay
    g = profile.generation
    llm_url = overlay.model_servers.llm_url

    def chat(model: str, temperature: float) -> Any:
        return build_chat_openai(
            model=model,
            temperature=temperature,
            base_url=llm_url,
            max_tokens=g.max_tokens,
            read_timeout_s=overlay.timeouts.llm_read_s,
            disable_thinking=g.disable_thinking,
            sampling=g.sampling.model_dump(),
        )

    embedding = profile.retrieval.embedding
    return Runtime(
        settings=settings,
        langfuse=langfuse,
        handler=make_handler(authenticated),
        prompts=PromptService(settings.prompts),
        answer_llm=chat(g.model, g.temperature),
        analysis=ChatOpenAIModel(chat(g.analysis_model, g.analysis_temperature), g.analysis_max_tokens),
        counter=VllmTokenCounter(llm_url, g.model, profile.guard.context_window),
        embedder=EmbeddingClient(
            overlay.model_servers.embedding_url,
            query_prefix=embedding.query_prefix,
            passage_prefix=embedding.passage_prefix,
        ),
        slot=asyncio.Semaphore(overlay.compute.slots),
    )


def graph_deps(rt: Runtime, index: IndexHandles) -> GraphDeps:
    """What the graph's nodes close over, for the index just opened (also python -m assistant_api.scope_scores)."""
    overlay = rt.settings.overlay
    return GraphDeps(
        profile=rt.settings.profile,
        messages=rt.settings.messages,
        prompts=rt.prompts,
        analysis=rt.analysis,
        answer_llm=rt.answer_llm,
        counter=rt.counter,
        timeouts=overlay.timeouts,
        retrieval=RetrievalService(
            rt.embedder,
            index.vector,
            index.full_text,
            index.sections,
            RerankerHttp(overlay.model_servers.reranker_url),
            rt.settings.profile.retrieval,
            rt.settings.profile.ranking,
            overlay.timeouts,
            AllowAll(),
        ),
    )


async def reload(rt: Runtime) -> None:
    """Loads the retriever, the section lookup and the graph for the live collection."""
    logging.info("Attempting to reload data and re-initialize retrievers...")
    old = rt.index
    old_generation = rt.generation
    logging.info(f"Using collection: {rt.collection} (views {rt.collection}_chunks, {rt.collection}_sections)")
    index = await open_index(rt.collection, rt.settings.profile, rt.settings.overlay.database)
    logging.info(f"Successfully loaded retriever with FTS for collection: {rt.collection}")
    logging.info(f"Successfully loaded PostgreSQL parent lookup for collection: {rt.collection}")
    rt.index = index
    overlay = rt.settings.overlay
    try:
        rt.generation = await rt.turns.live_generation(rt.collection) if rt.turns is not None else None
        logging.info(f"[Turns] live index generation {rt.generation}")
        deps = graph_deps(rt, index)
        rt.chain = await asyncio.to_thread(build_graph, deps)
    except BaseException:
        # no graph uses the new index: put the old one back and close the new pool (also at start-up)
        rt.index, rt.generation = old, old_generation
        await dispose_engines(*index.engines)
        raise
    logging.info("LangGraph chain has been re-created with new data.")
    if old is not None:
        retire(rt, old.engines, overlay.timeouts.request_deadline_s)


def retire(rt: Runtime, engines: tuple[Any, ...], grace_s: float) -> None:
    """Closes a previous load's pool once every request that may still hold it has passed its deadline: psycopg_pool's close()
    fails every later getconn with PoolClosed (SQLAlchemy's dispose() let a request open new connections). Held on the runtime,
    so shutdown closes it at once (stop_retiring)."""

    async def close_later() -> None:
        await asyncio.sleep(grace_s)
        closed = await dispose_engines(*engines)
        if closed:
            logging.info(f"[Reload] disposed {closed} previous database engine(s)")

    task = asyncio.create_task(close_later())
    rt.retiring[task] = engines
    task.add_done_callback(lambda done: rt.retiring.pop(done, None))


def stop_retiring(rt: Runtime) -> list[Any]:
    """Shutdown: cancels the waiting closes and hands their engines to the one dispose_engines call."""
    engines = [engine for waiting in rt.retiring.values() for engine in waiting]
    for task in list(rt.retiring):
        task.cancel()
    rt.retiring.clear()
    return engines


async def try_reload(rt: Runtime, quiet: bool = False) -> bool:
    """One reload attempt (at most every RELOAD_RETRY_S seconds) when the graph is missing, e.g. after a failed start-up load.
    quiet drops its own lines: heal adds its [Heal] lines only on state changes."""
    async with rt.reload_lock:
        if rt.chain is not None:
            return True
        if time.monotonic() - rt.last_reload_attempt < RELOAD_RETRY_S:
            return False
        rt.last_reload_attempt = time.monotonic()
        token = REF.set(
            "-"
        )  # tasks the reload starts (the pool's workers, retire) copy this context: not the question's ref
        try:
            await reload(rt)
            if not quiet:
                logging.info("[Startup] data and retrievers loaded on request")
        except Exception as e:
            if not quiet:
                logging.warning(f"[Startup] reload on request failed: {e}")
        finally:
            REF.reset(token)
        return rt.chain is not None


def _logged_in(client: Any) -> bool:
    try:
        return bool(client.auth_check())
    except Exception:
        return False


async def heal(rt: Runtime) -> None:
    """The background task of the lifespan: every RELOAD_RETRY_S it retries what the start left broken, the index load
    (while rt.chain is None) and the Langfuse login (a client whose login failed has no handler), and returns when both hold.
    Logs a [Heal] line when the retrying starts and one per part healed (each failed attempt still logs the reload's and
    the pool's own lines). An unexpected error in an attempt is logged and the next attempt follows: the task never ends
    silently."""

    def broken() -> list[str]:
        return (["index"] if rt.chain is None else []) + (
            ["langfuse"] if rt.langfuse is not None and rt.handler is None else []
        )

    if parts := broken():
        logging.warning("[Heal] retrying every 60 s: " + ", ".join(parts))
    while parts := broken():
        await asyncio.sleep(RELOAD_RETRY_S)
        try:
            if "index" in parts and await try_reload(rt, quiet=True):
                logging.info("[Heal] index loaded")
            if "langfuse" in parts and await asyncio.to_thread(_logged_in, rt.langfuse):
                rt.handler = make_handler(True)
                logging.info("[Heal] Langfuse authenticated, tracing on")
        except Exception:  # CancelledError is a BaseException: it still ends the task
            logging.exception("[Heal] attempt failed unexpectedly, retrying")


async def _report(url: str, expected: tuple[str, str] | None) -> str:
    try:
        reply = await get_http_client().get(url, timeout=2.0)
    except Exception as e:
        return f"unavailable: {type(e).__name__}"
    if reply.status_code != 200:
        return f"unavailable: HTTP {reply.status_code}"
    if expected is None:
        return "ok"
    info = reply.json()
    found = (str(info.get("model")), str(info.get("revision")))
    return "ok" if found == expected else f"mismatch: {found[0]}@{found[1]}"


async def model_reports(rt: Runtime) -> dict[str, str]:
    """The model servers' /info and vLLM's /v1/models, reported, never gated on."""
    servers, retrieval = rt.settings.overlay.model_servers, rt.settings.profile.retrieval
    return {
        "embedding": await _report(
            f"{servers.embedding_url}/info", (retrieval.embedding.model, retrieval.embedding.revision)
        ),
        "reranker": await _report(f"{servers.reranker_url}/info", (retrieval.rerank.model, retrieval.rerank.revision)),
        "llm": await _report(f"{servers.llm_url}/models", None),
    }


async def readiness(rt: Any) -> dict[str, Any]:
    """Ready when the graph is loaded and the section view answers."""
    checks = {
        "index": "ok" if rt.chain is not None and rt.index is not None else "missing",
        "database": "not checked",
    }
    if rt.index is not None:
        try:
            await asyncio.wait_for(rt.index.sections.ping(), timeout=2.0)
            checks["database"] = "ok"
        except Exception as e:
            checks["database"] = f"error: {type(e).__name__}"
    reports = await model_reports(rt)
    return {
        "ready": checks["index"] == "ok" and checks["database"] == "ok",
        "checks": checks,
        "reports": reports,
        "git_commit": os.environ.get("GIT_COMMIT") or None,
    }
