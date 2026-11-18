"""The embedding server.

POST /embed {"texts": [str]} -> {"embeddings": [[float]]} in input order, through a bounded queue and one batch worker (503
when the queue stays full for 5 s, 504 after 60 s); GET /info (model, revisions, dimension, dtype, max_seq_length), /readyz
(the start-up reference check), /healthz, /metrics (queue gauges). The settings come from
the environment the Dockerfile sets; the model loads in the lifespan from its pinned snapshot in the Hugging Face cache,
never from the network; nothing happens at import. A torch.AcceleratorError (the CUDA context is gone) answers 500, turns
/readyz 503 and stops the process (SIGTERM, os._exit after FATAL_GRACE_S); Docker restarts it. With REQUIRE_CUDA=1 the model
does not load on the CPU. Run: uvicorn --factory embedding_server.main:create_app"""

import asyncio
import json
import logging
import math
import os
import re
import signal
import sys
import threading
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("embedding_server")

COMMIT = re.compile(r"[0-9a-f]{40}")
MAX_SEQ_LENGTH = 512  # children stay below 512 tokens; the model was evaluated at 512 (arctic allows 8192)
MAX_QUEUE_SIZE = 500  # pending requests before /embed answers 503
MAX_BATCH_SIZE = 32  # requests per batch
BATCH_TIMEOUT = 0.05  # seconds to collect a batch
QUEUE_PUT_TIMEOUT = 5.0
RESULT_TIMEOUT = 60.0
READY_COSINE = 0.9999  # fp16 output is not bit-stable, so no exact checksum
REFERENCE = Path(__file__).with_name("reference.json")

FATAL_GRACE_S = 30.0  # after SIGTERM, a daemon timer ends the process this much later if the shutdown hangs


def is_fatal(e: BaseException) -> bool:
    """torch.AcceleratorError (a CUDA context that is gone: every later call fails the same way) and nothing else: an
    OutOfMemoryError or any other RuntimeError may pass. No import at module level (nothing happens at import): a torch error
    implies torch is loaded."""
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(e, getattr(torch, "AcceleratorError", ()))


def stop_self(grace_s: float = FATAL_GRACE_S) -> None:
    """SIGTERM to this process: uvicorn (PID 1) sends the failing reply, shuts down and exits; Docker (unless-stopped)
    restarts the container. A call hanging on the dead context keeps the shutdown (or the executor's join at exit) waiting,
    so a daemon timer ends the process grace_s later."""
    timer = threading.Timer(grace_s, os._exit, (1,))
    timer.daemon = True
    timer.start()
    os.kill(os.getpid(), signal.SIGTERM)


@dataclass(frozen=True)
class Settings:
    model: str
    revision: str
    code_revision: str
    config_kwargs: dict[str, Any]
    release_after_texts: int = 64  # an index build's batches hand cached GPU memory back; an answer embeds 1-6 queries

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        names = ("EMBEDDING_MODEL", "EMBEDDING_REVISION", "EMBEDDING_CODE_REVISION", "EMBEDDING_CONFIG_KWARGS")
        missing = [name for name in names if not env.get(name)]
        if missing:
            raise ValueError("missing in the environment: " + ", ".join(missing))
        for name in ("EMBEDDING_REVISION", "EMBEDDING_CODE_REVISION"):
            if not COMMIT.fullmatch(env[name]):
                raise ValueError(name + " is not a 40-character commit hash")
        config_kwargs = json.loads(env["EMBEDDING_CONFIG_KWARGS"])
        if not isinstance(config_kwargs, dict):
            raise ValueError("EMBEDDING_CONFIG_KWARGS is not a JSON object")
        return cls(
            env["EMBEDDING_MODEL"],
            env["EMBEDDING_REVISION"],
            env["EMBEDDING_CODE_REVISION"],
            config_kwargs,
            int(env.get("EMBED_RELEASE_AFTER_TEXTS", "64")),
        )


@dataclass(frozen=True)
class Loaded:
    embed: Callable[[list[str]], list[list[float]]]
    dimension: int
    dtype: str
    device: str
    empty_cache: Callable[[], None]


def model_kwargs(settings: Settings, device: str) -> dict[str, Any]:
    """SentenceTransformer's arguments: the remote code (in arctic-embed-m-v2.0's own snapshot) is trusted only at its
    pinned revision; the config overrides keep it off CUDA-only libraries (no xformers in this image); nothing comes from
    the network."""
    return {
        "device": device,
        "trust_remote_code": True,
        "local_files_only": True,
        "config_kwargs": {**settings.config_kwargs, "code_revision": settings.code_revision},
        "model_kwargs": {"code_revision": settings.code_revision},
    }


def release_cached_memory(n_texts: int, device: str, empty_cache: Callable[[], None], threshold: int) -> bool:
    """After a large batch (an index build), hand PyTorch's cached GPU blocks back to the card: the
    caching allocator otherwise keeps the build's peak until the next restart, while the reranker and the language
    model share the card. Returns whether it released; a failing release never fails the batch."""
    if device != "cuda" or n_texts < threshold:
        return False
    try:
        empty_cache()
        return True
    except Exception:
        return False


def load_model(settings: Settings) -> Loaded:
    """The model from its pinned snapshot (a missing snapshot fails the start), through langchain-huggingface
    (its embed_documents turns newlines into spaces), max_seq_length 512, fp16 on CUDA (a failed conversion keeps fp32 and
    says so)."""
    import torch
    from huggingface_hub import snapshot_download
    from langchain_huggingface import HuggingFaceEmbeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and os.environ.get("REQUIRE_CUDA") == "1":
        raise RuntimeError("REQUIRE_CUDA=1 but CUDA is not available: no CPU fallback, the container restarts")
    logger.info("Embedding source: local HuggingFace (device: %s)", device)
    logger.info(
        "Loading embedding model: %s at %s (code %s)", settings.model, settings.revision, settings.code_revision
    )
    logger.info("Embedding model config overrides: %s", settings.config_kwargs)
    path = snapshot_download(settings.model, revision=settings.revision, local_files_only=True)
    embeddings = HuggingFaceEmbeddings(model_name=path, model_kwargs=model_kwargs(settings, device))
    model = embeddings._client
    model.max_seq_length = min(model.max_seq_length or MAX_SEQ_LENGTH, MAX_SEQ_LENGTH)
    if device == "cuda":
        try:
            model.half()
            logger.info("Using fp16 for embedding model (CUDA)")
        except Exception as e:
            logger.warning("fp16 conversion failed, embedding model stays fp32: %s", e)
    logger.info("Embedding model loaded successfully.")
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    return Loaded(
        embeddings.embed_documents, model.get_sentence_embedding_dimension(), dtype, device, torch.cuda.empty_cache
    )


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def reference_check(loaded: Loaded, settings: Settings, reference: Mapping[str, Any]) -> dict[str, Any]:
    """Once at start, never per probe (the GPU must not compute for a healthcheck): the reference texts against the stored
    reference vectors. Ready when the reference belongs to this model and revision and every cosine is at
    least 0.9999."""
    if (reference["model"], reference["revision"]) != (settings.model, settings.revision):
        return {
            "ready": False,
            "detail": "the reference was taken with %s at %s" % (reference["model"], reference["revision"]),
        }
    vectors = loaded.embed(list(reference["texts"]))
    if len(vectors) != len(reference["vectors"]):
        return {
            "ready": False,
            "detail": "%d vectors for %d reference texts" % (len(vectors), len(reference["vectors"])),
        }
    lowest = min(cosine(a, b) for a, b in zip(vectors, reference["vectors"]))
    return {"ready": lowest >= READY_COSINE, "texts": len(vectors), "lowest_cosine": lowest}


class EmbeddingRequest(BaseModel):
    texts: list[str]


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]


@dataclass(frozen=True)
class Job:
    texts: list[str]
    future: asyncio.Future


async def batch_worker(
    queue: asyncio.Queue, loaded: Loaded, settings: Settings, *, on_fatal: Callable[[], None]
) -> None:
    """Waits for a request, collects more for BATCH_TIMEOUT (at most MAX_BATCH_SIZE), embeds all their texts in one call and
    hands each request its slice."""
    loop = asyncio.get_running_loop()
    while True:
        batch = [await queue.get()]
        deadline = loop.time() + BATCH_TIMEOUT
        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        texts = [text for job in batch for text in job.texts]
        logger.info("Processing batch of %d requests (%d total texts)", len(batch), len(texts))
        try:
            vectors = await asyncio.to_thread(loaded.embed, texts)
            if release_cached_memory(len(texts), loaded.device, loaded.empty_cache, settings.release_after_texts):
                logger.debug("Released cached GPU memory after %d texts", len(texts))
            start = 0
            for job in batch:
                if not job.future.done():  # a request that timed out (504) cancelled its future
                    job.future.set_result(vectors[start : start + len(job.texts)])
                start += len(job.texts)
        except Exception as e:
            logger.exception("Error during batch embedding: %s", e)
            if is_fatal(e):
                on_fatal()
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(e)


def create_app(
    settings: Settings | None = None,
    load: Callable[[Settings], Loaded] = load_model,
    reference_path: Path = REFERENCE,
    stop: Callable[[], None] = stop_self,
) -> FastAPI:
    logging.basicConfig(level=logging.INFO)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        current = settings or Settings.from_env()
        loaded = load(current)
        app.state.readiness = reference_check(loaded, current, json.loads(reference_path.read_text(encoding="utf-8")))
        logger.info("Reference check: %s", app.state.readiness)
        app.state.settings, app.state.loaded = current, loaded
        app.state.queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        app.state.stopping = False
        worker = asyncio.create_task(batch_worker(app.state.queue, loaded, current, on_fatal=fatal))
        logger.info(
            "Started embedding batch worker (max_batch_size=%d, batch_timeout=%ss)", MAX_BATCH_SIZE, BATCH_TIMEOUT
        )
        yield
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        logger.info("Embedding service shutdown complete.")

    app = FastAPI(title="Embedding Service", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def fatal() -> None:
        """Once: /readyz turns 503 (a dead context never serves again), then the process stops itself."""
        if app.state.stopping:
            return
        app.state.stopping = True
        app.state.readiness = {"ready": False, "detail": "fatal CUDA error (AcceleratorError): the server restarts"}
        logger.error("Fatal CUDA error: stopping the server, Docker restarts it")
        stop()

    @app.post("/embed", response_model=EmbeddingResponse)
    async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
        if not request.texts:
            return EmbeddingResponse(embeddings=[])
        job = Job(request.texts, asyncio.get_running_loop().create_future())
        try:
            await asyncio.wait_for(app.state.queue.put(job), timeout=QUEUE_PUT_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Embedding service overloaded. Please retry later.")
        try:
            return EmbeddingResponse(embeddings=await asyncio.wait_for(job.future, timeout=RESULT_TIMEOUT))
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Embedding request timed out.")
        except Exception as e:
            logger.exception("Error during embedding generation.")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/info")
    async def info() -> dict[str, Any]:
        current, loaded = app.state.settings, app.state.loaded
        return {
            "model": current.model,
            "revision": current.revision,
            "code_revision": current.code_revision,
            "dimension": loaded.dimension,
            "dtype": loaded.dtype,
            "max_seq_length": MAX_SEQ_LENGTH,
        }

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        readiness = app.state.readiness
        status = "ready" if readiness["ready"] else "not ready"
        return JSONResponse({"status": status, **readiness}, status_code=200 if readiness["ready"] else 503)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return {
            "queue_size": app.state.queue.qsize(),
            "max_queue_size": MAX_QUEUE_SIZE,
            "batch_size": MAX_BATCH_SIZE,
            "batch_timeout_ms": BATCH_TIMEOUT * 1000,
            "device": app.state.loaded.device,
        }

    return app
