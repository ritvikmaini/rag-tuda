"""The reranker server.

POST /v2/rerank {"queries": [str], "documents": [str]} -> {"scores": [float]} aligned to the documents, each the highest
over the queries (multi-query). One job on the GPU at a time: a waiting request gets 503 after MULTI_QUERY_WAIT_S, and a job
stops between queries when its client has left. GET /info (model, revision, max_length, dtype, device, mode), /readyz (the
start-up reference pair, from reference.<device>.json when it exists, else reference.json), /healthz. The settings come
from the environment the Dockerfile sets (MAX_LENGTH tokens per pair; RERANK_MODE cross_encoder, or qwen_yes_no for a
Qwen3-Reranker scored by the probability of "yes"); the model loads in the lifespan from its pinned snapshot, never from the
network; nothing happens at import. The device is RAG_DEVICE (cuda, mps or cpu; unset: the first available in that order),
and a device that is not available stops the start. A torch.AcceleratorError (the CUDA context is gone) answers 500, turns
/readyz 503 and stops the process (SIGTERM, os._exit after FATAL_GRACE_S); Docker restarts it.
Run: uvicorn --factory reranker_server.main:create_app"""

import asyncio
import json
import logging
import os
import re
import signal
import sys
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("reranker_server")

COMMIT = re.compile(r"[0-9a-f]{40}")
MODES = ("cross_encoder", "qwen_yes_no")
READY_DIFFERENCE = 1e-3  # the reference pair's score may differ by at most this
REFERENCE = Path(__file__).with_name("reference.json")

FATAL_GRACE_S = 30.0  # after SIGTERM, a daemon timer ends the process this much later if the shutdown hangs

DEVICES = ("cuda", "mps", "cpu")


def pick_device(env: Mapping[str, str], cuda_ok: bool, mps_ok: bool) -> str:
    """RAG_DEVICE (cuda, mps or cpu) when set: a device that is not available is an error, never a fallback. Unset: cuda, then
    mps, then cpu."""
    available = {"cuda": cuda_ok, "mps": mps_ok, "cpu": True}
    wanted = env.get("RAG_DEVICE", "").strip().lower()
    if not wanted:
        return next(device for device in DEVICES if available[device])
    if wanted not in available:
        raise ValueError("RAG_DEVICE=%s is not one of %s" % (wanted, ", ".join(DEVICES)))
    if not available[wanted]:
        raise RuntimeError(
            "RAG_DEVICE=%s but %s is not available: no fallback, the container restarts" % (wanted, wanted)
        )
    return wanted


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


def positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    """env[name] as an integer of at least 1; unset or empty: the default."""
    if not env.get(name):
        return default
    value = int(env[name])
    if value < 1:
        raise ValueError(name + " is not a positive integer")
    return value


@dataclass(frozen=True)
class Settings:
    model: str
    revision: str
    predict_batch_size: int = 32  # CrossEncoder.predict's default
    wait_s: float = 20.0  # a request waiting for the GPU job gives up after this (the assistant's timeout is 15 s)
    max_length: int = 1024  # tokens per (query, passage) pair; a parent section is ~500-700 tokens
    mode: str = "cross_encoder"  # or qwen_yes_no: a generative model whose yes/no logits are the score

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        missing = [name for name in ("RERANKER_MODEL", "RERANKER_REVISION") if not env.get(name)]
        if missing:
            raise ValueError("missing in the environment: " + ", ".join(missing))
        if not COMMIT.fullmatch(env["RERANKER_REVISION"]):
            raise ValueError("RERANKER_REVISION is not a 40-character commit hash")
        mode = env.get("RERANK_MODE") or "cross_encoder"
        if mode not in MODES:
            raise ValueError("RERANK_MODE=%s is not one of %s" % (mode, ", ".join(MODES)))
        return cls(
            env["RERANKER_MODEL"],
            env["RERANKER_REVISION"],
            int(env.get("RERANKER_PREDICT_BATCH_SIZE", "32")),
            float(env.get("MULTI_QUERY_WAIT_S", "20")),
            positive_int(env, "MAX_LENGTH", 1024),
            mode,
        )


@dataclass(frozen=True)
class Loaded:
    predict: Callable[..., Sequence[float]]
    dtype: str
    device: str


def length_sorted_predict(predict, pairs, batch_size):
    """Scores in the input order from one predict call over the pairs sorted by length, shortest first: the
    CrossEncoder (sentence-transformers 4.1.0) pads every batch to its longest pair
    and keeps the order it is given, and one request mixes ~450-character passages with 2,000-character sections.
    Padding is masked, so a pair's score does not depend on its batch; only the wasted work goes."""
    if not pairs:
        return []
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    scores = predict([pairs[i] for i in order], batch_size=batch_size, convert_to_numpy=True)
    out = [0.0] * len(pairs)
    for position, i in enumerate(order):
        out[i] = float(scores[position])
    return out


QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_PAIR = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def qwen_yes_no_predict(tokenizer, model, max_length):
    """The predict function of a Qwen3-Reranker, a generative model that answers "yes" or "no" to whether the document meets
    the query: each pair goes into the model card's prompt, left-padded so the answer position is the last, and the score is
    the softmax of the "no" and "yes" logits there, taken at "yes" (the probability of yes, between 0 and 1). The pairs are cut
    to max_length tokens with the prompt's own tokens kept. Same call as CrossEncoder.predict."""
    import torch

    tokenizer.padding_side = "left"  # whatever the tokenizer was configured with: the answer position must be the last
    no, yes = tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes")
    prefix = tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)

    def predict(pairs, batch_size=32, convert_to_numpy=True):
        scores = []
        for start in range(0, len(pairs), batch_size):
            texts = [
                QWEN_PAIR.format(instruction=QWEN_INSTRUCTION, query=query, document=document)
                for query, document in pairs[start : start + batch_size]
            ]
            inputs = tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_length - len(prefix) - len(suffix),
            )
            inputs["input_ids"] = [prefix + ids + suffix for ids in inputs["input_ids"]]
            batch = tokenizer.pad(inputs, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                last = model(**batch, logits_to_keep=1).logits[
                    :, -1, :
                ]  # only the last position's logits: no 150k-wide rows
            answers = torch.stack([last[:, no], last[:, yes]], dim=1).float()
            scores += torch.softmax(answers, dim=1)[:, 1].tolist()
        return scores

    return predict


def load_cross_encoder(path, settings, device):
    """(predict, torch module) of the CrossEncoder at path, fp16 on CUDA and MPS (a failed conversion keeps fp32 and says
    so), converted on the CPU before the move."""
    from sentence_transformers import CrossEncoder

    # on the CPU: s-t 4 would move the fp32 weights to the GPU in __init__, where the card has no room for them
    model = CrossEncoder(path, max_length=settings.max_length, device="cpu")
    if device != "cpu":
        try:
            model.model = model.model.half()  # fp16 halves the GPU memory (~2 GB vs ~3.5 GB for bge-reranker-v2-m3)
            logger.info("Converted reranker model to fp16 (%s)", device)
        except Exception as e:
            logger.warning("fp16 conversion failed, the reranker stays fp32: %s", e)
        model.model.to(device)  # the accelerator only ever holds the fp16 weights
    return model.predict, model.model


def load_qwen(path, settings, device):
    """(predict, torch module) of the Qwen3-Reranker at path, loaded in fp16 on CUDA and MPS, fp32 on the CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left", local_files_only=True)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, local_files_only=True).to(device).eval()
    return qwen_yes_no_predict(tokenizer, model, settings.max_length), model


def load_model(settings: Settings) -> Loaded:
    """The reranker from its pinned snapshot (a missing snapshot fails the start) on the picked device, max_length tokens per
    pair: the CrossEncoder, or with RERANK_MODE=qwen_yes_no a Qwen3-Reranker scored by its yes/no logits. fp16 on CUDA and
    MPS, fp32 on the CPU. No remote code."""
    import torch
    from huggingface_hub import snapshot_download

    device = pick_device(os.environ, torch.cuda.is_available(), torch.backends.mps.is_available())
    logger.info("Reranker source: local %s (device: %s)", settings.mode, device)
    logger.info("Loading reranking model: %s at %s", settings.model, settings.revision)
    path = snapshot_download(settings.model, revision=settings.revision, local_files_only=True)
    load = load_qwen if settings.mode == "qwen_yes_no" else load_cross_encoder
    predict, module = load(path, settings, device)
    logger.info("Reranking model loaded successfully.")
    return Loaded(predict, str(next(module.parameters()).dtype).removeprefix("torch."), device)


async def best_scores(
    loaded: Loaded,
    batch_size: int,
    queries: Sequence[str],
    documents: Sequence[str],
    disconnected: Callable[[], Awaitable[bool]],
    *,
    on_fatal: Callable[[], None],
) -> list[float] | None:
    """Each document's highest score over the queries, by position; one predict call per query, so a call's GPU memory is
    that of one query. Stops between queries when the client has left (None: it left before the first)."""
    best = None
    for query in queries:
        if await disconnected():
            logger.info("Multi-query rerank abandoned by the client; stopping early")
            break
        pairs = [(query, doc) for doc in documents]
        try:
            scores = await asyncio.to_thread(length_sorted_predict, loaded.predict, pairs, batch_size)
        except Exception as e:
            logger.exception("Error during multi-query reranking (%d documents): %s", len(documents), e)
            if is_fatal(e):
                on_fatal()
            raise HTTPException(status_code=500, detail="Reranking inference failed: %s" % e)
        best = scores if best is None else [max(b, s) for b, s in zip(best, scores)]
    return best


def reference_for(base: Path, device: str) -> Path:
    """reference.<device>.json beside the base reference when it exists, else the base: one stored reference serves every
    device class until a device's output moves outside the tolerance, and then it gets its own file."""
    specific = base.with_name("%s.%s%s" % (base.stem, device, base.suffix))
    return specific if specific.exists() else base


def reference_check(loaded: Loaded, settings: Settings, reference: Mapping[str, Any]) -> dict[str, Any]:
    """Once at start, never per probe (the GPU must not compute for a healthcheck): the reference pair against its stored
    score. Ready when the reference belongs to this model and revision and the score is within
    1e-3."""
    if (reference["model"], reference["revision"]) != (settings.model, settings.revision):
        return {
            "ready": False,
            "detail": "the reference was taken with %s at %s" % (reference["model"], reference["revision"]),
        }
    pair = [(reference["query"], reference["document"])]
    score = length_sorted_predict(loaded.predict, pair, settings.predict_batch_size)[0]
    difference = abs(score - reference["score"])
    return {"ready": difference <= READY_DIFFERENCE, "score": score, "difference": difference}


class RerankRequest(BaseModel):
    queries: list[str]
    documents: list[str]


class ScoresResponse(BaseModel):
    scores: list[float]


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
        app.state.readiness = reference_check(
            loaded, current, json.loads(reference_for(reference_path, loaded.device).read_text(encoding="utf-8"))
        )
        logger.info("Reference check: %s", app.state.readiness)
        app.state.settings, app.state.loaded = current, loaded
        app.state.stopping = False
        app.state.gate = asyncio.Semaphore(1)  # one job on the GPU at a time (two GPU consumers freeze the card)
        logger.info("Reranker service ready: one job at a time, a waiting request gives up after %ss", current.wait_s)
        yield
        logger.info("Reranker service shutdown complete.")

    app = FastAPI(title="Reranker Service", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def fatal() -> None:
        """Once: /readyz turns 503 (a dead context never serves again), then the process stops itself."""
        if app.state.stopping:
            return
        app.state.stopping = True
        app.state.readiness = {"ready": False, "detail": "fatal CUDA error (AcceleratorError): the server restarts"}
        logger.error("Fatal CUDA error: stopping the server, Docker restarts it")
        stop()

    @asynccontextmanager
    async def gpu_job():
        try:
            await asyncio.wait_for(app.state.gate.acquire(), timeout=app.state.settings.wait_s)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Reranker busy.")
        try:
            yield
        finally:
            app.state.gate.release()

    async def scored(body: RerankRequest, request: Request) -> list[float] | None:
        async with gpu_job():
            batch = app.state.settings.predict_batch_size
            return await best_scores(
                app.state.loaded, batch, body.queries, body.documents, request.is_disconnected, on_fatal=fatal
            )

    @app.post("/v2/rerank", response_model=ScoresResponse)
    async def rerank_v2(body: RerankRequest, request: Request) -> ScoresResponse:
        if not body.queries:
            raise HTTPException(status_code=422, detail="queries: at least one query is needed")
        if not body.documents:
            return ScoresResponse(scores=[])
        # None: the client left before the first query and reads no answer
        return ScoresResponse(scores=await scored(body, request) or [])

    @app.get("/info")
    async def info() -> dict[str, Any]:
        current = app.state.settings
        return {
            "model": current.model,
            "revision": current.revision,
            "max_length": current.max_length,
            "dtype": app.state.loaded.dtype,
            "device": app.state.loaded.device,
            "mode": current.mode,
        }

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        readiness = app.state.readiness
        status = "ready" if readiness["ready"] else "not ready"
        return JSONResponse({"status": status, **readiness}, status_code=200 if readiness["ready"] else 503)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
