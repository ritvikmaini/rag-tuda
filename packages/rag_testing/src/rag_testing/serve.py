"""The harness's fake servers: one module, one image (deploy/fakes/), one
compose service per fake, each started as `python -m rag_testing.serve <name> --port <port> [--root <folder>]`.

  llm        vllm:8000                 /v1/chat/completions (streamed or not), /tokenize; the per-test script
                                       (POST /_script), the request log (GET /_calls), POST /_reset
  embedder   embedding_service:8001    /embed
  reranker   reranker_service:8003     /v2/rerank

Never installed in a service image: import-linter's testing-dev-only contract keeps packages/ and apps/ from importing it."""

import argparse
import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from rag_testing import llm, scoring


def llm_app() -> FastAPI:
    app = FastAPI()
    rules: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/tokenize")
    async def tokenize(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return {"count": llm.count_tokens(body), "max_model_len": llm.MAX_MODEL_LEN}

    @app.post("/_script")
    async def script(body: dict[str, Any] = Body(...)) -> dict[str, int]:
        rules[:] = list(body.get("rules") or [])
        return {"rules": len(rules)}

    @app.post("/_reset")
    async def reset() -> dict[str, str]:
        rules.clear()
        calls.clear()
        return {"status": "ok"}

    @app.get("/_calls")
    async def recorded() -> list[dict[str, Any]]:
        return calls

    @app.post("/v1/chat/completions")
    async def chat(body: dict[str, Any] = Body(...)) -> Response:
        messages = list(body.get("messages") or [])
        kind = llm.purpose(messages)
        rule = llm.pick(rules, kind, messages)
        stream = bool(body.get("stream"))
        calls.append(
            {"purpose": kind, "question": llm.question(messages, kind), "messages": messages, "stream": stream}
        )
        if rule.get("fail"):
            return JSONResponse({"error": {"message": "scripted failure", "type": "server_error"}}, int(rule["fail"]))
        text = llm.reply(rule, kind, messages)
        first, between = float(rule.get("first_delay_s", 0)), float(rule.get("token_delay_s", 0))
        if not stream:
            await asyncio.sleep(first)
            return JSONResponse(llm.completion(text))

        async def events() -> AsyncIterator[str]:
            await asyncio.sleep(first)
            yield llm.chunk({"role": "assistant", "content": ""})
            for n, piece in enumerate(llm.pieces(text)):
                if n:
                    await asyncio.sleep(between)
                yield llm.chunk({"content": piece})
            yield llm.chunk({}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def embedder_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/embed")
    async def embed(body: dict[str, Any] = Body(...)) -> dict[str, list[list[float]]]:
        return {"embeddings": [scoring.vector(str(text)) for text in body.get("texts") or []]}

    return app


def reranker_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v2/rerank")
    async def rerank_v2(body: dict[str, Any] = Body(...)) -> dict[str, list[float]]:
        queries = [str(q) for q in body.get("queries") or []]
        documents = [str(d) for d in body.get("documents") or []]
        if not queries:
            raise HTTPException(status_code=422, detail="queries: at least one query is needed")
        return {"scores": scoring.scores(queries, documents)}

    return app


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rag_testing.serve", description="Start one fake of the harness")
    builders = {"llm": llm_app, "embedder": embedder_app, "reranker": reranker_app}
    ap.add_argument("name", choices=list(builders))
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args(argv)
    app = builders[args.name]()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
