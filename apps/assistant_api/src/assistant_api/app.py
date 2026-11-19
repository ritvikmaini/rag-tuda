"""The assistant API: create_app() builds the FastAPI object; the lifespan builds the runtime (settings, Langfuse, the turn
store's pool, the models, the prompt service) and loads the index and the graph.
A failed load or Langfuse login leaves the app degraded: /readyz says so, and a background task and the request path heal it."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from rag_core.adapters.http import close_http_client
from rag_core.adapters.langfuse import init_langfuse
from rag_core.adapters.pg_index import read_pool
from rag_core.domain.resilience import dispose_engines

from assistant_api.api import chat, ops, profile
from assistant_api.api.deps import limiter, set_chat_rate
from assistant_api.api.middleware import security_headers_and_cors
from assistant_api.chat.turns_store import TurnStore, store_conninfo
from assistant_api.runtime import build_runtime, heal, reload, stop_retiring
from assistant_api.settings import AssistantSettings, load_settings


class Page(StaticFiles):
    """The page's files: GET and HEAD only, any other method on a path no route serves is a 404 like everywhere else. A checkout's
    dotfiles, its tests and node_modules are not part of the page: they answer 404 too."""

    NOT_SERVED = frozenset({"tests", "node_modules"})

    async def get_response(self, path: str, scope):
        parts = path.split("/")
        if (
            scope["method"] not in ("GET", "HEAD")
            or parts[0] in self.NOT_SERVED
            or any(p.startswith(".") and p != "." for p in parts)
        ):
            raise HTTPException(status_code=404)
        return await super().get_response(path, scope)


def page_dir() -> Path | None:
    """The folder of the page: RAG_PAGE_DIR (the image sets it), else web/page of the checkout this file sits in; None when it
    does not exist (the backend then serves the API only)."""
    found = Path(os.environ.get("RAG_PAGE_DIR") or Path(__file__).resolve().parents[4] / "web" / "page")
    return found if found.is_dir() else None


def create_app(settings: AssistantSettings | None = None) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime = None
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # pyright: ignore[reportArgumentType]
    app.middleware("http")(security_headers_and_cors)
    app.include_router(ops.router)
    app.include_router(chat.router)
    app.include_router(profile.router)
    if (page := page_dir()) is not None:  # last: the routes above win, the page answers every other GET
        app.mount("/", Page(directory=page, html=True), name="page")
    return app


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = app.state.settings or load_settings()
    set_chat_rate(settings.overlay.server.rate_limit)
    langfuse, authenticated = await asyncio.to_thread(init_langfuse, settings.overlay.tracing.url)
    db = settings.overlay.database
    # the turn store's pool: rag_app on pgvector like the index reads, at most 4 connections; each transaction sets
    # its own statement_timeout
    async with read_pool(store_conninfo(db), max_size=4, name="turns") as pool:
        rt = build_runtime(settings, langfuse, authenticated)
        rt.turns = TurnStore(pool)
        app.state.runtime = rt
        try:
            await reload(rt)
        except Exception as e:
            logging.warning(
                f"Could not load retrievers on startup (no ingested data yet?): {e}. "
                "Backend will start in degraded mode and retry the load every minute."
            )
            rt.chain = None
            rt.index = None
        healing = asyncio.create_task(heal(rt))
        yield
        healing.cancel()  # before the dispose: no reload opens a pool after it, a task error never fails the shutdown
        await asyncio.gather(healing, return_exceptions=True)
        logging.info("Shutting down: cleaning up resources...")
        try:
            closed = await dispose_engines(*(rt.index.engines if rt.index is not None else ()), *stop_retiring(rt))
            logging.info(f"Closed {closed} database engine(s)")
            if langfuse:
                langfuse.flush()
                logging.info("Flushed Langfuse traces")
            await close_http_client()
            logging.info("Graceful shutdown complete")
        except Exception as e:
            logging.error(f"Error during shutdown cleanup: {e}")
