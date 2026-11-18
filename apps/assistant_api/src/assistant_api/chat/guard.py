"""
Request guard: runs the answer pipeline as its own task and streams its SSE frames, so that every request ends in bounded
time with an answer or a German message. The page ignores `error` events, so every failure is also sent as a
`chunk` the page shows.

- total deadline and maximum silence between two frames; on expiry the pipeline
  task is cancelled (which closes the model stream and frees the concurrency slot)
- an `error` frame from the pipeline is followed by the German message, chosen by its code
- an exception in the pipeline, or an answer without any text, ends with the error message
- a `variant` frame (a version switch or the end of a variant) counts as an answer
- a client that disconnects cancels the pipeline
- stream format 1: `meta` first, `done` last except after a client stop, `code` on the guard's own errors, the German text
  chosen by code
- `on_end` receives the turn's outcome once, `stopped` after a client stop, and the first error the guard sent (`timeout`,
  `empty_answer`, `exception:<Class>`, a pipeline `Error` frame's code; None for completed and stopped), which becomes
  chat.turns.error and the `[Turn]` line's code
- `done` names the turn's index generation and prompt versions, read from `run_info` (the turn record's run, which the
  service writes before the pipeline ends)
- the guard's own log lines carry the turn's `ref=` through the logging filter (assistant_api.logref)

Pure asyncio.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from assistant_api.sse import frames as fr

logger = logging.getLogger(__name__)

_DONE = object()
# The pipeline yields PROGRESS while it works without sending anything (slot acquired,
# analysis, retrieval, reranking, a model chunk without text such as a reasoning model's
# thinking): it counts as activity for the silence limit and is never forwarded to the page.
PROGRESS = ": progress"
CANCEL_WAIT_S = 5.0  # how long the guard waits for the cancelled pipeline to clean up
GERMAN = {
    "busy": "guard.busy",
    "not_ready": "guard.not_ready",
    "timeout": "guard.timeout",
}  # every other code: guard.error
# turn ends that outlive a cancelled request (a client stop): referenced until done
_ENDING: set[asyncio.Task[None]] = set()


async def _record_end(
    task: asyncio.Task[Any], on_end: Callable[[str, str | None], Awaitable[None]], outcome: str, error: str | None
) -> None:
    """The turn's end once the pipeline has stopped (it may still be filling the record or holding the slot)."""
    if not task.done():
        await asyncio.wait({task}, timeout=CANCEL_WAIT_S)
    try:
        await on_end(outcome, error)
    except Exception as exc:
        logger.error("[Turns] the turn's end was not recorded: %s: %s", type(exc).__name__, exc)


async def guard(
    producer_factory: Callable[[], AsyncIterator[Any]],
    deadline_s: float,
    idle_s: float,
    meta: fr.Meta,
    messages: Mapping[str, str],
    on_end: Callable[[str, str | None], Awaitable[None]] | None = None,
    run_info: Callable[[], Mapping[str, Any]] | None = None,
    scope_fallback: Callable[[], bool] | None = None,
) -> AsyncGenerator[str, None]:
    """Yield the encoded frames of producer_factory() within deadline_s in total and idle_s between two items. on_end(outcome, error)
    runs once when the request is over: the outcome of the `done` frame, `stopped` when the client left before it; error is the
    first error the guard sent (`<code>`, `exception:<Class>`), None for completed and stopped. run_info() (the turn record's run)
    is read when `done` is built: its index generation and prompt versions go into the frame. scope_fallback() tells whether the
    turn was answered by the out-of-scope fallback: the frame says so only then, every other `done` keeps its bytes."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def done(outcome: fr.Outcome) -> str:
        run = run_info() if run_info is not None else {}
        fallback = True if scope_fallback is not None and scope_fallback() else None
        return fr.encode(fr.done(outcome, run.get("index_generation"), run.get("prompts") or None, fallback))

    def german(code: str, after_text: bool) -> fr.Chunk:
        return fr.Chunk(content=("\n\n" if after_text else "") + messages[GERMAN.get(code, "guard.error")])

    async def produce() -> None:
        try:
            async for item in producer_factory():
                queue.put_nowait(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[RequestGuard] pipeline failed: %s", type(exc).__name__, exc_info=True)
            queue.put_nowait(exc)
        finally:
            queue.put_nowait(_DONE)

    task = asyncio.create_task(produce())
    started = loop.time()
    text_sent, failed, declared = False, False, None
    final: str | None = None
    error: str | None = None
    try:
        yield fr.encode(meta)
        while True:
            left = deadline_s - (loop.time() - started)
            try:
                if left <= 0:
                    raise asyncio.TimeoutError
                item = await asyncio.wait_for(queue.get(), timeout=min(left, idle_s))
            except asyncio.TimeoutError:
                logger.warning(
                    "[RequestGuard] stopped: no answer within %.0f s in total or %.0f s of silence", deadline_s, idle_s
                )
                error = error or "timeout"
                yield fr.encode(fr.Error(content="timeout", code="timeout"))
                yield fr.encode(german("timeout", text_sent))
                final = "failed"
                yield done(final)
                return
            if item is _DONE:
                if not text_sent:
                    logger.warning("[RequestGuard] the pipeline ended without any answer text")
                    error = error or "empty_answer"
                    yield fr.encode(fr.Error(content="empty answer", code="empty_answer"))
                    yield fr.encode(german("empty_answer", False))
                    final = "failed"
                    yield done(final)
                    return
                final = declared or ("failed" if failed else "completed")
                yield done(final)
                return
            if isinstance(item, BaseException):
                error = error or "exception:" + type(item).__name__
                yield fr.encode(fr.Error(content=type(item).__name__, code="exception"))
                yield fr.encode(german("exception", text_sent))
                final = "failed"
                yield done(final)
                return
            if item == PROGRESS:
                continue
            if isinstance(item, fr.Declared):
                declared = item.outcome
                continue
            yield fr.encode(item)
            if isinstance(item, fr.Chunk) and item.content:
                text_sent = True
            elif isinstance(item, fr.Variant):
                text_sent = True
            elif isinstance(item, fr.Error):
                error = error or item.code
                yield fr.encode(german(item.code, text_sent))
                text_sent, failed = True, True
    finally:
        if not task.done():
            task.cancel()
        if on_end is not None:
            # the turn's end, "stopped" when the client left before done. Scheduled before the first await of this
            # finally: a client stop's cancellation may hit every await here, the shielded task finishes anyway.
            ending = asyncio.ensure_future(_record_end(task, on_end, final or "stopped", error))
            _ENDING.add(ending)
            ending.add_done_callback(_ENDING.discard)
            await asyncio.shield(ending)
        elif not task.done():
            await asyncio.wait({task}, timeout=CANCEL_WAIT_S)
