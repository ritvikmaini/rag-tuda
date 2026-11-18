"""The chat service: `pipeline()` runs one question through the graph over the runtime and yields the stream's frames
(`assistant_api.sse.frames`). It limits concurrency, handles the variants of the last answer and the edit/retry cut, and
names its outcome for the guard's `done` frame with `Declared` (too long, busy, refused, a failed variant). The conversation
lives in the turn store (`Runtime.turns`): a question's row is stored (and the edit cut made) before the early exits, its
history is read once the slot is held, a variant runs from the stored context and keeps its version only with an answer, and
the guard's end writes the row's outcome from the `TurnRecord` the route made."""

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Optional

from rag_core.domain.access import ANONYMOUS, Principal
from rag_core.domain.answer_shapes import FORMATS_VERSION
from rag_core.domain.history import kept_ids, valid_turn_id

from assistant_api.chat.guard import PROGRESS
from assistant_api.chat.turns_store import TurnRecord
from assistant_api.graph.history import documents, history_messages, question_message, variant_inputs
from assistant_api.graph.store import VARIANT_CODES
from assistant_api.graph.streaming import StreamResult, run_stream
from assistant_api.graph.tracing import run_config
from assistant_api.runtime import Runtime, try_reload
from assistant_api.sse import frames as fr
from assistant_api.sse.events import shape_frame


async def pipeline(
    rt: Runtime,
    *,
    question: str,
    user: str,
    turn_id: Optional[str] = None,
    replace_from: Optional[str] = None,
    kept: Optional[str] = None,
    formats: Optional[str] = None,
    variant: Optional[str] = None,
    variant_of: Optional[str] = None,
    variant_select: Optional[str] = None,
    turn: Optional[str] = None,
    trace_id: Optional[str] = None,
    principal: Principal = ANONYMOUS,
    record: TurnRecord | None = None,
) -> AsyncIterator[Any]:
    """
    RAG pipeline with concurrency control. `turn` is the turn id the route sent in `meta` (the stored question takes it) and
    `trace_id` the Langfuse trace id it sent (None: untraced). `record` is the question's turn (None for a variant or a switch):
    the store gets its row before the early exits and the run fills it, the guard's end writes it.

    Uses a semaphore to limit concurrent executions and prevent resource exhaustion
    under high load. Excess requests will wait for a slot to become available.
    """
    # Check if data has been ingested yet; a worker whose start-up load failed tries to
    # heal itself first
    if rt.chain is None and not await try_reload(rt):
        yield fr.Error(
            content="No index has been built yet. Build it first with `indexer build`.",
            code="not_ready",
        )
        return

    # Response formats v1: a version switch needs no model and no slot; a variant is checked before it waits for the slot,
    # so a refused one changes nothing.
    formats_on = formats == FORMATS_VERSION
    clean_variant_of = valid_turn_id(variant_of)
    if variant_select is not None:
        selected = formats_on and await rt.turns.select(user, clean_variant_of, variant_select)
        logging.info(f"[Variant] select name={variant_select!r} ok={selected} (user: {user})")
        yield fr.variant(variant_select, selected, "formats_off" if not formats_on else "no_version")
        if not selected:
            yield fr.Declared("refused")
        return
    if variant is not None:
        problem = "formats off"
        if formats_on:
            problem = await rt.turns.variant_problem(user, variant, clean_variant_of)
        if problem:
            logging.info(f"[Variant] refused name={variant!r} reason={problem} (user: {user})")
            yield fr.Chunk(content=rt.settings.messages.variant.refused)
            yield fr.variant(variant, False, VARIANT_CODES[problem])
            yield fr.Declared("refused")
            return

    # Edit or retry: store the turn and cut the conversation at the edited question first, so a request
    # that ends early below (too long, busy) still leaves the stored conversation as the page shows it.
    clean_turn_id, clean_replace_from = valid_turn_id(turn_id), valid_turn_id(replace_from)
    clean_kept = kept_ids(kept)  # the page's last kept questions: place an edit of a never-stored question
    if turn_id and not clean_turn_id:
        logging.warning(f"[History] ignoring an invalid turn_id (user: {user})")
    if replace_from and not clean_replace_from:
        logging.warning(f"[History] ignoring an invalid replace_from (user: {user})")
    if kept and clean_kept is None:
        logging.warning(f"[History] ignoring an invalid kept list (user: {user})")
    if record is not None:
        await rt.turns.begin(record, clean_replace_from, clean_kept)
        # an early exit below (too long, busy) names its configuration too; the finally block records the full run
        record.run = {**rt.run_info(), "prompts": {}}
        yield PROGRESS  # the store write counts as activity: the slot wait keeps its whole share of the silence limit

    # Over-long messages would overflow the rewriter call; answer them directly.
    max_chars = rt.settings.profile.guard.max_question_chars
    if len(question) > max_chars:
        limit_text = f"{max_chars:,}".replace(",", ".")
        yield fr.Chunk(content=rt.settings.messages.question.too_long.format(limit=limit_text))
        yield fr.Declared("too_long")
        return

    # Try to acquire semaphore with timeout to prevent indefinite blocking
    try:
        await asyncio.wait_for(
            rt.slot.acquire(),
            timeout=rt.settings.overlay.compute.slot_wait_s,  # then the busy message
        )
    except asyncio.TimeoutError:
        logging.warning(f"RAG pipeline timeout waiting for slot (user: {user})")
        yield fr.Error(content="Service is currently busy. Please try again shortly.", code="busy")
        yield fr.Declared("busy")
        return

    result = StreamResult()
    if record is not None:
        record.result = result
    try:
        yield PROGRESS  # the slot wait is over: the guard's silence limit starts again
        thread_id = user
        if variant is None:
            # With the slot held the store cuts again (a request of this conversation may have stored a turn the page
            # removed while this one waited), then reads the completed turns before this one.
            past = await rt.turns.history(record, clean_replace_from, clean_kept) if record is not None else []
            inputs: dict[str, Any] = {
                "messages": history_messages(past)
                + [question_message(question, turn or clean_turn_id, datetime.now(timezone.utc))],
                "variant": None,
                "formats": FORMATS_VERSION if formats_on else "",
            }
        else:
            # the stored context of the turn: the run changes nothing until its version is kept below
            taken = await rt.turns.variant_input(user, variant, clean_variant_of)
            if isinstance(taken, str):  # the reason it was refused
                yield fr.Chunk(content=rt.settings.messages.variant.refused)
                yield fr.variant(variant, False, VARIANT_CODES[taken])
                yield fr.Declared("refused")
                return
            result.retrieved_docs = documents(taken.candidates)
            result.requested_shape = taken.shape
            inputs = variant_inputs(taken, variant, FORMATS_VERSION)
            yield shape_frame(result.requested_shape)

        chain = rt.chain

        config = run_config(rt.handler if trace_id else None, thread_id, rt.collection, trace_id=trace_id)
        # configurable is not state: the Principal object reaches the nodes, never the turn store (pass the object,
        # never its name as a str)
        config["configurable"]["principal"] = principal

        try:
            async for frame in run_stream(
                chain,
                inputs,
                config,
                formats_on=formats_on,
                variant=variant,
                sources_mode=rt.settings.profile.formats.sources_event,
                none_text=rt.settings.messages.docs.none,
                result=result,
                user=user,
                strip_cjk=rt.settings.card.quirks.strip_cjk,
            ):
                yield frame
        except asyncio.CancelledError:
            # Client disconnected - this is normal, don't crash
            logging.info(f"Request cancelled by client (user: {user})")
        except Exception as e:
            logging.error(
                f"Error in RAG pipeline (user: {user}): {type(e).__name__}",
                exc_info=True,
            )
            if record is not None:
                record.error = "pipeline_error:" + type(e).__name__
            yield fr.Error(content="An error occurred processing your request.", code="pipeline_error")
        if variant is not None:
            # a variant without an answer (a stop, a failed run) keeps nothing: the turn stays as it was
            ok = result.answer is not None and await rt.turns.keep_version(
                user, str(clean_variant_of), variant, result.answer, result.requested_shape
            )
            logging.info(f"[Variant] name={variant} ok={ok} (user: {user})")
            yield fr.variant(variant, ok, "failed")
            if not ok:
                yield fr.Declared("failed")
    finally:
        try:
            if record is not None:
                record.run = rt.run_info()  # the configuration this turn ran on: index generation, prompt versions
        finally:
            rt.slot.release()
        # No per-request langfuse.flush(): it blocked the event loop; the tracing handler sends in the background
