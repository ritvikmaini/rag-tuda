"""GET /api/chat: streams the answer; the pipeline runs inside the request guard (deadline and idle timeout)."""

import time
import uuid
from functools import partial
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from rag_core.domain.access import ANONYMOUS
from rag_core.domain.filters import Chips, check_chips
from rag_core.domain.history import valid_turn_id
from rag_core.domain.language import LANGUAGES

from assistant_api.api.deps import chat_rate, limiter, verify_token
from assistant_api.chat.guard import guard
from assistant_api.chat.service import pipeline
from assistant_api.chat.turns_store import TurnRecord, end_turn
from assistant_api.graph import tracing
from assistant_api.logref import set_ref
from assistant_api.sse import frames

router = APIRouter()


@router.get("/api/chat")
@limiter.limit(chat_rate)  # per client address
async def main(
    request: Request,
    question: str,
    conversation_id: str,
    turn_id: Optional[str] = None,
    replace_from: Optional[str] = None,
    kept: Optional[str] = None,
    formats: Optional[str] = None,
    variant: Optional[str] = None,
    variant_of: Optional[str] = None,
    variant_select: Optional[str] = None,
    degree: Optional[str] = None,
    programme: Optional[str] = None,
    po: Optional[int] = None,
    lang: Optional[str] = None,
    token: None = Depends(verify_token),
) -> StreamingResponse:
    rt = request.app.state.runtime
    lang = lang if lang in LANGUAGES else None  # the page's language: the texts that fire before the question is read
    chips = Chips(degree or None, programme or None, po)
    if chips != Chips():  # an empty selector is "not chosen"; a value the roles do not know is refused, never ignored
        roles = rt.settings.roles.model_dump() if rt.settings.roles else {}
        if problem := check_chips(chips, roles):
            raise HTTPException(status_code=422, detail=f"unknown filter: {problem}")
    timeouts = rt.settings.card.timeouts
    principal = ANONYMOUS  # the route is where a deployment names its principal
    # stream format 1: meta (the first frame) names the turn and the trace; a variant or a version switch belongs to the turn
    # it changes (variant_of), a question to its own id (the page's, else a new one)
    switch_or_variant = variant is not None or variant_select is not None
    turn = (valid_turn_id(variant_of) if switch_or_variant else valid_turn_id(turn_id)) or str(uuid.uuid4())
    set_ref(turn)  # every log line of this request, in its tasks and worker threads, carries ref=<first 8 characters>
    too_long = len(question) > rt.settings.profile.guard.max_question_chars
    trace = None if (variant_select is not None or too_long) else tracing.trace_id(rt.handler, turn, variant)
    # the turn's row: every question request; a variant or a switch changes the versions of an existing turn
    record = (
        None
        if switch_or_variant
        else TurnRecord(turn_id=turn, conversation_id=conversation_id, question=question, principal=principal.name)
    )
    on_end = partial(
        end_turn, rt.turns, record
    )  # the [Turn] line for every request; the store's finish where both exist
    # every request ends within the deadline with an answer or a message in the page's language
    deadline_at = time.monotonic() + timeouts.deadline_s
    return StreamingResponse(
        guard(
            lambda: pipeline(
                rt,
                question=question,
                user=conversation_id,
                turn_id=turn_id,
                replace_from=replace_from,
                kept=kept,
                formats=formats,
                variant=variant,
                variant_of=variant_of,
                variant_select=variant_select,
                turn=turn,
                trace_id=trace,
                principal=principal,
                record=record,
                deadline_at=deadline_at,
                chips=chips,
                lang=lang,
            ),
            deadline_s=timeouts.deadline_s,
            idle_s=timeouts.idle_s,
            meta=frames.meta(turn, trace),
            messages=rt.messages_for(lang),
            on_end=on_end,
            run_info=(lambda: record.run) if record is not None else None,
            scope_fallback=(lambda: getattr(record.result, "scope_fallback", False) is True)
            if record is not None
            else None,
        ),
        media_type="text/event-stream",
    )
