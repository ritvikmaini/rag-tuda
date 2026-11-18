"""The turn store: the conversations' turns, their answer versions and their context in the app's own tables (schema chat).
The history, the edit and retry cut, the answer versions and a variant's sources are rows. Every write, and every read a
decision rests on, runs in one transaction that sets statement_timeout and takes pg_advisory_xact_lock(1,
hashtext(conversation_id)) first: the lock is transaction-scoped, so it needs no session. A
failure never costs the answer: the question runs without history, a variant is refused, an end is logged.

The guard's on_end is end_turn, which closes every turn with one INFO line
`[Turn] outcome=<o> code=<c|-> intent=<i|-> ttft_ms=<n|-> total_ms=<n|->` (its lines carry the turn's `ref=`, assistant_api.logref;
` fallback=yes` follows total_ms only on a turn the out-of-scope fallback answered)
and stores the error as `<code>` or `<code>:<ExceptionClass>` in chat.turns.error, never a message."""

import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from psycopg import AsyncConnection
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from rag_core.adapters.db import credentials
from rag_core.domain.answer_shapes import SHAPE_VARIANTS, shape_of, turn_variant_problem
from rag_core.domain.history import LastTurn, PastTurn, VariantInput, cut_turns
from rag_core.domain.retrieval import section_identity
from rag_core.domain.source_meta import select_sources_for_display
from rag_core.settings.sections import DatabaseSection

from assistant_api.sse.frames import finite

logger = logging.getLogger(__name__)

LIVE = (
    "SELECT turn_id::text FROM chat.turns WHERE conversation_id = %s AND superseded_by IS NULL "
    "ORDER BY created_at, turn_id"
)
INSERT = (
    "INSERT INTO chat.turns (turn_id, conversation_id, principal, question, outcome, created_at) "
    "VALUES (%s, %s, %s, %s, 'failed', clock_timestamp()) ON CONFLICT (turn_id) DO NOTHING RETURNING turn_id"
)
SUPERSEDE = "UPDATE chat.turns SET superseded_by = %s WHERE turn_id = ANY(%s::uuid[]) AND turn_id <> %s"
HISTORY = """SELECT t.turn_id::text, t.question, coalesce(v.text, t.answer, ''), t.created_at
FROM chat.turns t LEFT JOIN chat.turn_versions v ON v.turn_id = t.turn_id AND v.active
WHERE t.conversation_id = %s AND t.superseded_by IS NULL AND t.outcome = 'completed' AND t.turn_id <> %s
  AND t.created_at < coalesce((SELECT s.created_at FROM chat.turns s WHERE s.turn_id = %s), clock_timestamp())
ORDER BY t.created_at, t.turn_id"""
LAST = """SELECT t.turn_id::text, t.outcome = 'completed' AND btrim(coalesce(t.answer, '')) <> '',
       EXISTS (SELECT 1 FROM chat.turn_context c WHERE c.turn_id = t.turn_id)
FROM chat.turns t
WHERE t.conversation_id = %s AND t.superseded_by IS NULL AND t.outcome NOT IN ('too_long', 'busy')
ORDER BY t.created_at DESC, t.turn_id DESC LIMIT 1"""
TARGET = """SELECT t.question, t.created_at, t.answer_shape, c.sections, o.shape
FROM chat.turns t JOIN chat.turn_context c ON c.turn_id = t.turn_id
LEFT JOIN chat.turn_versions o ON o.turn_id = t.turn_id AND o.name = 'original'
WHERE t.turn_id = %s"""
KEEP_ORIGINAL = (
    "INSERT INTO chat.turn_versions (turn_id, name, text, shape, active) "
    "SELECT turn_id, 'original', answer, answer_shape, false FROM chat.turns WHERE turn_id = %s "
    "ON CONFLICT (turn_id, name) DO NOTHING"
)
DEACTIVATE = "UPDATE chat.turn_versions SET active = false WHERE turn_id = %s"
KEEP = (
    "INSERT INTO chat.turn_versions (turn_id, name, text, shape, active) VALUES (%s, %s, %s, %s, true) "
    "ON CONFLICT (turn_id, name) DO UPDATE SET text = excluded.text, shape = excluded.shape, active = true, "
    "created_at = now()"
)
SELECT_VERSION = (
    "UPDATE chat.turn_versions SET active = (name = %s) WHERE turn_id = %s "
    "AND EXISTS (SELECT 1 FROM chat.turn_versions WHERE turn_id = %s AND name = %s)"
)
FINISH = (
    "UPDATE chat.turns SET outcome = %s, standalone_question = %s, intent = %s, answer = %s, answer_shape = %s, "
    "retrieved_section_ids = %s, cited_section_ids = %s, prompts = %s, collection = %s, index_generation = %s, "
    "config_hash = %s, stage_ms = %s, usage = %s, error = %s WHERE turn_id = %s"
)
CONTEXT = "INSERT INTO chat.turn_context (turn_id, sections) VALUES (%s, %s) ON CONFLICT (turn_id) DO NOTHING"
GENERATION = "SELECT generation FROM index_generations WHERE collection = %s AND status = 'live'"
AGES = """SELECT count(*) FILTER (WHERE created_at >= now() - interval '1 day'),
       count(*) FILTER (WHERE created_at < now() - interval '1 day' AND created_at >= now() - interval '7 days'),
       count(*) FILTER (WHERE created_at < now() - interval '7 days' AND created_at >= now() - interval '30 days'),
       count(*) FILTER (WHERE created_at < now() - interval '30 days' AND created_at >= now() - interval '365 days'),
       count(*) FILTER (WHERE created_at < now() - interval '365 days')
FROM chat.turns"""
AGE_BUCKETS = ("<1d", "1-7d", "7-30d", "30-365d", ">365d")


@dataclass
class TurnRecord:
    """One question request's turn: the route makes it, the service fills result and run, the guard's end writes it.
    Variants and version switches have none: they change turn_versions of an existing turn."""

    turn_id: str
    conversation_id: str
    question: str
    principal: str = "anonymous"
    started: float = field(default_factory=time.monotonic)
    stored: bool = False
    result: Any = None  # assistant_api.graph.streaming.StreamResult
    run: dict[str, Any] = field(default_factory=dict)  # collection, index_generation, config_hash, prompts
    error: str | None = None  # "<code>" or "<code>:<ExceptionClass>", never a message


def stage_ms(t: TurnRecord) -> dict[str, int | None]:
    """Time to the first chunk and in total, in milliseconds (the first chunk is None while no text came)."""
    first = getattr(t.result, "first_chunk", None)
    return {
        "ttft": round((first - t.started) * 1000) if first else None,
        "total": round((time.monotonic() - t.started) * 1000),
    }


async def end_turn(store: "TurnStore | None", record: TurnRecord | None, outcome: str, error: str | None) -> None:
    """The guard's on_end: one [Turn] line for every turn (variants and switches have no record or store: dashes), then the
    store's finish. The error of the guard fills the record only where the pipeline did not set its own (with the class)."""
    if record is not None:
        record.error = record.error or error
    code = ((record.error if record is not None else error) or "-").split(":", 1)[0]
    ms = stage_ms(record) if record is not None else {}
    intent = (getattr(record.result, "analysis", None) or {}).get("intent") if record is not None else None
    fallback = " fallback=yes" if getattr(getattr(record, "result", None), "scope_fallback", False) is True else ""
    logger.info(
        "[Turn] outcome=%s code=%s intent=%s ttft_ms=%s total_ms=%s%s",
        outcome,
        code,
        intent or "-",
        _ms(ms.get("ttft")),
        _ms(ms.get("total")),
        fallback,
    )
    if store is not None and record is not None:
        await store.finish(record, outcome)


def _ms(value: int | None) -> str:
    return "-" if value is None else str(value)


def store_conninfo(db: DatabaseSection) -> str:
    """rag_app on pgvector (nothing prepared, the lock transaction-scoped)."""
    user, password, name = credentials(db)
    return make_conninfo(host=db.host, port=db.port, dbname=name, user=user, password=password, connect_timeout=5)


def turn_context(
    candidates: Sequence[Any], analysis: dict[str, Any], sources: Sequence[Any], scope_fallback: bool = False
) -> dict[str, Any]:
    """What a variant re-fits from: the candidates in their grouped order with their metadata and
    identity, what the query analysis found, and the ids actually sent. NaN and Infinity become null (jsonb refuses them). A turn
    answered by the out-of-scope fallback adds `"scope_fallback": true`, so its variants re-fit from the sources."""
    return finite(
        {
            "candidates": [
                {
                    "page_content": d.page_content,
                    "metadata": dict(d.metadata or {}),
                    "section": section_identity(d.page_content, d.metadata or {}),
                }
                for d in candidates
            ],
            "standalone_question": analysis.get("standalone_question") or "",
            "intent": analysis.get("intent") or "",
            "language": analysis.get("language") or "",
            "sent": [s.get("section_id") for s in sources if isinstance(s, dict)],
            **({"scope_fallback": True} if scope_fallback else {}),
        }
    )


def _past(rows: Sequence[Any]) -> list[PastTurn]:
    return [PastTurn(turn_id=r[0], question=r[1], answer=r[2], asked_at=r[3]) for r in rows]


class TurnStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    @asynccontextmanager
    async def _locked(self, conversation_id: str) -> AsyncIterator[AsyncConnection[Any]]:
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute("SET LOCAL statement_timeout = '5s'")
            await conn.execute("SELECT pg_advisory_xact_lock(1, hashtext(%s))", (conversation_id,))
            yield conn

    async def begin(self, t: TurnRecord, replace_from: str | None, kept: list[str] | None) -> None:
        """Before the question's early exits (too long, busy): the edit or retry cut, then the turn's row (outcome failed until
        its end is written, so a request that dies is failed)."""
        try:
            async with self._locked(t.conversation_id) as conn:
                ids = [r[0] for r in await (await conn.execute(LIVE, (t.conversation_id,))).fetchall()]
                cut = cut_turns(ids, replace_from, kept)
                row = await (
                    await conn.execute(INSERT, (t.turn_id, t.conversation_id, t.principal, t.question))
                ).fetchone()
                if row is None:
                    logger.warning(
                        "[Turns] turn %s is already stored; this request is not recorded (user: %s)",
                        t.turn_id,
                        t.conversation_id,
                    )
                    return
                t.stored = True
                if cut:
                    await conn.execute(SUPERSEDE, (t.turn_id, cut, t.turn_id))
            if cut:
                logger.info("[History] replaced from %s: %d turns superseded", replace_from, len(cut))
            elif replace_from and replace_from not in ids and kept is None:
                logger.warning(
                    "[History] replace_from %s is not in the conversation (%d stored turns); nothing removed",
                    replace_from,
                    len(ids),
                )
        except Exception as exc:
            t.stored = False
            logger.warning(
                "[Turns] the turn could not be stored, so it is not recorded and the edit cut is not made: %s: %s",
                type(exc).__name__,
                exc,
            )

    async def history(self, t: TurnRecord, replace_from: str | None, kept: list[str] | None) -> list[PastTurn]:
        """Once the slot is held: the cut again (a request of the same conversation may have stored a turn the page removed
        while this one waited), then the completed, not superseded turns before this one."""
        try:
            async with self._locked(t.conversation_id) as conn:
                if replace_from and t.stored:
                    ids = [
                        r[0]
                        for r in await (await conn.execute(LIVE, (t.conversation_id,))).fetchall()
                        if r[0] != t.turn_id
                    ]
                    cut = cut_turns(ids, replace_from, kept)
                    if cut:
                        await conn.execute(SUPERSEDE, (t.turn_id, cut, t.turn_id))
                        logger.warning(
                            "[History] replaced again from %s after the slot wait: %d turns superseded",
                            replace_from,
                            len(cut),
                        )
                rows = await (await conn.execute(HISTORY, (t.conversation_id, t.turn_id, t.turn_id))).fetchall()
            return _past(rows)
        except Exception as exc:
            logger.warning(
                "[Turns] the history could not be read, the question runs without it: %s: %s", type(exc).__name__, exc
            )
            return []

    async def finish(self, t: TurnRecord, outcome: str) -> None:
        """The turn's end (the guard's on_end): the outcome and what the run produced. A completed request without a model
        message (a fallback text) is failed: it never enters the history."""
        if not t.stored:
            return
        r = t.result
        answer = getattr(r, "answer", None)
        if outcome == "completed" and not answer:
            outcome = "failed"
        analysis = dict(getattr(r, "analysis", None) or {})
        candidates = list(getattr(r, "candidates", None) or [])
        sources = list(getattr(r, "sources", None) or [])
        usage = dict(getattr(r, "usage", None) or {})
        retrieved = [str((d.metadata or {}).get("section_id") or "") for d in candidates]
        cited = (
            [str(s.get("section_id") or "") for s in select_sources_for_display(sources, answer, "cited")]
            if answer
            else []
        )
        async with self._locked(t.conversation_id) as conn:
            await conn.execute(
                FINISH,
                (
                    outcome,
                    analysis.get("standalone_question"),
                    analysis.get("intent"),
                    answer,
                    getattr(r, "requested_shape", None) if r is not None else None,
                    retrieved,
                    cited,
                    Jsonb(t.run.get("prompts") or {}),
                    t.run.get("collection"),
                    t.run.get("index_generation"),
                    t.run.get("config_hash"),
                    Jsonb(stage_ms(t)),
                    Jsonb(usage) if usage else None,
                    t.error,
                    t.turn_id,
                ),
            )
            if outcome == "completed" and candidates:
                await conn.execute(
                    CONTEXT,
                    (
                        t.turn_id,
                        Jsonb(turn_context(candidates, analysis, sources, getattr(r, "scope_fallback", False) is True)),
                    ),
                )

    async def _last(self, conn: AsyncConnection[Any], conversation_id: str) -> LastTurn | None:
        row = await (await conn.execute(LAST, (conversation_id,))).fetchone()
        return LastTurn(turn_id=row[0], answered=bool(row[1]), has_sources=bool(row[2])) if row else None

    async def variant_problem(self, conversation_id: str, variant: Any, variant_of: str | None) -> str:
        """Before the slot: why the variant cannot be made ("" when it can; "state unavailable" when the store fails)."""
        try:
            async with self._locked(conversation_id) as conn:
                return turn_variant_problem(variant, variant_of, await self._last(conn, conversation_id))
        except Exception as exc:
            logger.warning("[Variant] state read failed: %s", exc)
            return "state unavailable"

    async def variant_input(self, conversation_id: str, variant: str, variant_of: str | None) -> VariantInput | str:
        """With the slot held: the turn checked again, then its question, the history before it, its stored candidates and the
        shape the variant is written in (a shape variant its own; every other the original's). Nothing is cut:
        the variant's run leaves the target out of its history by construction."""
        try:
            async with self._locked(conversation_id) as conn:
                problem = turn_variant_problem(variant, variant_of, await self._last(conn, conversation_id))
                if problem:
                    logger.info("[Variant] take-out refused: %s", problem)
                    return problem
                target = await (await conn.execute(TARGET, (variant_of,))).fetchone()
                if target is None:  # the checked turn has its context row under the lock; none means a broken store
                    return "state unavailable"
                question, asked_at, answer_shape, sections, original_shape = target
                rows = await (await conn.execute(HISTORY, (conversation_id, variant_of, variant_of))).fetchall()
            shape = variant if variant in SHAPE_VARIANTS else shape_of(original_shape or answer_shape)
            return VariantInput(
                turn_id=str(variant_of),
                question=question,
                asked_at=asked_at,
                history=tuple(_past(rows)),
                candidates=tuple(sections.get("candidates") or ()),
                standalone_question=sections.get("standalone_question") or "",
                intent=sections.get("intent") or "",
                language=sections.get("language") or "",
                shape=shape,
                scope_fallback=sections.get("scope_fallback") is True,
            )
        except Exception as exc:
            logger.warning("[Variant] take-out failed: %s", exc)
            return "state unavailable"

    async def keep_version(self, conversation_id: str, turn_id: str, name: str, text: str, shape: str) -> bool:
        """After a variant's run with an answer: the original kept the first time, the variant stored (or replaced) and active."""
        try:
            async with self._locked(conversation_id) as conn:
                await conn.execute(KEEP_ORIGINAL, (turn_id,))
                await conn.execute(DEACTIVATE, (turn_id,))
                await conn.execute(KEEP, (turn_id, name, text, shape))
            return True
        except Exception as exc:
            logger.error("[Variant] keeping %s failed: %s", name, exc)
            return False

    async def select(self, conversation_id: str, variant_of: str | None, name: str) -> bool:
        """A version switch, without the model: only for the conversation's last answered turn and a stored version."""
        try:
            async with self._locked(conversation_id) as conn:
                last = await self._last(conn, conversation_id)
                if not variant_of or last is None or not last.answered or last.turn_id != variant_of:
                    return False
                cur = await conn.execute(SELECT_VERSION, (name, variant_of, variant_of, name))
                return cur.rowcount > 0
        except Exception as exc:
            logger.warning("[Variant] selecting %s failed: %s", name, exc)
            return False

    async def live_generation(self, collection: str) -> int | None:
        try:
            async with self.pool.connection() as conn:
                row = await (await conn.execute(GENERATION, (collection,))).fetchone()
            return int(row[0]) if row else None
        except Exception as exc:
            logger.warning("[Turns] the live index generation could not be read: %s: %s", type(exc).__name__, exc)
            return None

    async def ages(self) -> dict[str, int]:
        async with self.pool.connection() as conn:
            row = await (await conn.execute(AGES)).fetchone()  # one aggregate row, always
        return dict(zip(AGE_BUCKETS, (int(n) for n in row or (0,) * 5), strict=True))
