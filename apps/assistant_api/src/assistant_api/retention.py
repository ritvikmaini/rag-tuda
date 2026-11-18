"""python -m assistant_api.retention [--apply]: the retention job. The periods are constants here and nowhere else.
Without --apply it only reports what would go. One run, in this order:
  1. conversations whose newest turn is older than CONVERSATION_DAYS: all their rows in chat.turn_context,
     chat.turn_versions and chat.turns (whole conversations, one transaction);
  2. the legacy LangGraph checkpoint threads whose newest checkpoint is older than CONVERSATION_DAYS, in
     checkpoint_writes, checkpoint_blobs and checkpoints (the tables stay);
  3. Langfuse traces with an observation older than TRACE_DAYS, listed through the observations API and deleted through the
     trace API of the client the backend uses (at most MAX_PAGES list pages per run);
  4. one summary line.
A step that fails or is not allowed prints one line and is skipped; the others still run. Exit 0 when every step ran, 1 when
one did not. It runs inside the assistant container with the assistant's own settings (rag_app on pgvector):
docker compose exec backend python -m assistant_api.retention --apply."""

import argparse
import asyncio
import sys
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from typing import Any

from rag_core.adapters.langfuse import init_langfuse
from rag_core.adapters.pg_index import read_pool

from assistant_api.chat.turns_store import AGE_BUCKETS, TurnStore, store_conninfo
from assistant_api.settings import load_settings

CONVERSATION_DAYS = 90
TRACE_DAYS = 30
BATCH = 100  # traces per delete request
OBSERVATIONS_PAGE = 1000  # observations per list page
MAX_PAGES = 200  # list pages per run
DB_TIMEOUT = "120s"
API_TIMEOUT = {"timeout_in_seconds": 60}

# the turns of the expired conversations, read once; the deletes run by these turn ids, so a turn committed to a resumed
# conversation after the read is kept (superseded_by points from older to newer turns, so an all-old set is FK-safe)
OLD_CONVERSATIONS = (
    "SELECT turn_id, conversation_id FROM chat.turns WHERE conversation_id IN ("
    "SELECT conversation_id FROM chat.turns GROUP BY conversation_id "
    "HAVING max(created_at) < now() - %s::int * interval '1 day')"
)
DELETE_CONVERSATIONS = (
    "DELETE FROM chat.turn_context WHERE turn_id = ANY(%s)",
    "DELETE FROM chat.turn_versions WHERE turn_id = ANY(%s)",
    "DELETE FROM chat.turns WHERE turn_id = ANY(%s)",
)
OLD_THREADS = (
    "SELECT thread_id FROM public.checkpoints GROUP BY thread_id "
    "HAVING max((checkpoint->>'ts')::timestamptz) < now() - %s::int * interval '1 day'"
)
DELETE_THREADS = (
    "DELETE FROM public.checkpoint_writes WHERE thread_id = ANY(%s)",
    "DELETE FROM public.checkpoint_blobs WHERE thread_id = ANY(%s)",
    "DELETE FROM public.checkpoints WHERE thread_id = ANY(%s)",
)


async def conversations(pool: Any, apply: bool) -> tuple[int, int]:
    """(conversations, turns) older than the period; deleted when apply."""
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SET LOCAL statement_timeout = '%s'" % DB_TIMEOUT)
        rows = await (await conn.execute(OLD_CONVERSATIONS, (CONVERSATION_DAYS,))).fetchall()
        turn_ids = [r[0] for r in rows]
        if apply and turn_ids:
            for statement in DELETE_CONVERSATIONS:
                await conn.execute(statement, (turn_ids,))
        return len({r[1] for r in rows}), len(turn_ids)


async def checkpoint_threads(pool: Any, apply: bool) -> int:
    """The legacy threads older than the period; deleted when apply. A missing table or privilege raises."""
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SET LOCAL statement_timeout = '%s'" % DB_TIMEOUT)
        ids = [r[0] for r in await (await conn.execute(OLD_THREADS, (CONVERSATION_DAYS,))).fetchall()]
        if apply and ids:
            for statement in DELETE_THREADS:
                await conn.execute(statement, (ids,))
        return len(ids)


def traces(client: Any, apply: bool, now: datetime | None = None) -> int:
    """The Langfuse traces that have an observation older than TRACE_DAYS (the cutoff is on the observation's start time);
    deleted in batches when apply. With events_only storage the server rejects the trace list, so the old observations are
    listed (cursor paging, ids only) and their distinct trace ids collected. All ids are listed before the first delete, so the
    paging does not shift under the deletions (Langfuse deletes in the background). At most MAX_PAGES pages per run: a larger
    backlog prints one line and continues the next day."""
    cutoff = (now or datetime.now(UTC)) - timedelta(days=TRACE_DAYS)
    ids: dict[str, None] = {}
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        got = client.api.observations.get_many(
            to_start_time=cutoff, fields="core", limit=OBSERVATIONS_PAGE, cursor=cursor, request_options=API_TIMEOUT
        )
        ids.update(dict.fromkeys(o.trace_id for o in got.data if o.trace_id))
        cursor = got.meta.cursor
        if not cursor:
            break
    else:
        print("[Retention] traces: page cap of %d reached; the rest follows in the next run" % MAX_PAGES, flush=True)
    found = list(ids)
    if apply:
        for start in range(0, len(found), BATCH):
            client.api.trace.delete_multiple(trace_ids=found[start : start + BATCH], request_options=API_TIMEOUT)
    return len(found)


def _reason(exc: Exception) -> str:
    return "%s: %s" % (type(exc).__name__, (str(exc).strip().splitlines() or [""])[0])


def _trace_client(url: str) -> Any:
    client, authenticated = init_langfuse(url, max_retries=2)
    if client is None or not authenticated:
        raise RuntimeError("Langfuse is not available")
    return client


def report(found: dict[str, Any], kept: dict[str, int] | str, apply: bool) -> str:
    """The summary line. found: conversations (n, turns), checkpoints n, traces n; a step that did not run is a string."""
    conv, threads, trace = found["conversations"], found["checkpoints"], found["traces"]
    sign = "-" if apply else ""
    parts = [
        "conversations skipped"
        if isinstance(conv, str)
        else "conversations %s%d (turns %s%d)" % (sign, conv[0], sign, conv[1]),
        "checkpoint threads skipped" if isinstance(threads, str) else "checkpoint threads %s%d" % (sign, threads),
        "traces skipped" if isinstance(trace, str) else "traces %s%d" % (sign, trace),
    ]
    ages = kept if isinstance(kept, str) else ", ".join("%s %d" % (b, kept.get(b, 0)) for b in AGE_BUCKETS)
    if apply:
        return "[Retention] %s; kept: %s" % (", ".join(parts), ages)
    return "[Retention] report only, would go: %s; turns by age: %s" % (", ".join(parts), ages)


async def run(settings: Any, apply: bool) -> tuple[str, bool]:
    """(the summary line, whether every step ran). A skipped step prints its own line first."""
    found: dict[str, Any] = {}
    kept: dict[str, int] | str = "unknown"
    # max_size 2: read_pool opens min_size=2, and psycopg_pool refuses max_size < min_size
    async with read_pool(store_conninfo(settings.overlay.database), max_size=2, name="retention") as pool:

        async def step(name: str, work: Awaitable[Any]) -> None:
            try:
                found[name] = await work
            except Exception as exc:
                found[name] = "skipped"
                print("[Retention] %s skipped: %s" % (name, _reason(exc)), flush=True)

        async def trace_step() -> int:
            client = await asyncio.to_thread(_trace_client, settings.overlay.tracing.url)
            try:
                return await asyncio.to_thread(traces, client, apply)
            finally:
                await asyncio.to_thread(client.shutdown)

        await step("conversations", conversations(pool, apply))
        await step("checkpoints", checkpoint_threads(pool, apply))
        await step("traces", trace_step())
        try:
            kept = await TurnStore(pool).ages()
        except Exception as exc:
            print("[Retention] ages skipped: %s" % _reason(exc), flush=True)
    return report(found, kept, apply), all(not isinstance(v, str) for v in found.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assistant_api.retention", description="the retention periods")
    ap.add_argument("--apply", action="store_true", help="delete (without it: report only)")
    args = ap.parse_args(argv)
    line, complete = asyncio.run(run(load_settings(), args.apply))
    print(line)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
