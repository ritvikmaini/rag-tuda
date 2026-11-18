"""Measure the reranker's best score of every probe conversation's first question. Read-only, run inside the running
backend container; `retrieval.rerank.scope_fallback_min` is set from its output:

  docker compose --env-file .env -f deploy/compose.yaml exec -T backend python -m assistant_api.scope_scores [--sets a,b] [--limit N]

For each conversation of the chosen sets of gate.yaml (default: every set with a file of its own; panels are views and add nothing)
it runs the analyse node and `deps.retrieval.retrieve` on the first turn exactly as a turn does, and writes one JSON line to stdout:
`set`, `id`, `intent`, `top1` (null without hits), `top3`, `hits`, `error` (the exception's class name, null when it worked). Never a
question, a query, a standalone text or a document (a gold set may hold real users' questions). Logs go to stderr. It bypasses the
backend's request slot: run it with no probe run beside it. Exit 0 when a row was scored, 1 when nothing could be."""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml
from rag_core.adapters.http import close_http_client
from rag_core.adapters.pg_index import open_index
from rag_core.domain.access import ANONYMOUS
from rag_core.domain.doc_scores import sort_score
from rag_core.domain.resilience import dispose_engines
from rag_core.settings import SettingsError

from assistant_api.graph.history import question_message
from assistant_api.graph.nodes.analyse import make_analyse
from assistant_api.graph.state import GraphDeps
from assistant_api.runtime import build_runtime, graph_deps
from assistant_api.settings import load_settings

logger = logging.getLogger(__name__)


def file_sets(profile_dir: Path) -> dict[str, Path]:
    """gate.yaml's sets that have a probe file of their own (an entry with `ids` is a panel: a view on another file)."""
    sets = yaml.safe_load((profile_dir / "gate.yaml").read_text(encoding="utf-8"))["sets"]
    return {name: profile_dir / entry["file"] for name, entry in sets.items() if not entry.get("ids")}


def first_question(conversation: Mapping[str, Any]) -> str:
    """The first turn's text; `repeat` builds a long input without storing it (as the evals runner does)."""
    turn = conversation["turns"][0]
    repeat = turn.get("repeat")
    return (repeat["text"] + " ") * int(repeat["times"]) + turn["q"] if repeat else turn["q"]


async def score_one(deps: GraphDeps, analyse: Any, question: str) -> dict[str, Any]:
    """The row's fields for one question; a failure keeps what was found so far and names the exception's class."""
    row: dict[str, Any] = {"intent": None, "top1": None, "top3": [], "hits": None, "error": None}
    if len(question) > deps.profile.guard.max_question_chars:
        row["error"] = "too_long"  # such a question never reaches the analysis
        return row
    try:
        state = await analyse(
            {"messages": [question_message(question, None, datetime.now(timezone.utc))], "formats": ""}
        )
        row["intent"] = state["intent"]
        hits = await deps.retrieval.retrieve(
            state["retrieval_queries"],
            question=state["standalone_question"] or question,
            date_from=state["date_from"],
            date_to=state["date_to"],
            principal=ANONYMOUS,
        )
    except Exception as exc:
        row["error"] = type(exc).__name__
        return row
    scores = sorted((sort_score(h.metadata) for h in hits), reverse=True)  # the node's top-1 rule (None counts as 0.0)
    row.update(top1=scores[0] if scores else None, top3=scores[:3], hits=len(scores))
    return row


async def score(deps: GraphDeps, profile_dir: Path, names: Sequence[str] | None, limit: int | None, out: TextIO) -> int:
    """Writes the rows of the chosen sets (gate.yaml's file sets by default; at most `limit` conversations of each) to out."""
    known = file_sets(profile_dir)
    chosen = list(names) if names else list(known)
    if unknown := [n for n in chosen if n not in known]:
        logger.error("not a set with a probe file of its own: %s; known: %s", ", ".join(unknown), ", ".join(known))
        return 1
    analyse = make_analyse(deps)
    scored = 0
    for name in chosen:
        conversations = json.loads(known[name].read_text(encoding="utf-8"))["conversations"][:limit]
        logger.info("[ScopeScores] set %s: %d conversations", name, len(conversations))
        for conversation in conversations:
            row = {
                "set": name,
                "id": conversation["id"],
                **await score_one(deps, analyse, first_question(conversation)),
            }
            scored += row["error"] is None
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    logger.info("[ScopeScores] %d rows scored", scored)
    return 0 if scored else 1


async def run(args: argparse.Namespace, environ: Mapping[str, str], secrets_dir: str | Path | None) -> int:
    settings = load_settings(environ, secrets_dir=secrets_dir)
    rt = build_runtime(settings, None, False)  # no Langfuse client: nothing is traced
    index = await open_index(rt.collection, settings.profile, settings.overlay.database)
    try:
        return await score(graph_deps(rt, index), settings.profile_dir, args.sets, args.limit, sys.stdout)
    finally:
        await dispose_engines(*index.engines)
        await close_http_client()


def main(argv: Sequence[str], *, environ: Mapping[str, str] = os.environ, secrets_dir: str | Path | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant_api.scope_scores")
    parser.add_argument(
        "--sets", type=lambda value: [n for n in value.split(",") if n], help="comma-separated set names"
    )
    parser.add_argument("--limit", type=int, help="only the first N conversations of each set (a test run)")
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return asyncio.run(run(args, environ, secrets_dir))
    except SettingsError as exc:  # names a key or a file, never a value
        logger.error("[ScopeScores] could not run: SettingsError: %s", exc)
        return 1
    except Exception as exc:  # the class only: a message may echo a question
        logger.error("[ScopeScores] could not run: %s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
