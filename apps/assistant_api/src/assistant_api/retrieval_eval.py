"""Where the gold source ranks for each probe question, after fusion and after reranking, and the ablation of the retrieval stages.
Read-only, run natively with the backend's settings (`rag eval --retrieval-only [--ablation]`) or inside the backend container:

  python -m assistant_api.retrieval_eval [--ablation] [--sets a,b] [--limit N]

A conversation takes part when its first turn carries `"gold": {"url": ..., "heading": ... | "page": N, "quote": ...}`: the hit is
the gold when its url (a `#` fragment ignored) equals the gold's and, when the gold names a heading or a page, its section path
contains the heading / its page number equals the page. The rank is the 1-based position of the gold among the sections of the fused
list the reranker is given (a section once, at its best child) and among the reranked sections the retrieval returns. One table goes
to stdout: hit@1/3/5/10 and MRR per stage (a plain run), or per ablation row after reranking. Never a question, a query or a
document. Logs go to stderr. A conversation without a gold is counted and left out.

The analysis runs once per distinct question per run, with the filter readers on (the profile's analysis_temperature is 0); the
`filters` switch of a row only decides whether the resolved filter is applied. Each ablation row sets the five switches of
`ORDER` (`retrieval.switches` of the profile): the first row has all off, each next row turns one more on, in the order of ORDER.
The retrieval of a row is a service built with those switches; `acronyms` is read when the index is opened, so every row opens its
own. Exit 0 when a conversation with a gold was scored, 1 when none could be."""

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from rag_core.adapters.http import close_http_client
from rag_core.adapters.pg_index import open_index
from rag_core.domain.access import ANONYMOUS
from rag_core.domain.answer_shapes import answer_language
from rag_core.domain.filters import Filter
from rag_core.domain.resilience import dispose_engines
from rag_core.services.retrieval import RetrievalService
from rag_core.settings import SettingsError
from rag_core.settings.profile import Profile

from assistant_api.graph.history import question_message
from assistant_api.graph.nodes.analyse import make_analyse
from assistant_api.graph.state import GraphDeps
from assistant_api.runtime import Runtime, build_runtime, graph_deps
from assistant_api.scope_scores import file_sets, first_question
from assistant_api.settings import load_settings

logger = logging.getLogger(__name__)

KS = (1, 3, 5, 10)
ORDER = ("filters", "acronyms", "fusion_cut", "whole_parents", "reference_hop")
LABELS = ("no filters", "+ filters", "+ acronyms", "+ fusion cut", "+ whole parents", "+ reference hop")
ABLATION: tuple[tuple[str, dict[str, bool]], ...] = tuple(
    (label, {name: name in ORDER[:i] for name in ORDER}) for i, label in enumerate(LABELS)
)
PLAIN: tuple[tuple[str, dict[str, bool]], ...] = (("profile", {}),)  # the profile's own switches


def gold_of(conversation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The gold of the first turn, None without one."""
    gold = conversation["turns"][0].get("gold")
    return gold if isinstance(gold, Mapping) else None


def is_gold(gold: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    """Whether a section with this metadata is the gold: the url, then the heading in the section path or the page."""
    if str(metadata.get("url") or "").partition("#")[0] != gold["url"].partition("#")[0]:
        return False
    if "heading" in gold:
        return gold["heading"] in str(metadata.get("section_path") or "")
    if "page" in gold:
        return metadata.get("page_number") == gold["page"]
    return True


def rank_of(gold: Mapping[str, Any], sections: Sequence[Mapping[str, Any]]) -> int | None:
    """The 1-based position of the first gold section, None when it is not in the list."""
    return next((i for i, metadata in enumerate(sections, 1) if is_gold(gold, metadata)), None)


def summarise(ranks: Sequence[int | None]) -> dict[str, float]:
    """hit@k for k in KS and the mean reciprocal rank (a missing gold counts 0); all 0.0 without ranks."""
    n = len(ranks) or 1
    return {
        **{f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS},
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


class Traced:
    """A RetrievalService that also keeps the fused list: the children the reranker was given, in the order it got them."""

    def __init__(self, service: RetrievalService) -> None:
        self.service = service
        self.children: list[Any] = []
        rerank = service._rerank  # the one point where the fused list is complete and not yet scored

        async def spy(queries: list[str], documents: list[Any], parents: dict[str, str]) -> list[Any]:
            self.children = list(documents)
            return await rerank(queries, documents, parents)

        service._rerank = spy  # type: ignore[method-assign]

    async def places(self, queries: list[str], filter: Filter | None, **kwargs: Any) -> tuple[list[Any], list[Any]]:
        """The metadata of the fused list's sections (once each, in order) and of the sections the retrieval returns."""
        self.children = []
        final = await self.service.retrieve(queries, filter, **kwargs)
        ids = list(dict.fromkeys(i for c in self.children if (i := c.metadata.get("section_id"))))
        found = await self.service.sections.get_many(ids, None)
        return [found[i].metadata for i in ids if i in found], [d.metadata for d in final]


OpenRow = Callable[[Mapping[str, bool]], AbstractAsyncContextManager[Traced]]


class Analyses:
    """The analysis node's state per question, asked once per run. The node runs with the filter readers on."""

    def __init__(self, deps: GraphDeps) -> None:
        analysis_deps = dataclasses.replace(deps, profile=with_switches(deps.profile, {"filters": True}))
        self.node: Any = make_analyse(analysis_deps)  # Any: fed the question's fields only, as scope_scores does
        self.states: dict[str, dict[str, Any]] = {}

    async def of(self, question: str) -> dict[str, Any]:
        if question not in self.states:
            messages = [question_message(question, None, datetime.now(timezone.utc))]
            self.states[question] = await self.node({"messages": messages, "formats": ""})
        return self.states[question]


def with_switches(profile: Profile, overrides: Mapping[str, bool]) -> Profile:
    """The profile with these retrieval switches set."""
    retrieval = profile.retrieval
    switches = retrieval.switches.model_copy(update=dict(overrides))
    return profile.model_copy(update={"retrieval": retrieval.model_copy(update={"switches": switches})})


class Tally:
    def __init__(self) -> None:
        self.fused: list[int | None] = []
        self.reranked: list[int | None] = []
        self.failed = 0


async def evaluate(
    deps: GraphDeps,
    open_row: OpenRow,
    profile_dir: Path,
    names: Sequence[str] | None,
    limit: int | None,
    rows: Sequence[tuple[str, Mapping[str, bool]]],
    out: TextIO,
) -> int:
    """Writes the table of the rows over the chosen sets (gate.yaml's file sets by default; at most `limit` conversations of each)."""
    known = file_sets(profile_dir)
    chosen = list(names) if names else list(known)
    if unknown := [n for n in chosen if n not in known]:
        logger.error("not a set with a probe file of its own: %s; known: %s", ", ".join(unknown), ", ".join(known))
        return 1
    conversations = [
        c for name in chosen for c in json.loads(known[name].read_text(encoding="utf-8"))["conversations"][:limit]
    ]
    golds = [(first_question(c), g) for c in conversations if (g := gold_of(c))]
    logger.info("[RetrievalEval] %d conversations with a gold, %d without", len(golds), len(conversations) - len(golds))
    analyses = Analyses(deps)
    tallies: dict[str, Tally] = {}
    for label, overrides in rows:
        tally = tallies[label] = Tally()
        switches = {**deps.profile.retrieval.switches.model_dump(), **overrides}
        async with open_row(overrides) as traced:
            for question, gold in golds:
                try:
                    if len(question) > deps.profile.guard.max_question_chars:
                        raise ValueError("too_long")  # such a question never reaches the analysis
                    state = await analyses.of(question)
                    fused, final = await traced.places(
                        state["retrieval_queries"],
                        state.get("filter") if switches["filters"] else None,
                        question=state["standalone_question"] or question,
                        date_from=state["date_from"],
                        date_to=state["date_to"],
                        principal=ANONYMOUS,
                        language=answer_language(state),
                    )
                except Exception as exc:  # the class only: a message may echo a question
                    logger.error("[RetrievalEval] row %s: %s", label, type(exc).__name__)
                    tally.failed += 1
                    continue
                tally.fused.append(rank_of(gold, fused))
                tally.reranked.append(rank_of(gold, final))
    out.write(table(rows, tallies, len(conversations) - len(golds)))
    out.flush()
    return 0 if any(t.reranked for t in tallies.values()) else 1


def table(rows: Sequence[tuple[str, Mapping[str, bool]]], tallies: Mapping[str, Tally], no_gold: int) -> str:
    """The plain table: one row (a plain run) shows both stages, several (an ablation) the rows after reranking."""
    lines = [("stage", "n", "failed", *(f"hit@{k}" for k in KS), "MRR")]
    for label, _ in rows:
        t = tallies[label]
        stages = (
            (("after fusion", t.fused), ("after reranking", t.reranked)) if len(rows) == 1 else ((label, t.reranked),)
        )
        for stage, ranks in stages:
            s = summarise(ranks)
            lines.append(
                (stage, str(len(ranks)), str(t.failed), *(f"{s[f'hit@{k}']:.2f}" for k in KS), f"{s['mrr']:.3f}")
            )
    widths = [max(len(line[i]) for line in lines) for i in range(len(lines[0]))]
    body = [
        "  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(line, widths)))
        for line in lines
    ]
    return f"{no_gold} conversations without a gold left out\n" + "\n".join(body) + "\n"


def row_opener(rt: Runtime) -> OpenRow:
    """Opens the retrieval of a row: the profile with the row's switches, its own index (the acronyms are read at opening)."""
    settings = rt.settings

    @asynccontextmanager
    async def open_row(overrides: Mapping[str, bool]) -> AsyncIterator[Traced]:
        profile = with_switches(settings.profile, overrides)
        row = dataclasses.replace(rt, settings=dataclasses.replace(settings, profile=profile))
        index = await open_index(rt.collection, profile, settings.overlay.database, settings.acronyms)
        try:
            yield Traced(graph_deps(row, index).retrieval)  # type: ignore[arg-type]
        finally:
            await dispose_engines(*index.engines)

    return open_row


async def run(args: argparse.Namespace, environ: Mapping[str, str], secrets_dir: str | Path | None) -> int:
    settings = load_settings(environ, secrets_dir=secrets_dir)
    rt = build_runtime(settings, None, False)  # no Langfuse client: nothing is traced
    index = await open_index(rt.collection, settings.profile, settings.overlay.database, settings.acronyms)
    try:  # this index serves the analysis' deps only; every row opens its own
        return await evaluate(
            graph_deps(rt, index),
            row_opener(rt),
            settings.profile_dir,
            args.sets,
            args.limit,
            ABLATION if args.ablation else PLAIN,
            sys.stdout,
        )
    finally:
        await dispose_engines(*index.engines)
        await close_http_client()


def main(argv: Sequence[str], *, environ: Mapping[str, str] = os.environ, secrets_dir: str | Path | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant_api.retrieval_eval")
    parser.add_argument("--ablation", action="store_true", help="one row per retrieval stage turned on, cumulatively")
    parser.add_argument(
        "--sets", type=lambda value: [n for n in value.split(",") if n], help="comma-separated set names"
    )
    parser.add_argument("--limit", type=int, help="only the first N conversations of each set (a test run)")
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return asyncio.run(run(args, environ, secrets_dir))
    except SettingsError as exc:  # names a key or a file, never a value
        logger.error("[RetrievalEval] could not run: SettingsError: %s", exc)
        return 1
    except Exception as exc:  # the class only: a message may echo a question
        logger.error("[RetrievalEval] could not run: %s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
