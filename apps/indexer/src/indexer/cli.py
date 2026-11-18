"""The indexer's CLI: build, publish and, from the generations module, status, housekeep, rollback and compare. In the
container: indexer build."""

import asyncio
import logging
import sys
import traceback
from collections.abc import Callable, Mapping
from typing import Any

from rag_core.adapters.embedding_http import EmbeddingClient
from rag_core.ports.sources import DocumentSource
from rag_core.settings import SettingsError

from indexer import build, gates
from indexer.index import generations as gen
from indexer.settings import Settings, Wiring, wire
from indexer.sources import discover

USAGE = gen.USAGE


SOURCE_SETS = ("core", "full")


def build_options(args: list[str]) -> dict[str, Any] | None:
    """The options of `indexer build [--sources core|full] [--resume]`; None for anything else. The core set is the default,
    a fresh crawl the other."""
    options: dict[str, Any] = {"sources": "core", "resume": False}
    rest = iter(args)
    for arg in rest:
        if arg == "--resume":
            options["resume"] = True
        elif arg == "--sources" or arg.startswith("--sources="):
            chosen = arg.partition("=")[2] if "=" in arg else next(rest, "")
            if chosen not in SOURCE_SETS:
                return None
            options["sources"] = chosen
        else:
            return None
    return options


LISTED = 20  # lines of one kind the summary prints before it counts the rest


def _capped(level: str, name: str, lines: list[str], more: str) -> list[str]:
    shown = [f"{level}: {name}: {line}" for line in lines[:LISTED]]
    return shown + ([f"{level}: {name}: {len(lines) - LISTED} more {more}"] if len(lines) > LISTED else [])


def summary_lines(name: str, report: Mapping[str, Any]) -> list[str]:
    """What a source's report names for the build summary: pages whose content could not be found (the site may have
    changed), documents not indexed for lacking required metadata, hub copies skipped, and URLs not fetched with the reason."""
    return [
        *_capped(
            "warning",
            name,
            [f"empty page {p['url']} ({p['problem']})" for p in report.get("empty_pages", ())],
            "empty pages",
        ),
        *_capped(
            "warning",
            name,
            [f"not indexed, lacks {', '.join(d['missing'])}: {d['url']}" for d in report.get("not_indexed", ())],
            "not indexed",
        ),
        *_capped(
            "note",
            name,
            [f"copy skipped, its original is indexed: {url}" for url in report.get("copies", ())],
            "copies skipped",
        ),
        *_capped(
            "note",
            name,
            [f"not fetched ({s['reason']}): {s['url']}" for s in report.get("skipped", ())],
            "not fetched",
        ),
    ]


def build_index(
    wiring: Wiring,
    embedding: Any,
    sources: Mapping[str, DocumentSource],
    out: Callable[[str], None],
    options: Mapping[str, Any] | None = None,
) -> int:
    """One build of every source the profile lists: collect, extract, chunk, gate, embed, write, verify, swap the live views
    to the new generation and housekeep, then the stats file (still under the lock) and the counts per source; a held
    lock, a database that cannot be reached, a refusing gate or any failed step leaves the live index as it was and exits 1."""
    settings, target, connect = wiring.settings, wiring.target, wiring.connect
    lock = gen.BuildLock(target, "cli build", connect)
    try:
        lock.acquire()
        built = build.build_corpus(settings, sources, None, False, embedding=embedding, options=options)
        published = asyncio.run(
            build.write_and_publish(
                settings,
                target,
                connect,
                embedding,
                built["units"],
                swap=True,
                build_minutes=built["minutes_total"],
                prov=build.provenance(settings, built["embedding"]),
            )
        )
        documents = {name: report["documents"] for name, report in built["sources"].items()}
        gates.record_build_stats(
            settings.overlay.paths.state_dir,
            {"parents": built["parents"], "children": built["children"], "documents": documents},
        )
    except gen.BuildBusy as exc:
        out(f"busy: {exc}")
        return 1
    except build.BuildRefused as exc:
        for name, report in exc.reports.items():
            for line in summary_lines(name, report):
                out(line)
        out(f"refused: {exc}")
        return 1
    except Exception as exc:
        logging.error(f"Corpus pipeline failed: {traceback.format_exc()}")
        out(f"failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        lock.release()
    for name, count in documents.items():
        out(f"source {name}: {count} documents")
        for line in summary_lines(name, built["sources"][name]):
            out(line)
    counts, embeddings, changes = published["index_counts"], published["embeddings"], published["changes"]
    out(
        f"generation {published['generation']} published (previous {published['previous_generation'] or 'none'}): "
        f"{counts['parents']} sections, {counts['children']} chunks, {embeddings['computed']} embedded, "
        f"{embeddings['cached']} cached; {changes['added']} added, {changes['changed']} changed, "
        f"{changes['removed']} removed"
    )
    for warning in published.get("warnings", ()):
        out(f"warning: {warning}")
    return 0


def publish(
    target: gen.Target, n: int, connect_fn: Callable[..., Any], out: Callable[[str], None], state_dir: str
) -> int:
    """A staging generation goes live: verify again, publish, annotate, record the stats. No housekeeping here; the next
    build housekeeps."""
    lock = gen.BuildLock(target, f"cli publish {n}", connect_fn)
    try:
        lock.acquire()
    except gen.BuildBusy as exc:
        out(f"busy: {exc}")
        return 1
    try:
        conn = connect_fn(application_name="indexer-cli publish")
        try:
            row = next((r for r in gen.generations(conn, target) if r["generation"] == n), None)
            if row is None or row["status"] != "staging":
                out(f"refused: generation {n} is {row['status'] if row else 'unknown'}, not staging")
                return 1
            counts = gen.verify(conn, target, n)
            try:
                previous, seconds = gen.publish(conn, target, n)
            except gen.PublishedUnchecked as exc:  # committed: the views read generation n
                previous, seconds = exc.previous, exc.seconds
                out(f"warning: {exc}")
            try:
                gen.annotate(conn, target, n, counts=counts, timings={"publish_s": seconds})
            except Exception as exc:  # the publish is committed: the stats still follow
                out(f"warning: annotate: {exc}")
        finally:
            conn.close()
    except gen.GenerationError as exc:
        out(f"refused: {exc}")
        return 1
    finally:
        lock.release()
    gates.record_build_stats(state_dir, {"parents": counts["parents"], "children": counts["children"]})
    out(f"published generation {n} (previous {previous}) in {seconds} s")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    connect_fn: Callable[..., Any] | None = None,
    embedding: Any = None,
    sources: Mapping[str, DocumentSource] | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """The console script `indexer` (sys.exit(main()) through [project.scripts]). embedding and sources are the build's
    clients and connectors, made from the settings when not given."""
    if argv is None:  # the console script: the [Generations] log lines on stderr, as python -m printed them
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["--help"], ["-h"]):
        out(USAGE)
        return 0
    try:
        return _dispatch(args, settings, connect_fn, embedding, sources, out)
    except SettingsError as exc:  # a missing secret file or a bad setting: the message names the key or file
        out(f"failed: {exc}")
        return 1


def _dispatch(
    args: list[str],
    settings: Settings | None,
    connect_fn: Callable[..., Any] | None,
    embedding: Any,
    sources: Mapping[str, DocumentSource] | None,
    out: Callable[[str], None],
) -> int:
    wiring = wire(settings, connect_fn)
    if args[:1] == ["build"]:
        options = build_options(args[1:])
        if options is None:
            out(USAGE)
            return 2
        try:
            embedding = embedding or EmbeddingClient(
                wiring.settings.overlay.model_servers.embedding_url,
                passage_prefix=wiring.settings.profile.retrieval.embedding.passage_prefix,
            )
            sources = discover(wiring.settings.profile) if sources is None else sources
        except Exception as exc:  # a missing connector or an unreadable setting
            out(f"failed: {type(exc).__name__}: {exc}")
            return 1
        return build_index(wiring, embedding, sources, out, options)
    if args[:1] == ["publish"]:
        if len(args) != 2 or not args[1].isdigit():
            out(USAGE)
            return 2
        return publish(wiring.target, int(args[1]), wiring.connect, out, wiring.settings.overlay.paths.state_dir)
    return gen.main(args, target=wiring.target, connect_fn=wiring.connect, out=out)
