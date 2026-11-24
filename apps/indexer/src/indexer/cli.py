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
from indexer.crawl.manifest import ManifestDoc, snapshots_dir, write_manifest
from indexer.index import generations as gen
from indexer.settings import Settings, Wiring, wire
from indexer.sources import discover

USAGE = gen.USAGE


SOURCE_SETS = ("core", "full")


def build_options(args: list[str]) -> dict[str, Any] | None:
    """The options of `indexer build [--sources core|full] [--resume] [--reuse-inputs] [--no-swap]`; None for anything else.
    The core set, a fresh crawl, fresh inputs and a swap to the new generation are the defaults. `reuse_inputs` builds from
    what the last build fetched, fetching nothing; `swap` false leaves the new generation staging (`indexer publish <n>`
    makes it live)."""
    options: dict[str, Any] = {"sources": "core", "resume": False, "reuse_inputs": False, "swap": True}
    rest = iter(args)
    for arg in rest:
        if arg == "--resume":
            options["resume"] = True
        elif arg == "--reuse-inputs":
            options["reuse_inputs"] = True
        elif arg == "--no-swap":
            options["swap"] = False
        elif arg == "--sources" or arg.startswith("--sources="):
            chosen = arg.partition("=")[2] if "=" in arg else next(rest, "")
            if chosen not in SOURCE_SETS:
                return None
            options["sources"] = chosen
        else:
            return None
    return options


STATS_NOT_WRITTEN = "the build stats were not written; the next build has no size gate"
LISTED = 20  # lines of one kind the summary prints before it counts the rest


def _capped(level: str, name: str, lines: list[str], more: str) -> list[str]:
    shown = [f"{level}: {name}: {line}" for line in lines[:LISTED]]
    return shown + ([f"{level}: {name}: {len(lines) - LISTED} more {more}"] if len(lines) > LISTED else [])


def summary_lines(name: str, report: Mapping[str, Any]) -> list[str]:
    """What a source's report names for the build summary: a folder that is gone, pages whose content could not be found (the site may have
    changed), documents not indexed for lacking required metadata, PDFs without usable text, hub copies skipped, and URLs not fetched with the reason."""
    return [
        *(
            [f"warning: {name}: folder not found, its documents are not indexed: {report['missing_folder']}"]
            if report.get("missing_folder")
            else []
        ),
        *_capped(
            "warning",
            name,
            [f"empty page {p['url']} ({p['problem']})" for p in report.get("empty_pages", ())],
            "empty pages",
        ),
        *_capped(
            "warning",
            name,
            [f"host down {d['host']}: robots.txt unreadable ({d['reason']})" for d in report.get("hosts_down", ())],
            "hosts down",
        ),
        *_capped(
            "warning",
            name,
            [f"not indexed, lacks {', '.join(d['missing'])}: {d['url']}" for d in report.get("not_indexed", ())],
            "not indexed",
        ),
        *_capped(
            "warning",
            name,
            [f"no usable text, not indexed: {url}" for url in report.get("no_text", ())],
            "PDFs without text",
        ),
        *_capped(
            "warning",
            name,
            [
                f"PDF kept though this crawl did not list it: {url}"
                for url in report.get("pdf_sync", {}).get("retained", ())
            ],
            "PDFs kept",
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


def write_snapshot(crawl_dir: str, generation: int, reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The snapshot manifest of a generation just written (live or staging): the documents the sources' reports list under
    "manifest". The generation exists, so a manifest that cannot be written is a warning."""
    docs = [ManifestDoc(**d) for report in reports.values() for d in report.get("manifest", ())]
    if not docs:
        return []
    try:
        write_manifest(str(generation), docs, directory=snapshots_dir(crawl_dir))
    except OSError as exc:
        return [f"the snapshot manifest of generation {generation} was not written: {exc}"]
    return []


def build_index(
    wiring: Wiring,
    embedding: Any,
    sources: Mapping[str, DocumentSource],
    out: Callable[[str], None],
    options: Mapping[str, Any] | None = None,
) -> int:
    """One build of every source the profile lists: collect, extract, chunk, gate, embed, write, verify, swap the live views
    to the new generation and housekeep, then the stats file (still under the lock) and the counts per source; a held
    lock, a database that cannot be reached, a refusing gate or any failed step leaves the live index as it was and exits 1.
    Without a swap (`swap` false in `options`) the generation stays staging: no housekeeping and no stats, but its snapshot
    manifest is written, so `indexer publish <n>` makes live a generation that has one."""
    settings, target, connect = wiring.settings, wiring.target, wiring.connect
    options = dict(options or {})
    swap, reuse_inputs = options.pop("swap", True), options.pop("reuse_inputs", False)  # the rest is the connectors'
    lock = gen.BuildLock(target, "cli build", connect)
    try:
        lock.acquire()
        built = build.build_corpus(settings, sources, None, reuse_inputs, embedding=embedding, options=options)
        published = asyncio.run(
            build.write_and_publish(
                settings,
                target,
                connect,
                embedding,
                built["units"],
                swap=swap,
                build_minutes=built["minutes_total"],
                prov=build.provenance(settings, built["embedding"]),
            )
        )
        documents = {name: report["documents"] for name, report in built["sources"].items()}
        warnings = write_snapshot(settings.overlay.paths.crawl_dir, published["generation"], built["sources"])
        if swap:
            recorded = gates.record_build_stats(
                settings.overlay.paths.state_dir,
                {
                    "parents": built["parents"],
                    "children": built["children"],
                    "documents": documents,
                    "profile": built["profile"],
                    "sources": built["source_set"],
                    "generation": published["generation"],
                },
            )
            warnings += [] if recorded else [STATS_NOT_WRITTEN]
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
    if skipped := built["gates"]["size"].get("skipped"):
        out(f"note: no size gate: {skipped}")
    counts, embeddings, changes = published["index_counts"], published["embeddings"], published["changes"]
    state = f"published (previous {published['previous_generation'] or 'none'})" if swap else "staged (not published)"
    out(
        f"generation {published['generation']} {state}: "
        f"{counts['parents']} sections, {counts['children']} chunks, {embeddings['computed']} embedded, "
        f"{embeddings['cached']} cached; {changes['added']} added, {changes['changed']} changed, "
        f"{changes['removed']} removed"
    )
    for warning in [*published.get("warnings", ()), *warnings]:
        out(f"warning: {warning}")
    return 0


def publish(
    target: gen.Target,
    n: int,
    connect_fn: Callable[..., Any],
    out: Callable[[str], None],
    state_dir: str,
    profile: str = "",
) -> int:
    """A staging generation goes live: verify again, publish, annotate, record the stats. No housekeeping here; the next
    build housekeeps. The stats name the profile and the generation but no source set: the next build is not size-gated."""
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
    if not gates.record_build_stats(
        state_dir,
        {"parents": counts["parents"], "children": counts["children"], "profile": profile, "generation": n},
    ):
        out(f"warning: {STATS_NOT_WRITTEN}")
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
        return publish(
            wiring.target,
            int(args[1]),
            wiring.connect,
            out,
            wiring.settings.overlay.paths.state_dir,
            wiring.settings.profile_name,
        )
    return gen.main(
        args,
        target=wiring.target,
        connect_fn=wiring.connect,
        out=out,
        on_rollback=lambda conn: record_live_stats(conn, wiring, out),
    )


def record_live_stats(conn: Any, wiring: Wiring, out: Callable[[str], None]) -> None:
    """After a rollback: the stats become those of the generation that is live again, so the next build is not measured
    against the build that was rolled back. The generation's source set is not known, so that build is not size-gated."""
    live = next((r for r in gen.generations(conn, wiring.target) if r["status"] == "live"), None)
    counts = (live or {}).get("counts") or {}
    if not gates.record_build_stats(
        wiring.settings.overlay.paths.state_dir,
        {k: counts[k] for k in ("parents", "children") if k in counts}
        | {"profile": wiring.settings.profile_name, "generation": (live or {}).get("generation")},
    ):
        out(f"warning: {STATS_NOT_WRITTEN}")
