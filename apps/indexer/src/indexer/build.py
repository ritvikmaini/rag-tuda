"""
Corpus pipeline: one build of every source the profile lists, chunked into one list of parents for write_v2.

The sources are the profile's connectors, the values come from the profile, and the embedding server's /info is
checked and recorded once per build.
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rag_core.domain.sources import PageItem, PdfItem
from rag_core.ports.sources import DocumentSource
from rag_core.settings.profile import Embedding

from indexer import gates
from indexer.chunking.units import Parent, chunk_documents
from indexer.crawl.manifest import manifest_urls, snapshots_dir
from indexer.extract.documents import pdf_documents
from indexer.extract.pdf import extract_pdfs
from indexer.index import cache
from indexer.index import generations as gen
from indexer.index.writer import write_v2
from indexer.settings import Settings
from indexer.sources import context


TIMINGS = ("write_s", "verify_s", "publish_s")  # the timings write_and_publish returns


class BuildRefused(RuntimeError):
    """A gate or the embedding check refused the build (`indexer build` prints "refused"; any other error "failed").
    `reports` are the sources' reports so far, which the command still prints: they name the pages behind a build
    that got less than the last one."""

    def __init__(self, message: str, reports: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.reports = reports or {}


class EmbeddingMismatch(BuildRefused):
    """The embedding server serves another model or dimension than the profile names."""


def inventory_entry(item: PdfItem) -> dict:
    """The inventory dict extract_pdfs and pdf_documents read, from a connector's PdfItem."""
    return {
        "filename": Path(item.path).name,
        "path": item.path,
        "file_hash": item.file_hash,
        "url": item.metadata.get("url", ""),
        "download_url": item.url,
        "download_hash": item.metadata.get("download_hash", ""),
        "download_title": item.metadata.get("download_title", ""),
    }


def check_embedding(profile: Embedding, info: Mapping[str, Any] | None) -> None:
    if info is None:
        logging.warning(
            "[Corpus] the embedding server has no /info (an older server image); provenance recorded as null"
        )
        return
    problems = []
    if info.get("model") != profile.model:
        problems.append(f"model {info.get('model')!r} (profile {profile.model!r})")
    if info.get("dimension") != profile.dimensions:
        problems.append(f"dimension {info.get('dimension')} (profile {profile.dimensions})")
    if problems:
        raise EmbeddingMismatch("the embedding server serves " + " and ".join(problems) + "; build refused")


def provenance(settings: Settings, info: Mapping[str, Any] | None) -> dict[str, Any]:
    """The generation row's provenance columns (index_generations)."""
    return {
        "git_commit": os.environ.get("GIT_COMMIT") or None,
        "config_hash": settings.config_hash(),
        "embedding_model": info.get("model") if info else None,
        "embedding_revision": info.get("revision") if info else None,
    }


def build_corpus(
    settings: Settings,
    sources: Mapping[str, DocumentSource],
    limit: int | None = None,
    reuse_inputs: bool = False,
    *,
    embedding: Any,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect every source in profile order, extract its PDFs, chunk and gate; the Parents go back under "units". Refuses
    before any download on a corrupt stats file or connector state and on an embedding server that serves another model.
    Builds with limit are never published and therefore not gated. `options` go to every source's context."""
    started = time.monotonic()
    last = gates.previous_stats(settings.overlay.paths.state_dir)
    built_set = gates.source_set(options, sources)
    previous = gates.comparable(last, settings.profile_name, built_set)
    contexts = {name: context(settings, name, limit, reuse_inputs, options) for name in sources}
    for name, source in sources.items():
        for state_file in source.state_files:
            contexts[name].state.load(state_file)
    info = embedding.info()
    check_embedding(settings.profile.retrieval.embedding, info)
    ingest = settings.profile.ingest
    docs, reports, checks = [], {}, {}
    minutes = {"collect": 0.0, "extract": 0.0, "chunk": 0.0}
    seen_hashes: set[str] = set()
    for name, source in sources.items():
        t = time.monotonic()
        result = source.collect(contexts[name])
        minutes["collect"] += time.monotonic() - t
        connector = settings.profile.ingest.sources[name].connector
        pages = [
            {"page_content": i.text, "metadata": dict(i.metadata), "source": connector}
            for i in result.items
            if isinstance(i, PageItem)
        ]
        pdfs = [i for i in result.items if isinstance(i, PdfItem)]
        report = {**result.report, "documents": len(pages)}
        count, kind = len(pages), "documents"
        if pdfs:
            t = time.monotonic()
            paths = settings.overlay.paths
            extracted = extract_pdfs(
                inventory=[inventory_entry(i) for i in pdfs],
                extraction=ingest.extraction,
                extracted_dir=paths.extracted_dir,
                skipped_log=paths.skipped_pdfs_log,
            )
            pdf_docs, missing = pdf_documents(
                pdfs, seen_hashes, extracted_dir=paths.extracted_dir, clean=ingest.extraction.clean_markup
            )
            minutes["extract"] += time.monotonic() - t
            hashes = {d["metadata"]["file_hash"] for d in pdf_docs}
            count, kind = sum(1 for i in pdfs if i.file_hash in hashes), "pdfs"
            pages += [{**d, "source": connector} for d in pdf_docs]
            report |= {
                "documents": len(pages),
                "with_documents": count,
                "missing_cache": missing,
                "extract": {k: extracted.get(k) for k in ("processed", "cached", "failed", "skipped_page_limit")},
            }
        docs += pages
        reports[name] = report
        checks[name] = gates.source_gate(name, count, result.expected_minimum, kind)
    t = time.monotonic()
    chunking = ingest.chunking
    parents = chunk_documents(
        docs,
        mode=chunking.mode,
        parent_chars=chunking.parent_chars,
        children=chunking.children,
        header=chunking.header,
        parent_overlap=chunking.parent_overlap,
        child_chars=chunking.child_chars,
        child_overlap=chunking.child_overlap,
        scope=ingest.scope,
    )
    minutes["chunk"] = time.monotonic() - t
    children = sum(len(p.children) for p in parents)
    checks["hosts"] = gates.hosts_gate(reports)
    checks["size"] = gates.size_gate(len(parents), previous, ingest.collections[settings.collection].min_build_ratio)
    generation = previous.get("generation")
    live_urls = (
        manifest_urls(str(generation), directory=snapshots_dir(settings.overlay.paths.crawl_dir)) if generation else ()
    )
    checks["pages"] = gates.pages_gate(reports, live_urls, generation)
    if last and not previous:
        checks["size"]["skipped"] = "the last build was of another profile or source set"
    refused = None if limit else gates.refusal(checks)
    if refused:
        raise BuildRefused(refused, reports)
    result = {
        "status": "success",
        "reused": reuse_inputs,
        "sources": reports,
        "gates": checks,
        "parents": len(parents),
        "units": parents,  # the Parents themselves; "parents" is their count
        "children": children,
        "profile": settings.profile_name,
        "source_set": built_set,
        "embedding": info,
        "minutes": {k: round(v / 60, 1) for k, v in minutes.items()},
        "minutes_total": round((time.monotonic() - started) / 60, 1),
    }
    summary = {
        "documents": {name: report["documents"] for name, report in reports.items()},
        "parents": len(parents),
        "children": children,
        "minutes": result["minutes_total"],
    }
    logging.info(f"[Corpus] build finished: {json.dumps(summary, ensure_ascii=False)}")
    return result


async def write_and_publish(
    settings: Settings,
    target: gen.Target,
    connect_fn: Callable[..., Any],
    embedding: Any,
    parents: list[Parent],
    *,
    swap: bool,
    build_minutes: float | None,
    prov: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """begin -> seed the cache -> write -> verify -> change report -> publish -> housekeep. A failed write, verify or
    publish marks the generation failed and leaves the views as they are; swap=false keeps it as staging until the next
    publish. Once the publish has committed nothing raises: a failed check, annotation or housekeeping is logged and
    listed under "warnings", and the caller still writes the stats. The reply carries write_s,
    verify_s and (after a publish) publish_s."""

    async def step(fn, *args, **kwargs):
        return await asyncio.to_thread(gen.with_connection, fn, *args, connect_fn=connect_fn, **kwargs)

    n, _ = await step(gen.begin, target, prov)
    cache_key = cache.key(
        target.collection, settings.profile.retrieval.embedding, (prov or {}).get("embedding_revision")
    )
    started = time.monotonic()
    warnings = []
    try:
        await step(cache.seed, cache_key)
        written = await step(
            write_v2,
            embedding.embed_passages,
            target.tables(n),
            parents,
            settings.profile,
            generation=n,
            cache_key=cache_key,
        )
        write_s = round(time.monotonic() - started, 1)
        index_counts = await step(gen.verify, target, n, written)
        index_s = round(time.monotonic() - started, 1)
        verify_s = round(index_s - write_s, 1)
        changes = await step(gen.document_changes, target, n)
        previous, publish_s = None, None  # set by the publish; read only when swap
        if swap:
            try:
                previous, publish_s = await step(gen.publish, target, n)
            except gen.PublishedUnchecked as exc:  # committed: the views read generation n
                previous, publish_s = exc.previous, exc.seconds
                warnings.append(str(exc))
    except Exception as exc:
        try:
            await step(gen.mark_failed, target, n, f"{type(exc).__name__}: {exc}")
        except Exception as mark_exc:
            logging.error(f"[Generations] could not mark generation {n} failed: {mark_exc}")
        raise
    timings = {"build_minutes": build_minutes, "index_s": index_s}
    reply: dict[str, Any] = {
        "generation": n,
        "generation_table": target.table(n),
        "index_counts": index_counts,
        "embeddings": written["embeddings"],
        "changes": changes,
    }
    if not swap:
        await step(gen.annotate, target, n, counts=index_counts, timings=timings)
        return {**reply, "swapped": False, "staging_table": target.table(n), "write_s": write_s, "verify_s": verify_s}

    async def after_publish(fn, *args, **kwargs):
        try:
            return await step(fn, *args, **kwargs)
        except Exception as exc:
            warnings.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")

    await after_publish(gen.annotate, target, n, counts=index_counts, timings={**timings, "publish_s": publish_s})
    housekeeping = await after_publish(gen.housekeep, target)
    for warning in warnings:
        logging.error(f"[Generations] generation {n} is live; after the publish: {warning}")
    published = {**reply, "swapped": True, "previous_generation": previous, "housekeeping": housekeeping}
    if warnings:
        published["warnings"] = warnings
    return {**published, "write_s": write_s, "verify_s": verify_s, "publish_s": publish_s}
