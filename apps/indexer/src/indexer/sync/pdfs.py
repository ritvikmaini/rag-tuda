"""Sync the PDFs of a source's list into the PDF folder through the context's fetcher.

A PDF is fetched again only when the hash its list publishes differs from the registry's (an entry without a hash is
fetched every run); a failed download never records a hash, and keeps the file it replaces. A full sync also runs the
list safeguard, so a PDF missing from one list is kept for a few runs."""

import logging
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

from rag_core.domain.sources import CorruptStateError
from rag_core.ports.sources import Fetcher, SourceContext

from indexer.sync import registry as reg
from indexer.sync.registry import PdfEntry, file_hash, local_path
from indexer.sync.safeguard import LIST_STATE, Safeguard

__all__ = ["PDF_DOWNLOAD_TIMEOUT", "PdfEntry", "Safeguard", "SyncResult", "sync_pdfs"]

PDF_DOWNLOAD_TIMEOUT = 300


@dataclass
class SyncResult:
    entries: list[PdfEntry]  # what to build from: this run's entries (the first `limit`) plus the retained ones
    retained: list[PdfEntry]  # the entries kept by the safeguard although this run's list lacks them
    results: list[dict]  # per entry: url, file, status (downloaded, cached, retained_on_error, failed, missing), error
    removed: int = 0  # links the safeguard dropped, with their files
    check: dict = field(default_factory=dict)  # the safeguard's list check; error and corrupt when it failed

    def _count(self, *statuses: str) -> int:
        counts = Counter(r["status"] for r in self.results)
        return sum(counts[s] for s in statuses)

    @property
    def downloaded(self) -> int:
        return self._count("downloaded")

    @property
    def unchanged(self) -> int:
        return self._count("cached", "retained_on_error")

    @property
    def failed(self) -> int:
        return self._count("failed", "retained_on_error")

    @property
    def missing(self) -> int:
        return self._count("missing")


def sync_pdfs(
    entries: list[PdfEntry],
    ctx: SourceContext,
    *,
    pdf_dir: str | Path | None = None,
    allow_empty: bool = False,
    dropped: Collection[str] = (),
) -> SyncResult:
    """Sync `entries` into pdf_dir (default: the context's paths.pdf_dir). ctx.reuse (frozen inputs): nothing is fetched,
    a PDF whose file is missing counts as missing, the registry is not written and the safeguard runs dry. ctx.limit: only
    the first N entries, and no safeguard. A corrupt registry raises CorruptStateError; a corrupt list state is reported in `check` (corrupt) with
    every missing PDF kept. An URL listed twice is synced once (the first entry wins). An empty list raises ValueError: it
    is a failed crawl, not a list, unless `allow_empty`: then the list safeguard treats it as a list that shrank to nothing, so
    every registered PDF is kept for its grace runs. A registered link in `dropped` (left out on purpose) is removed with its file
    at once, and its absence is no shrinking of the list."""
    entries = list({entry.url: entry for entry in reversed(entries)}.values())[::-1]
    if not entries and not allow_empty:
        raise ValueError("no PDF entries to sync")
    pdf_dir = pdf_dir or ctx.paths["pdf_dir"]
    registry = reg.load(ctx.state)
    selected = entries[: ctx.limit] if ctx.limit is not None else entries
    Path(pdf_dir).mkdir(parents=True, exist_ok=True)

    results = [_sync_one(entry, registry, ctx, pdf_dir) for entry in selected]

    retained: list[PdfEntry] = []
    removed, check = 0, {}
    if ctx.limit is None:
        registered = set(registry)
        safeguard = Safeguard.from_settings(ctx.settings)
        try:
            retained, check, removed = safeguard.apply(
                entries,
                registry,
                ctx.state.path(LIST_STATE),
                ctx.cache_dir,
                pdf_dir,
                dry_run=ctx.reuse,
                dropped=dropped,
            )
        except Exception as exc:
            # never let the safeguard itself delete or drop documents
            logging.error(f"[PdfSync] list safeguard failed ({exc}); every missing PDF is kept this run")
            retained = safeguard.keep_missing(registry, entries, ctx.cache_dir, pdf_dir)
            check = {"error": str(exc), "corrupt": isinstance(exc, CorruptStateError)}
            removed = len(registered - set(registry))
    if not ctx.reuse:
        reg.save(ctx.state, registry)
    return SyncResult(selected + retained, retained, results, removed, check)


def _sync_one(entry: PdfEntry, registry: dict[str, str], ctx: SourceContext, pdf_dir: str | Path) -> dict:
    path = local_path(entry.url, pdf_dir)
    result = {"url": entry.url, "file": path.name}
    if ctx.reuse:
        return {**result, "status": "cached" if path.exists() else "missing"}
    exists = path.exists()
    if entry.sha256 and registry.get(entry.url) == entry.sha256 and exists:
        return {**result, "status": "cached"}
    before = file_hash(path) if exists else None
    try:
        after = _download(entry.url, path, ctx.fetch)
    except Exception as exc:
        logging.error(f"Failed to sync PDF {entry.url}: {exc}")
        return {**result, "status": "retained_on_error" if exists else "failed", "error": str(exc)}
    if before and before != after:
        try:
            reg.drop_cache_entry(before, ctx.cache_dir)
        except OSError as exc:  # an orphaned cache entry only costs disk space; the registry must still be saved
            logging.error(f"Failed to drop the extraction cache entry of {entry.url}: {exc}")
    registry[entry.url] = entry.sha256 or after
    return {**result, "status": "downloaded"}


def _download(url: str, destination: Path, fetch: Fetcher) -> str:
    """Stream the URL into destination (through a temporary file) and return the file's sha256."""
    response = fetch.get(url, timeout=PDF_DOWNLOAD_TIMEOUT)  # a host the overlay does not allow raises
    response.raise_for_status()
    temp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with open(temp, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        digest = file_hash(temp)
        temp.replace(destination)
        return digest
    finally:
        temp.unlink(missing_ok=True)
