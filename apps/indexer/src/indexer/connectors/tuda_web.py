"""The TU web connector: the pages and PDFs of the hosts in sources.yaml become documents with their metadata.

A build crawls the hosts of its source set (`options["sources"]`: the core hosts, or with "full" the full hosts and the
full blocks too). Discovery lists each host's pages; every page is fetched, scope-checked after redirects and extracted
into documents (a hub page gives one document per section that carries its own PO version). The PDFs the pages link are
taken when their own host lists them (its prefixes), whichever host the page is on; the PDF metadata comes from the page
it was linked from. All documents of all hosts then go through the fail-closed selection once: a programme regulation
without programme or PO version is not indexed and a hub's copy of a university-wide regulation yields to the original.
The PDFs that stay are handed to the PDF sync as the files the fetcher already holds (nothing is fetched twice, and a
download that broke off is neither registered nor kept), which keeps a PDF missing from one crawl for a few builds.

One fetcher serves the whole crawl, on one thread: it spaces the requests to a host, reads robots.txt and caches every
response under the crawl folder. A URL cached since the crawl began is not asked for again, so a page fetched for its links
is not fetched a second time; `--resume` continues the interrupted crawl of the same source set instead of beginning anew.

The report's `manifest` lists, per kept document this crawl fetched, its URL, content hash, fetch date and metadata, for the
snapshot manifest the build writes after it publishes. The report also lists what the build summary names: `empty_pages` (a main content selector that finds nothing: the site may have
changed), `not_indexed` (the fail-closed documents), `copies` (hub copies skipped), `skipped` (a URL that was not fetched,
and why) and `hosts_down` (a host of the source set whose robots.txt could not be read: nothing was fetched from it, and the
build refuses to swap). `fetcher_options` are Fetcher's own (a test injects a client and a clock)."""

import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from rag_core.domain.sources import CorruptStateError, PageItem, PdfItem, SourceResult
from rag_core.ports.sources import SourceContext
from rag_core.settings.profile import Roles

from indexer.connectors.crawl_setup import Host, RecordingFetcher, active_hosts, load_roles, load_sources, make_fetcher
from indexer.crawl.discover import discover, in_scope, stays_in_scope
from indexer.crawl.extract import Extraction, Link, extract
from indexer.crawl.fetch import Response, Skipped
from indexer.crawl.manifest import ManifestDoc
from indexer.crawl.metadata import (
    PROGRAMME_BOUND,
    Candidate,
    Metadata,
    PageEntry,
    page_entries,
    pdf_metadata,
    select_documents,
    twin_keys,
)
from indexer.sync.pdfs import PdfEntry, SyncResult, sync_pdfs
from indexer.sync.registry import REGISTRY, file_hash, local_path
from indexer.sync.safeguard import LIST_STATE

CRAWL_RUN = "crawl_run.json"  # in the state store: when the last crawl began and whether it finished
DOC_TYPES = {
    "po": "Prüfungsordnung",
    "mhb": "Modulhandbuch",
    "spp": "Studien- und Prüfungsplan",
    "apb": "Allgemeine Prüfungsbestimmungen",
    "eso": "Einschreibeordnung",
    "promotion": "Promotion",
    "residence": "Aufenthalt",
    "info": "Information",
    "form": "Formular",
}


def derived(metadata: Metadata) -> dict[str, Any]:
    """The document metadata in JSON form (lists, ISO text) without its empty values: NULL is an absent key, never an empty
    list."""
    values = {
        "regulation": metadata.regulation,
        "degree": list(metadata.degree) if metadata.degree else None,
        "programme": list(metadata.programme) if metadata.programme else None,
        "po_version": list(metadata.po_version) if metadata.po_version else None,
        "archived": metadata.archived,
        "valid_from": metadata.valid_from.isoformat() if metadata.valid_from else None,
        "lang": metadata.lang,
        "twin_key": metadata.twin_key,
        "official_url": metadata.official_url,
    }
    return {k: v for k, v in values.items() if v is not None}


def lacks(meta: Mapping[str, Any]) -> tuple[str, ...]:
    """The required fields a PDF's stored metadata lacks: its regulation, and for a PO, module handbook or study plan its
    programme and PO version. A PDF kept from an earlier build may have lost its metadata with the list state."""
    regulation = meta.get("regulation")
    if not regulation:
        return ("regulation",)
    return tuple(f for f in ("programme", "po_version") if regulation in PROGRAMME_BOUND and not meta.get(f))


class PdfFiles:
    """The fetch port the PDF sync downloads through, over the complete files the crawl fetcher already holds: a download is
    a copy of the cached file."""

    def __init__(self, files: Mapping[str, Path]) -> None:
        self.files = files

    def endpoint(self, *, timeout: Any) -> Any:
        raise RuntimeError("the PDF sync has no endpoint of its own: it copies the files the crawl fetched")

    def get(self, url: str, *, timeout: Any) -> "CachedFile":
        return CachedFile(self.files[url])


class CachedFile:
    def __init__(self, path: Path) -> None:
        self.path = path

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int = 8192) -> Iterator[bytes]:
        with open(self.path, "rb") as f:
            yield from iter(lambda: f.read(chunk_size), b"")


@dataclass(frozen=True)
class Doc:
    """A document of a page: its URL (the page's, plus the section's anchor), title, extracted entry and the page's response
    (for the snapshot manifest; the body is dropped, the cache holds it)."""

    url: str
    title: str
    entry: PageEntry
    reply: Response


@dataclass(frozen=True)
class Pdf:
    """A PDF a page links, fetched: the link, the page it is on, the response (the body is dropped: `path` is the complete file)
    and the metadata derived for it."""

    link: Link
    hub: Extraction
    reply: Response
    metadata: Metadata


@dataclass
class Crawl:
    """What one crawl found, and what it could not use."""

    docs: list[Doc]
    pdfs: dict[str, Pdf]  # by URL, in page order
    pages: int = 0
    empty_pages: list[dict[str, str]] = dataclasses.field(default_factory=list)
    skipped: list[dict[str, str]] = dataclasses.field(default_factory=list)


class TudaWeb:
    name = "tuda_web"
    state_files = (REGISTRY, LIST_STATE)

    def __init__(self, **fetcher_options: Any) -> None:
        self.fetcher_options = fetcher_options

    def collect(self, ctx: SourceContext) -> SourceResult:
        sources, roles = load_sources(ctx), load_roles(ctx)
        hosts = active_hosts(sources, ctx.options["sources"] == "full")
        began = self._began(ctx, self.fetcher_options.get("today", date.today)())
        fetcher = make_fetcher(ctx, hosts, since=began, **self.fetcher_options)
        crawl = Crawl([], {})
        for host in hosts.values():
            self._crawl_host(fetcher, host, hosts, roles, twin_keys(sources.pdf_twins), ctx.limit, crawl)
        crawl.skipped += [{"url": url, "reason": skip.reason} for url, skip in fetcher.skipped.items()]
        candidates = [Candidate(d.url, None, d.entry.metadata) for d in crawl.docs]
        candidates += [Candidate(url, p.reply.content_hash, p.metadata) for url, p in crawl.pdfs.items()]
        selection = select_documents(candidates, roles)
        kept = {c.url for c in selection.kept}
        items: list[PageItem | PdfItem] = [self._page_item(d) for d in crawl.docs if d.url in kept]
        merged = self._union_scopes({url: p for url, p in crawl.pdfs.items() if url in kept})
        synced = self._sync(ctx, [self._pdf_entry(p) for p in merged.values()], crawl.pdfs)
        pdf_dir = ctx.paths["pdf_dir"]
        not_indexed = [{"url": url, "missing": list(missing)} for url, missing in selection.not_indexed]
        for e in synced.entries:
            if missing := lacks(e.meta):
                not_indexed.append({"url": e.url, "missing": list(missing)})
            elif (path := local_path(e.url, pdf_dir)).is_file():
                items.append(PdfItem(str(path), e.url, file_hash(path), e.meta))
        if not ctx.reuse:
            ctx.state.save(
                CRAWL_RUN, {"started": began.isoformat(), "sources": ctx.options["sources"], "complete": True}
            )
        report: dict[str, Any] = {
            "sources": ctx.options["sources"],
            "pages": crawl.pages,
            "pdfs_linked": len(crawl.pdfs),
            "empty_pages": crawl.empty_pages,
            "not_indexed": not_indexed,
            "copies": list(selection.copies),
            "skipped": crawl.skipped,
            "hosts_down": [{"host": h, "reason": why} for h, why in fetcher.hosts_down().items()],
            "manifest": [asdict(m) for m in self._manifest(crawl, kept)],
        }
        report["pdf_sync"] = {
            "downloaded": synced.downloaded,
            "unchanged": synced.unchanged,
            "failed": synced.failed,
            "missing": synced.missing,
            "removed": synced.removed,
            "retained": [e.url for e in synced.retained],
            "list_check": synced.check,
        }
        return SourceResult(items, report, expected_minimum=int(ctx.settings.get("min_documents", 0)))

    @staticmethod
    def _began(ctx: SourceContext, today: date) -> date:
        """The day this crawl counts from: a URL cached since then is not asked for again. A resumed crawl keeps the day of
        the unfinished crawl of the same source set; any other crawl begins today."""
        sources = ctx.options["sources"]
        if ctx.reuse:
            return today
        run = ctx.state.load(CRAWL_RUN)
        if ctx.options["resume"] and run and not run.get("complete") and run.get("sources") == sources:
            return date.fromisoformat(run["started"])
        ctx.state.save(CRAWL_RUN, {"started": today.isoformat(), "sources": sources, "complete": False})
        return today

    def _crawl_host(
        self,
        fetcher: RecordingFetcher,
        host: Host,
        hosts: Mapping[str, Host],
        roles: Roles,
        twins: Mapping[str, str],
        limit: int | None,
        crawl: Crawl,
    ) -> None:
        """Fetch and extract the pages discovery lists for `host` and the PDFs they link."""
        for url in discover(host.source, fetcher):
            if limit is not None and crawl.pages >= limit:
                return
            if url.lower().endswith(".pdf"):
                continue  # a PDF is taken from the page that links it, which knows what it is
            reply = fetcher.get(url)
            if isinstance(reply, Skipped):
                continue  # the fetcher remembers it
            if not stays_in_scope(host.source, reply, url):
                crawl.skipped.append({"url": url, "reason": "redirected out of scope"})
                continue
            if "html" not in reply.content_type.lower():
                continue
            page = extract(reply.body, reply.final_url, host.selectors, reply.content_type)
            crawl.pages += 1
            if page.problem:
                crawl.empty_pages.append({"url": page.url, "problem": page.problem})
                continue
            for entry in page_entries(page, roles):
                anchor = f"{page.url}#{entry.section_id}" if entry.section_id else page.url
                title = " – ".join(dict.fromkeys(t for t in (page.title, entry.heading) if t)) or page.url
                crawl.docs.append(Doc(anchor, title, entry, dataclasses.replace(reply, body=b"")))
            for link in page.pdfs:
                owner = hosts.get(urlsplit(link.url).hostname or "")
                if link.url in crawl.pdfs or owner is None or not in_scope(owner.source, link.url):
                    continue  # linked from another section, or not the listing host's: a bundle of acts, say
                pdf = fetcher.get(link.url)
                if not isinstance(pdf, Skipped) and not stays_in_scope(owner.source, pdf, link.url):
                    crawl.skipped.append({"url": link.url, "reason": "redirected out of scope"})
                elif not isinstance(pdf, Skipped):
                    crawl.pdfs[link.url] = Pdf(
                        link, page, dataclasses.replace(pdf, body=b""), pdf_metadata(link, page, roles, twins)
                    )

    @staticmethod
    def _union_scopes(pdfs: Mapping[str, Pdf]) -> dict[str, Pdf]:
        """One file linked from several hubs (the same hash, the same regulation) is one document, which the first link
        stands for: its programmes, degrees and PO versions are those of every link. A link that states none stays NULL,
        which means every programme."""
        groups: dict[tuple[str, str], list[str]] = {}
        for url, pdf in pdfs.items():
            groups.setdefault((pdf.reply.content_hash, pdf.metadata.regulation), []).append(url)
        merged = dict(pdfs)
        for urls in groups.values():
            if len(urls) > 1:
                first = pdfs[urls[0]]
                scope = {}
                for field in ("degree", "programme", "po_version"):
                    values = [getattr(pdfs[u].metadata, field) for u in urls]
                    scope[field] = None if None in values else tuple(dict.fromkeys(v for each in values for v in each))
                merged[urls[0]] = dataclasses.replace(first, metadata=dataclasses.replace(first.metadata, **scope))
        return merged

    @staticmethod
    def _manifest(crawl: Crawl, kept: set[str]) -> list[ManifestDoc]:
        """The snapshot manifest's documents: every kept document this crawl fetched, with its content hash and fetch date."""
        docs = [
            ManifestDoc(d.url, d.reply.content_hash, d.reply.fetched, derived(d.entry.metadata)) for d in crawl.docs
        ]
        docs += [
            ManifestDoc(url, p.reply.content_hash, p.reply.fetched, derived(p.metadata))
            for url, p in crawl.pdfs.items()
        ]
        return [d for d in docs if d.url in kept]

    @staticmethod
    def _page_item(doc: Doc) -> PageItem:
        meta = doc.entry.metadata
        return PageItem(
            doc.url,
            doc.entry.markdown,
            {
                "title": doc.title,
                "url": doc.url,
                "source_type": "page",
                "doc_type": DOC_TYPES[meta.regulation],
                "origin": "website",
                "section": doc.entry.section_id or "",
                "doc_year": meta.valid_from.year if meta.valid_from else None,
                "doc_date": meta.valid_from.isoformat() if meta.valid_from else None,
                "page_number": -1,  # not a PDF page
                "doc_key": doc.url,
                **derived(meta),
            },
        )

    @staticmethod
    def _pdf_entry(pdf: Pdf) -> PdfEntry:
        link = pdf.link
        title = link.text or link.title or unquote(urlsplit(link.url).path.rsplit("/", 1)[-1])
        meta = {
            "url": pdf.hub.url,
            "download_url": link.url,
            "download_title": title,
            "doc_type": DOC_TYPES[pdf.metadata.regulation],
            "origin": "website",
            "section": link.section_id or "",
            **derived(pdf.metadata),
        }
        return PdfEntry(link.url, title, pdf.reply.content_hash, meta)

    @staticmethod
    def _sync(ctx: SourceContext, entries: list[PdfEntry], pdfs: Mapping[str, Pdf]) -> SyncResult:
        """The kept PDFs through the PDF sync, which copies the complete files the fetcher holds into the PDF folder, registers
        them by hash and keeps a PDF that this crawl missed for a few builds."""
        files = {url: pdf.reply.path for url, pdf in pdfs.items() if pdf.reply.path}
        with_files = dataclasses.replace(ctx, fetch=PdfFiles(files))  # pyright: ignore[reportArgumentType]
        result = sync_pdfs(entries, with_files, pdf_dir=ctx.paths["pdf_dir"], allow_empty=True)
        if result.check.get("corrupt"):
            raise CorruptStateError(f"the PDF list state cannot be used: {result.check['error']}")
        return result
