"""What the two web connectors share: the profile's crawl files, the hosts of the build's source set and the polite fetcher.

The files sit beside the profile (`options["profile_dir"]` of the context). The core set is every host marked core; the
full set adds the full hosts and, on a core host with a `full` block, swaps that block's sitemap, prefixes, exclusions and
depth in for the host's own. The fetcher is made for the context: its user agent carries the user's crawl contact when
there is one, it caches under the crawl folder, frozen inputs make it answer from the cache alone, and the overlay's routes
(the harness's) send the requests of a host to another origin."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from rag_core.ports.sources import SourceContext
from rag_core.settings import load_profile
from rag_core.settings.profile import CrawlHost, Roles, Sources

from indexer.crawl.cache import Cache
from indexer.crawl.discover import Source
from indexer.crawl.extract import Selectors
from indexer.crawl.fetch import Fetcher, Response, Skipped
from indexer.crawl.routes import Routed


@dataclass(frozen=True)
class Host:
    """A host of the active set: its entry, what discovery needs and where its pages keep their content."""

    entry: CrawlHost
    source: Source
    selectors: Selectors


class RecordingFetcher(Fetcher):
    """A fetcher that remembers every URL it skipped and why, whoever asked (discovery drops such a URL without a word)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skipped: dict[str, Skipped] = {}

    def get(self, url: str) -> Response | Skipped:
        reply = super().get(url)
        if isinstance(reply, Skipped):
            self.skipped[reply.url] = reply
        return reply


def load_sources(ctx: SourceContext) -> Sources:
    return load_profile(Sources, Path(ctx.options["profile_dir"]) / "sources.yaml")


def load_roles(ctx: SourceContext) -> Roles:
    return load_profile(Roles, Path(ctx.options["profile_dir"]) / "roles.yaml")


def active_hosts(sources: Sources, full: bool) -> dict[str, Host]:
    """The hosts of the core set, or of the full set, by host name."""
    hosts = {}
    for name, entry in sources.hosts.items():
        if entry.set == "full" and not full:
            continue
        block = entry.full if full and entry.full else entry
        selectors = {k: v for k, v in entry.selectors.model_dump().items() if v is not None}
        hosts[name] = Host(
            entry,
            Source(
                name,
                prefixes=tuple(block.prefixes),
                exclude=tuple(block.exclude),
                sitemaps=tuple(block.sitemap),
                seeds=tuple(entry.seeds),
                depth=block.depth,
                twins=entry.twins,
            ),
            Selectors(**selectors),
        )
    return hosts


def make_fetcher(
    ctx: SourceContext, hosts: Iterable[str], *, since: date | None = None, **options: Any
) -> RecordingFetcher:
    """A fetcher for `hosts` (nothing else is fetched) that caches under the crawl folder; `options` are Fetcher's own. The
    overlay's routes, when there are any, decide where the requests go."""
    if routes := ctx.options.get("routes"):
        options.setdefault("client", httpx.Client(transport=Routed(routes), timeout=30.0))
    return RecordingFetcher(
        hosts,
        contact=ctx.options.get("contact"),
        cache=Cache(ctx.paths["crawl_dir"]),
        since=since,
        offline=ctx.reuse,
        **options,
    )
