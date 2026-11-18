"""Discovery of the pages of one source: XML sitemaps, link-following and English twins.

A `Source` is one host with the path prefixes it may be crawled under, the prefixes left out, its sitemap URLs, its seed
URLs and how many link levels to follow from the seeds (0 for a host that is listed seeds only). The URLs come from
- the sitemaps (`urlset`, and `sitemapindex` followed to its sitemaps), kept when they are on the source's host, under a
  prefix and under no exclusion,
- the seeds themselves, always, and the links of a fetched page followed from them level by level, with the same rules; a
  URL the fetcher then skips (robots.txt, `<meta name="robots">`, an error) is dropped; PDFs are listed, never downloaded here.
Every URL is normalised once, and the normalised URL is what is checked and what is requested: the fragment is dropped, the
query kept, unreserved characters are percent-decoded, dot segments are resolved and `//` collapsed, so that `..` or `%2e%2e`
cannot lead out of a prefix. A URL is in scope when its scheme, host and port are the source's and its path is under a prefix
and no exclusion; a fetched page that a redirect ended outside that scope (and is not a seed) is dropped and its links not
followed. Which hosts may be fetched and what robots.txt allows stays
with the `Fetcher`, which is for one thread: discovery calls it from one.

With `twins`, each `.de.jsp` URL is also asked for as `.en.jsp` (as is any `.en.jsp` URL found); the twin is listed when it
is an English page, not when it only carries the notice that the content is available in German or when its text is German.

XML is parsed without entity resolution, network access or DTD loading."""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import SplitResult, urldefrag, urljoin, urlsplit, urlunsplit

import lxml.etree as etree
from lxml import html
from rag_core.domain.language import detect

from indexer.crawl.fetch import Fetcher, Response, Skipped

__all__ = ["Source", "discover", "in_scope", "normalise", "stays_in_scope"]

GERMAN_ONLY_NOTICE = "The content of this page is only available in German"
MAX_SITEMAP_NESTING = 3
XML = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, dtd_validation=False)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    """What to discover on one host. Prefixes are URL path prefixes (`/studium/`); with none, only the seeds are listed.
    A URL is the source's only on its scheme, host and port (the scheme's default port when `port` is None)."""

    host: str
    prefixes: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    sitemaps: tuple[str, ...] = ()
    seeds: tuple[str, ...] = ()
    depth: int = 0
    twins: bool = False
    scheme: str = "https"
    port: int | None = None


DEFAULT_PORTS = {"http": 80, "https": 443}
UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")


def _unescape(match: re.Match[str]) -> str:
    char = chr(int(match.group(1), 16))
    return char if char in UNRESERVED else match.group(0).upper()


def _clean_path(path: str) -> str:
    """The path with unreserved characters decoded, dot segments resolved and empty segments dropped (RFC 3986 5.2.4)."""
    segments = ESCAPE.sub(_unescape, path).split("/")
    kept: list[str] = []
    for segment in segments[1:]:
        if segment == "..":
            if kept:
                kept.pop()
        elif segment not in ("", "."):
            kept.append(segment)
    return "/" + "/".join(kept) + ("/" if kept and segments[-1] in ("", ".", "..") else "")


def normalise(url: str) -> str:
    """The URL without its fragment and surrounding blanks, scheme and host in lower case, the path cleaned (see `_clean_path`);
    the query stays. '' if unusable."""
    try:
        parts = urlsplit(urldefrag(url.strip())[0])
        return urlunsplit(parts._replace(netloc=parts.netloc.lower(), path=_clean_path(parts.path)))
    except ValueError:
        return ""


def _path(url: str) -> str:
    return urlsplit(url).path


def _origin(source: Source, parts: SplitResult) -> bool:
    try:
        port = parts.port or DEFAULT_PORTS.get(parts.scheme)
    except ValueError:
        return False
    return (
        parts.scheme == source.scheme
        and (parts.hostname or "").removesuffix(".") == source.host.lower()
        and port == (source.port or DEFAULT_PORTS.get(source.scheme))
    )


def _listed(source: Source, url: str) -> bool:
    """Of the source's scheme, host and port, under a prefix and under no exclusion (`url` is normalised)."""
    parts = urlsplit(url)
    path = parts.path
    return (
        _origin(source, parts)
        and any(path.startswith(p) for p in source.prefixes)
        and not any(path.startswith(p) for p in source.exclude)
    )


def in_scope(source: Source, url: str) -> bool:
    """Whether `url`, in any spelling (it is normalised first), is of the source's origin, under a prefix and under no exclusion."""
    return _listed(source, normalise(url))


def _sitemap_urls(fetcher: Fetcher, url: str, seen: set[str], nesting: int = 0) -> Iterator[str]:
    if url in seen or nesting > MAX_SITEMAP_NESTING:
        return
    seen.add(url)
    reply = fetcher.get(url)
    if isinstance(reply, Skipped):
        log.info("sitemap %s not read: %s", url, reply.reason)
        return
    try:
        root = etree.fromstring(reply.body, XML)
    except etree.XMLSyntaxError:
        log.info("sitemap %s is not XML", url)
        return
    locs = [(el.text or "").strip() for el in root.iter("{*}loc")]
    if etree.QName(root).localname == "sitemapindex":
        for loc in locs:
            yield from _sitemap_urls(fetcher, normalise(loc), seen, nesting + 1)
    else:
        yield from locs


def _links(reply: Response) -> Iterator[str]:
    if "html" not in reply.content_type.lower():
        return
    try:
        tree = html.fromstring(reply.body)
    except (ValueError, etree.ParserError):
        return
    for href in tree.xpath("//a/@href"):
        try:
            yield urljoin(reply.final_url, str(href))
        except ValueError:
            continue


def stays_in_scope(source: Source, reply: Response, url: str) -> bool:
    """False when the request for `url` ended, after redirects, outside the source's scope (a seed is in scope)."""
    final = normalise(reply.final_url)
    return final == url or _listed(source, final) or final in map(normalise, source.seeds)


def _follow(source: Source, fetcher: Fetcher, found: dict[str, None]) -> None:
    """Add the in-scope links of the pages in `found`, `source.depth` levels deep; drop a page the fetcher skips or whose
    redirect ends outside the scope."""
    seen = set(found)
    level = list(found)
    for _ in range(source.depth):
        following: list[str] = []
        for url in level:
            if _path(url).lower().endswith(".pdf"):
                continue
            reply = fetcher.get(url)
            if isinstance(reply, Skipped) or not stays_in_scope(source, reply, url):
                found.pop(url, None)
                continue
            for link in map(normalise, _links(reply)):
                if link not in seen and _listed(source, link):
                    seen.add(link)
                    found[link] = None
                    following.append(link)
        level = following


def _english_page(source: Source, fetcher: Fetcher, url: str) -> bool:
    reply = fetcher.get(url)
    if isinstance(reply, Skipped) or not stays_in_scope(source, reply, url) or "html" not in reply.content_type.lower():
        return False
    try:
        tree = html.fromstring(reply.body)
    except (ValueError, etree.ParserError):
        return False
    etree.strip_elements(tree, "script", "style", with_tail=False)
    text = " ".join(tree.text_content().split())
    return GERMAN_ONLY_NOTICE not in text and detect(text) == "en"


def _twin(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=parts.path.removesuffix(".de.jsp") + ".en.jsp"))


def discover(source: Source, fetcher: Fetcher) -> list[str]:
    """The URLs of the source, in discovery order and without repeats: seeds, then sitemap entries, then followed links."""
    found: dict[str, None] = {}
    for seed in map(normalise, source.seeds):
        if seed and _origin(source, urlsplit(seed)):
            found[seed] = None
    seen: set[str] = set()
    for sitemap in map(normalise, source.sitemaps):
        for url in map(normalise, _sitemap_urls(fetcher, sitemap, seen)):
            if _listed(source, url):
                found[url] = None
    _follow(source, fetcher, found)
    if not source.twins:
        return list(found)
    english = [u for u in found if _path(u).endswith(".en.jsp")]
    candidates = dict.fromkeys([_twin(u) for u in found if _path(u).endswith(".de.jsp")] + english)
    return [u for u in found if u not in english] + [u for u in candidates if _english_page(source, fetcher, u)]
