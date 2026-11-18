"""The polite fetcher: one GET per URL through the host policy, robots.txt, the rate limit and the page's own robots meta,
with a cache of what it fetched.

A URL is fetched only when its host is on the caller's allow-list and not one of the hosts that are never crawled, when the
host's robots.txt allows it for this user agent, and (for a page) when its `<meta name="robots">` carries none of `noindex`,
`noai` or `none`. Requests to one host are at least `interval` seconds apart; the clock and the sleep are injected. A
server error or a dropped connection is retried `retries` times, each retry through the rate limit. Redirects are followed
by hand, and every hop passes the same checks as the first URL.

robots.txt is read once per host (RFC 9309): a 4xx answer allows everything; a server error, no answer, a 429 or a 200 that
is an HTML page (a soft 404 that is not a robots file) disallows everything.

With a cache, a URL whose entry has a validator is asked conditionally (`If-None-Match`, `If-Modified-Since`; a 304 keeps the
cached body); without one the new body's hash is compared with the cached hash. `since` resumes an interrupted crawl: a URL
cached on or after that date is answered from the cache without a request (robots.txt still applies). A PDF is cached only
as a complete file that starts with `%PDF-`; a download that breaks off leaves the cache as it was."""

import hashlib
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from lxml import html

from indexer.crawl.cache import Cache, Entry
from indexer.crawl.robots import Robots

__all__ = ["NEVER_FETCHED", "Fetcher", "Response", "Skipped", "user_agent"]

# whatever the allow-list says: the city's main site blocks crawlers by name, and h-da.de is not part of this project
NEVER_FETCHED = ("www.darmstadt.de", "h-da.de")
BLOCKING_META = frozenset({"noindex", "noai", "none"})
REDIRECTS = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 5
MAX_BYTES = 100 * 1024 * 1024
log = logging.getLogger(__name__)


def user_agent(contact: str | None = None, release: str | None = None) -> str:
    """`rag-tuda/<version> (+<contact>)`; without a contact, `rag-tuda/<version> (student research prototype)`."""
    release = release or version("indexer")
    return f"rag-tuda/{release} ({'+' + contact if contact else 'student research prototype'})"


@dataclass(frozen=True)
class Response:
    """A fetched document: the URL asked for, the URL it ended at after redirects, and the body as sent. `fetched` is the date
    of the request that confirmed it; `changed` is False when the body equals the cached one, `from_cache` when no body was
    sent (a 304, or a resumed crawl); `path` is the cached file when there is a cache."""

    url: str
    final_url: str
    status: int
    body: bytes
    content_type: str
    content_hash: str
    fetched: str
    etag: str | None = None
    last_modified: str | None = None
    path: Path | None = None
    changed: bool = True
    from_cache: bool = False


@dataclass(frozen=True)
class _Reply:
    status: int
    headers: httpx.Headers
    body: bytes


@dataclass(frozen=True)
class Skipped:
    """A URL that was not fetched or not kept, and why (`reason` is one of a few fixed phrases; `detail` adds to it)."""

    url: str
    reason: str
    detail: str = ""


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _never_fetched(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in NEVER_FETCHED)


def blocking_meta(body: bytes) -> str | None:
    """The first blocking token of the page's `<meta name="robots">` tags, or None."""
    try:
        tree = html.fromstring(body)
    except (ValueError, html.etree.ParserError):  # empty or not a document
        return None
    for meta in tree.iter("meta"):
        if (meta.get("name") or "").strip().lower() in ("robots", "rag-tuda"):
            tokens = {t.strip().lower() for t in (meta.get("content") or "").split(",")}
            if found := sorted(tokens & BLOCKING_META):
                return found[0]
    return None


class Fetcher:
    def __init__(
        self,
        allowed_hosts: Iterable[str],
        *,
        contact: str | None = None,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        interval: float = 1.0,
        retries: int = 2,
        timeout: float = 30.0,
        cache: Cache | None = None,
        since: date | None = None,
        today: Callable[[], date] = date.today,
    ):
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)
        self.user_agent = user_agent(contact)
        self._client = client or httpx.Client(timeout=timeout)
        self._clock, self._sleep, self._interval, self._retries = clock, sleep, interval, retries
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, Robots] = {}
        self._cache, self._since, self._today = cache, since, today

    def get(self, url: str) -> Response | Skipped:
        """Fetch `url`, or say why not."""
        asked = url
        cached = self._cache.entry(asked) if self._cache else None
        if cached and self._since and cached.fetched >= self._since.isoformat():
            return self._refuse(asked, asked) or self._cached(cached, changed=False)
        headers = {}
        if cached and cached.etag:
            headers["If-None-Match"] = cached.etag
        if cached and cached.last_modified:
            headers["If-Modified-Since"] = cached.last_modified
        for _ in range(MAX_REDIRECTS + 1):
            if skipped := self._refuse(url, asked):
                return skipped
            reply = self._send(url, headers)
            if isinstance(reply, str):
                return Skipped(asked, reply)
            if reply.status in REDIRECTS and reply.headers.get("location"):
                url = urljoin(url, reply.headers["location"])
                continue
            if reply.status == 304 and cached and self._cache:
                entry = self._cache.touch(
                    cached, self._today().isoformat(), reply.headers.get("etag"), reply.headers.get("last-modified")
                )
                return self._cached(entry, changed=False)
            if reply.status >= 300:
                return Skipped(asked, f"http {reply.status}")
            return self._keep(asked, url, reply, cached)
        return Skipped(asked, "too many redirects")

    def _keep(self, asked: str, url: str, reply: "_Reply", cached: Entry | None) -> Response | Skipped:
        content_type = reply.headers.get("content-type", "")
        if "pdf" in content_type.lower() or urlsplit(url).path.lower().endswith(".pdf"):
            if not reply.body.startswith(b"%PDF-"):
                return Skipped(asked, "not a pdf", content_type)
        elif "html" in content_type.lower() and (token := blocking_meta(reply.body)):
            if self._cache:
                self._cache.drop(asked)
            return Skipped(asked, "meta robots", token)
        digest = hashlib.sha256(reply.body).hexdigest()
        etag, modified = reply.headers.get("etag"), reply.headers.get("last-modified")
        fetched = self._today().isoformat()
        entry, changed = None, True
        if self._cache and cached and cached.content_hash == digest:
            entry, changed = self._cache.touch(cached, fetched, etag, modified), False
        elif self._cache:
            entry = self._cache.store(asked, url, reply.body, content_type, etag, modified, fetched)
        path = self._cache.file(entry) if self._cache and entry else None
        return Response(
            asked, url, reply.status, reply.body, content_type, digest, fetched, etag, modified, path, changed
        )

    def _cached(self, entry: Entry, *, changed: bool) -> Response:
        assert self._cache
        return Response(
            entry.url,
            entry.final_url,
            200,
            self._cache.body(entry),
            entry.content_type,
            entry.content_hash,
            entry.fetched,
            entry.etag,
            entry.last_modified,
            self._cache.file(entry),
            changed,
            from_cache=True,
        )

    def robots(self, url: str) -> Robots:
        """The rules of the host of `url`, read on first use."""
        host = _host(url)
        if host not in self._robots:
            parts = urlsplit(url)
            self._robots[host] = self._read_robots(f"{parts.scheme}://{parts.netloc}/robots.txt")
        return self._robots[host]

    def _host_refusal(self, host: str) -> str | None:
        if _never_fetched(host):
            return "host never fetched"
        return None if host in self.allowed_hosts else "host not allowed"

    def _refuse(self, url: str, asked: str) -> Skipped | None:
        host = _host(url)
        if urlsplit(url).scheme not in ("http", "https") or not host:
            return Skipped(asked, "not an http url", url)
        if reason := self._host_refusal(host):
            return Skipped(asked, reason, host)
        if not self.robots(url).allowed(self.user_agent, url):
            return Skipped(asked, "robots", url)
        return None

    def _read_robots(self, url: str) -> Robots:
        reply = self._send(url)
        for _ in range(MAX_REDIRECTS):  # RFC 9309: follow redirects, here only to hosts that may be fetched
            if not (isinstance(reply, _Reply) and reply.status in REDIRECTS and reply.headers.get("location")):
                break
            url = urljoin(url, reply.headers["location"])
            reply = "refused" if self._host_refusal(_host(url)) else self._send(url)
        if isinstance(reply, str) or reply.status >= 500 or reply.status == 429 or 300 <= reply.status < 400:
            log.warning("robots.txt of %s is unreadable: nothing is fetched from that host", url)
            return Robots.disallow_all()
        if reply.status >= 400:
            return Robots.parse("")
        if "html" in reply.headers.get("content-type", "").lower():  # a soft 404 is allow-all only on a 4xx status
            return Robots.disallow_all()
        return Robots.parse(reply.body.decode("utf-8", "replace"))

    def _wait(self, host: str) -> None:
        now = self._clock()
        last = self._last_request.get(host)
        if last is not None and now - last < self._interval:
            self._sleep(self._interval - (now - last))
            now = self._clock()
        self._last_request[host] = now

    def _send(self, url: str, headers: dict[str, str] | None = None) -> _Reply | str:
        """One GET, retried on a server error, a 429 or a transport error: the last reply, or why there is none."""
        last: _Reply | None = None
        for _ in range(self._retries + 1):
            self._wait(_host(url))
            try:
                last = self._once(url, headers or {})
            except httpx.HTTPError as exc:
                log.info("GET %s failed: %s", url, exc)
                continue
            except OverflowError:
                return "too large"
            if last.status < 500 and last.status != 429:
                break
        return last or "unreachable"

    def _once(self, url: str, headers: dict[str, str]) -> _Reply:
        with self._client.stream("GET", url, headers={"User-Agent": self.user_agent, **headers}) as reply:
            body = bytearray()
            for chunk in reply.iter_bytes():
                body += chunk
                if len(body) > MAX_BYTES:
                    raise OverflowError(url)
            expected = reply.headers.get("content-length")
            if expected and "content-encoding" not in reply.headers and int(expected) != len(body):
                raise httpx.ReadError("the body is shorter than its content-length")
            return _Reply(reply.status_code, reply.headers, bytes(body))
