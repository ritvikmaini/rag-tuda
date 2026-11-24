"""The polite fetcher: one GET per URL through the host policy, robots.txt, the rate limit and the page's own robots meta,
with a cache of what it fetched.

A URL is fetched only when its host is on the caller's allow-list and not one of the hosts that are never crawled, when the
host's robots.txt allows it for this user agent, and (for a page) when its `<meta name="robots">` carries none of `noindex`,
`noai` or `none`. Requests to one host are at least `interval` seconds apart; the clock and the sleep are injected. A
server error or a dropped connection is retried `retries` times, each retry through the rate limit. After a 429 or 503
the host's interval becomes its Retry-After seconds (at most 120; 30 without the header) for the rest of the crawl. Redirects are followed
by hand, and every hop passes the same checks as the first URL. Host names are compared lower-cased, in their IDNA form and
without a trailing dot. A fetcher is for one thread: the rate limit and the robots rules are plain dictionaries.

One response is bounded in time and size, or the URL is skipped (no retry, nothing cached): `deadline` seconds in total
(a monotonic clock, apart from the rate limit's), `page_limit` bytes for anything but a PDF or a zip and for robots.txt,
`pdf_limit` bytes for a PDF, `zip_limit` bytes for a zip (the official law archives). A Content-Length
above the limit is refused before the body is read. A PDF goes to a `.part` file in the
cache as it arrives, never into memory beside it.

robots.txt is read once per host (RFC 9309): a 4xx answer allows everything; a server error, no answer, a 429 or a 200 that
is an HTML page (a soft 404 that is not a robots file) disallows everything.

With a cache, a URL whose entry has a validator is asked conditionally (`If-None-Match`, `If-Modified-Since`; a 304 keeps the
cached body); without one the new body's hash is compared with the cached hash. `since` resumes an interrupted crawl: a URL
cached on or after that date is answered from the cache without a request (robots.txt still applies). A PDF is cached only
as a complete file that starts with `%PDF-`; a download that breaks off leaves the cache as it was. An `offline` fetcher
sends no request at all, not even for robots.txt: it answers a URL from the cache or skips it ("not cached")."""

import hashlib
import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from urllib.parse import SplitResult, urljoin, urlsplit

import httpx
from lxml import html

from indexer.crawl.cache import Cache, Entry
from indexer.crawl.robots import Robots

__all__ = ["NEVER_FETCHED", "Fetcher", "Response", "Skipped", "user_agent"]

# whatever the allow-list says: www.darmstadt.de (Darmstadt's main site) blocks crawlers by name, and h-da.de is not
# part of this project
NEVER_FETCHED = ("www.darmstadt.de", "h-da.de")
BLOCKING_META = frozenset({"noindex", "noai", "none"})
REDIRECTS = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 5
PAGE_LIMIT = 5 * 1024 * 1024
PDF_LIMIT = 100 * 1024 * 1024
ZIP_LIMIT = 32 * 1024 * 1024
DEADLINE = 120.0
RETRY_AFTER_CAP = 120.0
RETRY_AFTER_DEFAULT = 30.0  # the slower interval after a 429 or 503 that names none
PDF_MAGIC = b"%PDF-"
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
    file: Path | None = None  # a PDF streamed into the cache's `.part` file; `body` is empty then
    digest: str = ""  # of that file


class _Refused(Exception):
    """A response that is dropped for a reason that retrying cannot change."""


@dataclass(frozen=True)
class Skipped:
    """A URL that was not fetched or not kept, and why (`reason` is one of a few fixed phrases; `detail` adds to it)."""

    url: str
    reason: str
    detail: str = ""


def _parts(url: str) -> SplitResult | None:
    try:
        return urlsplit(url)
    except ValueError:  # an unbalanced bracket in the host, say
        return None


def _normal(host: str) -> str:
    """A host name as compared: lower case, IDNA (as httpx sends it), one trailing dot removed."""
    try:
        host = host.lower().encode("idna").decode("ascii")
    except UnicodeError:
        return ""
    return host.removesuffix(".")


def _host(url: str) -> str:
    parts = _parts(url)
    return _normal(parts.hostname or "") if parts else ""


def _never_fetched(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in NEVER_FETCHED)


def _is_pdf(url: str, content_type: str) -> bool:
    return "pdf" in content_type.lower() or (_parts(url) or urlsplit("")).path.lower().endswith(".pdf")


def _is_zip(url: str, content_type: str) -> bool:
    return "zip" in content_type.lower() or (_parts(url) or urlsplit("")).path.lower().endswith(".zip")


def blocking_meta(body: bytes) -> str | None:
    """The first blocking token of the page's `<meta name="robots">` tags, or None."""
    try:
        tree = html.fromstring(body)
    except (ValueError, html.etree.ParserError):  # empty or not a document
        return None
    for meta in tree.iter("meta"):
        if (meta.get("name") or "").strip().lower() in ("robots", "rag-tuda"):
            tokens = set(re.split(r"[\s,]+", (meta.get("content") or "").lower()))
            if found := sorted(tokens & BLOCKING_META):
                return found[0]
    return None


def _retry_after(value: str | None) -> float:
    """The seconds a Retry-After header asks for, capped; a missing or non-numeric one (a date) counts as the default."""
    try:
        return min(max(float(value), 0.0), RETRY_AFTER_CAP) if value else RETRY_AFTER_DEFAULT
    except ValueError:
        return RETRY_AFTER_DEFAULT


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
        deadline: float = DEADLINE,
        page_limit: int = PAGE_LIMIT,
        pdf_limit: int = PDF_LIMIT,
        zip_limit: int = ZIP_LIMIT,
        monotonic: Callable[[], float] = time.monotonic,
        cache: Cache | None = None,
        since: date | None = None,
        today: Callable[[], date] = date.today,
        offline: bool = False,
    ):
        self.allowed_hosts = frozenset(_normal(h) for h in allowed_hosts)
        self.user_agent = user_agent(contact)
        self._client = client or httpx.Client(timeout=timeout)
        self._clock, self._sleep, self._interval, self._retries = clock, sleep, interval, retries
        self._last_request: dict[str, float] = {}
        self._intervals: dict[str, float] = {}  # a host that asked to slow down
        self._robots: dict[str, Robots] = {}
        self._deadline, self._page_limit, self._pdf_limit, self._monotonic = deadline, page_limit, pdf_limit, monotonic
        self._zip_limit = zip_limit
        self._cache, self._since, self._today, self._offline = cache, since, today, offline

    def get(self, url: str) -> Response | Skipped:
        """Fetch `url`, or say why not."""
        asked = url
        if self._offline:
            return self._answer_offline(asked)
        if skipped := self._refuse(asked, asked):
            return skipped
        cached = self._cache.entry(asked) if self._cache else None
        if cached and self._since and cached.fetched >= self._since.isoformat():
            return self._cached(cached, changed=False)
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
                try:
                    url = urljoin(url, reply.headers["location"])
                except ValueError:
                    return Skipped(asked, "bad redirect", reply.headers["location"])
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

    def _answer_offline(self, url: str) -> Response | Skipped:
        parts, host = _parts(url), _host(url)
        if parts is None or parts.scheme not in ("http", "https") or not host:
            return Skipped(url, "not an http url", url)
        if reason := self._host_refusal(host):
            return Skipped(url, reason, host)
        entry = self._cache.entry(url) if self._cache else None
        return self._cached(entry, changed=False) if entry else Skipped(url, "not cached")

    def _keep(self, asked: str, url: str, reply: "_Reply", cached: Entry | None) -> Response | Skipped:
        content_type = reply.headers.get("content-type", "")
        if reply.file is None and "html" in content_type.lower() and (token := blocking_meta(reply.body)):
            if self._cache:
                self._cache.drop(asked)
            return Skipped(asked, "meta robots", token)
        digest = reply.digest or hashlib.sha256(reply.body).hexdigest()
        etag, modified = reply.headers.get("etag"), reply.headers.get("last-modified")
        fetched = self._today().isoformat()
        same = bool(cached and cached.content_hash == digest)
        entry = None
        if self._cache and cached and same:
            entry = self._cache.touch(cached, fetched, etag, modified)
            if reply.file:
                reply.file.unlink()
        elif self._cache and reply.file:
            entry = self._cache.adopt(asked, url, reply.file, digest, content_type, etag, modified, fetched)
        elif self._cache:
            entry = self._cache.store(asked, url, reply.body, content_type, etag, modified, fetched)
        path = self._cache.file(entry) if self._cache and entry else None
        body = path.read_bytes() if reply.file and path else reply.body
        return Response(asked, url, reply.status, body, content_type, digest, fetched, etag, modified, path, not same)

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
        host, parts = _host(url), _parts(url)
        if host not in self._robots:
            self._robots[host] = (
                self._read_robots(f"{parts.scheme}://{parts.netloc}/robots.txt")
                if parts
                else Robots.disallow_all("not an http url")
            )
        return self._robots[host]

    def hosts_down(self) -> dict[str, str]:
        """The hosts whose robots.txt could not be read, with why: nothing was fetched from them."""
        return {host: robots.unreadable for host, robots in self._robots.items() if robots.unreadable}

    def _host_refusal(self, host: str) -> str | None:
        if _never_fetched(host):
            return "host never fetched"
        return None if host in self.allowed_hosts else "host not allowed"

    def _refuse(self, url: str, asked: str) -> Skipped | None:
        host, parts = _host(url), _parts(url)
        if parts is None or parts.scheme not in ("http", "https") or not host:
            return Skipped(asked, "not an http url", url)
        if reason := self._host_refusal(host):
            return Skipped(asked, reason, host)
        robots = self.robots(url)
        if robots.unreadable:
            return Skipped(asked, f"robots.txt unreadable ({robots.unreadable})", url)
        if not robots.allowed(self.user_agent, url):
            return Skipped(asked, "robots", url)
        return None

    def _read_robots(self, url: str) -> Robots:
        reply = self._send(url)
        for _ in range(MAX_REDIRECTS):  # RFC 9309: follow redirects, here only to hosts that may be fetched
            if not (isinstance(reply, _Reply) and reply.status in REDIRECTS and reply.headers.get("location")):
                break
            try:
                url = urljoin(url, reply.headers["location"])
            except ValueError:
                return Robots.disallow_all("bad redirect")
            reply = "refused" if self._host_refusal(_host(url)) else self._send(url)
        if isinstance(reply, _Reply) and reply.file:
            reply.file.unlink(missing_ok=True)  # robots.txt served as a PDF is not a robots file
            return Robots.disallow_all("a PDF")
        if isinstance(reply, str) or reply.status >= 500 or reply.status == 429 or 300 <= reply.status < 400:
            why = reply if isinstance(reply, str) else f"http {reply.status}"
            log.warning("robots.txt of %s is unreadable (%s): nothing is fetched from that host", url, why)
            return Robots.disallow_all(why)
        if reply.status >= 400:
            return Robots.parse("")
        if "html" in reply.headers.get("content-type", "").lower():  # a soft 404 is allow-all only on a 4xx status
            return Robots.disallow_all("an HTML page")
        return Robots.parse(reply.body.decode("utf-8", "replace"))

    def _wait(self, host: str) -> None:
        now = self._clock()
        last, interval = self._last_request.get(host), self._intervals.get(host, self._interval)
        if last is not None and now - last < interval:
            self._sleep(interval - (now - last))
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
            except _Refused as exc:
                return str(exc)
            if last.status in (429, 503):
                self._intervals[_host(url)] = max(self._interval, _retry_after(last.headers.get("retry-after")))
            if last.status < 500 and last.status != 429:
                break
        return last or "unreachable"

    def _once(self, url: str, headers: dict[str, str]) -> _Reply:
        started = self._monotonic()
        with self._client.stream("GET", url, headers={"User-Agent": self.user_agent, **headers}) as reply:
            status = reply.status_code
            pdf = 200 <= status < 300 and _is_pdf(url, reply.headers.get("content-type", ""))
            content_type = reply.headers.get("content-type", "")
            zipped = 200 <= status < 300 and not pdf and _is_zip(url, content_type)
            limit = self._pdf_limit if pdf else self._zip_limit if zipped else self._page_limit
            declared = reply.headers.get("content-length")
            try:
                size = int(declared) if declared is not None else None
            except ValueError:
                raise _Refused("bad content-length") from None
            if size is not None and not 0 <= size <= limit:
                raise _Refused("too large")
            part = self._cache.part(url) if pdf and self._cache else None
            chunks: list[bytes] = []
            digest, head, total = hashlib.sha256(), b"", 0
            try:
                out = part.open("wb") if part else None
                try:
                    for chunk in reply.iter_bytes():
                        if self._monotonic() - started > self._deadline:
                            raise _Refused("too slow")
                        total += len(chunk)
                        if total > limit:
                            raise _Refused("too large")
                        head = (head + chunk)[: len(PDF_MAGIC)] if len(head) < len(PDF_MAGIC) else head
                        if pdf and len(head) >= len(PDF_MAGIC) and head != PDF_MAGIC:
                            raise _Refused("not a pdf")
                        if out:
                            out.write(chunk)
                            digest.update(chunk)
                        else:
                            chunks.append(chunk)
                finally:
                    if out:
                        out.close()
                if pdf and head != PDF_MAGIC:
                    raise _Refused("not a pdf")
                if size is not None and "content-encoding" not in reply.headers and size != total:
                    raise httpx.ReadError("the body is shorter than its content-length")
            except BaseException:
                if part:
                    part.unlink(missing_ok=True)
                raise
            return _Reply(status, reply.headers, b"".join(chunks), part, digest.hexdigest() if part else "")
