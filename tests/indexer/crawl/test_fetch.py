"""The fetcher against the fixture mini-site served in-process: host policy, robots.txt handling, the `<meta name="robots">`
page rule, the rate limit (a fake clock, nothing sleeps) and the user agent. No request leaves the process."""

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from rag_testing import serve
from rag_testing.minisite import pages

from indexer.crawl.cache import Cache
from indexer.crawl.fetch import Fetcher, Response, Skipped, user_agent
from indexer.crawl.manifest import ManifestDoc, snapshots_dir, write_manifest
from indexer.crawl.robots import Robots

FACULTY = "faculty.test"
INTERN = "intern.test"
CONTACT = "dummy@example.invalid"
HUB = f"http://{FACULTY}{pages.BSC_HUB}.de.jsp"
PO_PDF = f"http://{FACULTY}{pages.PO_2023_PDF}"
APB = f"http://{INTERN}{pages.APB_ORIGINAL_PDF}"
PO_PDF_SHA256 = "be6f4a46d0307d9cb0855af0f2e404b1d5b1fcd8dce6347a58d7489759da666a"  # the generated PDF is deterministic
STUDIENBUERO = f"http://{FACULTY}{pages.STUDIENBUERO}/index.de.jsp"
PRUEFUNGEN = f"http://{FACULTY}{pages.STUDIENBUERO}/pruefungen/index.de.jsp"
TODAY = date(2026, 11, 17)


class Net:
    """The mini-site's two hosts behind an httpx client; `log` records every request that reaches a host."""

    def __init__(self, overrides=None, clock=None):
        self.log: list[httpx.Request] = []
        self.times: list[float] = []  # the fake clock at each request, when a clock is given
        self.clock = clock
        self.overrides = overrides or {}  # url -> a function returning the httpx.Response to serve instead
        self.apps = {
            FACULTY: TestClient(serve.minisite_app("fb20", f"http://{INTERN}"), base_url=f"http://{FACULTY}"),
            INTERN: TestClient(serve.minisite_app("intern", f"http://{INTERN}"), base_url=f"http://{INTERN}"),
        }
        self.client = httpx.Client(transport=httpx.MockTransport(self._serve))

    def _serve(self, request: httpx.Request) -> httpx.Response:
        self.log.append(request)
        self.times.append(self.clock() if self.clock else 0.0)
        if str(request.url) in self.overrides:
            return self.overrides[str(request.url)](request)
        app = self.apps.get(request.url.host.removesuffix("."))
        if app is None:
            raise httpx.ConnectError("no such host", request=request)
        reply = app.get(
            request.url.raw_path.decode(), headers={"if-none-match": request.headers.get("if-none-match", "")}
        )
        headers = {k: v for k, v in reply.headers.items() if k.lower() not in ("content-length", "content-encoding")}
        return httpx.Response(reply.status_code, headers=headers, content=reply.content)

    def paths(self, host=FACULTY):
        return [r.url.raw_path.decode() for r in self.log if r.url.host == host]


class Clock:
    """A fake monotonic clock; sleeping moves it."""

    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def fetcher(net, clock=None, hosts=(FACULTY, INTERN), **kw):
    clock = clock or Clock()
    kw.setdefault("monotonic", clock)
    return Fetcher(hosts, contact=CONTACT, client=net.client, clock=clock, sleep=clock.sleep, **kw)


def caching(net, tmp_path, today=TODAY, **kw):
    """A fetcher with a cache under tmp_path/crawl and a fixed date."""
    return fetcher(net, cache=Cache(tmp_path / "crawl"), today=lambda: today, **kw)


class Broken(httpx.SyncByteStream):
    """A body that stops after `sent` bytes: with an error (the connection dropped) or silently (a short body)."""

    def __init__(self, body, sent, fail):
        self.body, self.sent, self.fail = body, sent, fail

    def __iter__(self):
        yield self.body[: self.sent]
        if self.fail:
            raise httpx.ReadError("connection reset")


def truncated(body, fail):
    return lambda r: httpx.Response(
        200,
        headers={"content-type": "application/pdf", "content-length": str(len(body))},
        stream=Broken(body, len(body) // 2, fail),
    )


def test_rate_limit_one_per_second_per_host():
    clock = Clock()
    net = Net(clock=clock)
    f = fetcher(net, clock)
    assert isinstance(f.get(HUB), Response)
    assert isinstance(f.get(f"http://{FACULTY}{pages.STUDIENBUERO}/index.de.jsp"), Response)
    assert isinstance(f.get(APB), Response)  # another host: its own budget
    assert net.paths() == ["/robots.txt", f"{pages.BSC_HUB}.de.jsp", f"{pages.STUDIENBUERO}/index.de.jsp"]
    for host in (FACULTY, INTERN):
        times = [t for t, r in zip(net.times, net.log, strict=True) if r.url.host == host]
        assert all(later - earlier == pytest.approx(1.0) for earlier, later in zip(times, times[1:], strict=False))
    # robots.txt and the first page of the other host went out at once: only the repeat requests waited
    assert clock.slept == [pytest.approx(1.0)] * 3


def test_meta_noai_skips_page():
    net = Net()
    result = fetcher(net).get(f"http://{FACULTY}{pages.NOAI_PAGE}")
    assert isinstance(result, Skipped) and result.reason == "meta robots"
    assert "noai" in result.detail


def test_user_agent_carries_contact_when_set():
    net = Net()
    fetcher(net).get(HUB)
    agents = {r.headers["user-agent"] for r in net.log}
    assert len(agents) == 1
    (agent,) = agents
    assert agent.startswith("rag-tuda/") and agent.endswith(f"(+{CONTACT})")


def test_bare_user_agent_without_contact():
    net = Net()
    Fetcher((FACULTY,), client=net.client, clock=Clock(), sleep=Clock().sleep).get(HUB)
    (agent,) = {r.headers["user-agent"] for r in net.log}
    assert agent.startswith("rag-tuda/") and agent.endswith(" (student research prototype)")
    assert "+" not in agent and "@" not in agent
    assert user_agent("x@example.invalid", "1.2") == "rag-tuda/1.2 (+x@example.invalid)"
    assert user_agent(None, "1.2") == "rag-tuda/1.2 (student research prototype)"


@pytest.mark.parametrize("host", ["www.darmstadt.de", "h-da.de", "www.h-da.de", "WWW.DARMSTADT.DE"])
def test_city_and_hda_hosts_are_never_fetched(host):
    net = Net()
    f = Fetcher((host.lower(), "www.h-da.de"), client=net.client, clock=Clock(), sleep=Clock().sleep)
    result = f.get(f"https://{host}/anything")
    assert isinstance(result, Skipped) and result.reason == "host never fetched"
    assert net.log == []


def test_host_outside_the_allow_list_is_skipped():
    net = Net()
    result = fetcher(net, hosts=(FACULTY,)).get(APB)
    assert isinstance(result, Skipped) and result.reason == "host not allowed" and net.log == []


def test_robots_disallow_skips_without_requesting_the_page():
    net = Net()
    f = fetcher(net)
    for path in (pages.LOGIN_PAGE, f"{pages.STUDIENBUERO}/pruefungen/index.de.jsp?view=print"):
        result = f.get(f"http://{FACULTY}{path}")
        assert isinstance(result, Skipped) and result.reason == "robots"
    assert net.paths() == ["/robots.txt"]  # robots.txt once per host, no page


def test_robots_sitemaps_are_exposed_per_host():
    f = fetcher(Net())
    assert f.robots(f"http://{FACULTY}/").sitemaps == [f"http://{FACULTY}/sitemap.de.xml"]


def robots_served(status, body="", content_type="text/plain"):
    return {
        f"http://{FACULTY}/robots.txt": lambda r: httpx.Response(
            status, text=body, headers={"content-type": content_type}
        )
    }


@pytest.mark.parametrize("status", [404, 410, 403])
def test_robots_4xx_allows_everything(status):
    f = fetcher(Net(robots_served(status, "<html><body>Not found</body></html>", "text/html")))
    assert isinstance(f.get(HUB), Response)


@pytest.mark.parametrize("status", [500, 503])
def test_robots_5xx_disallows_everything(status):
    net = Net(robots_served(status, "User-agent: *\nDisallow:\n"))
    f = fetcher(net, retries=0)
    result = f.get(HUB)
    assert isinstance(result, Skipped) and result.reason == f"robots.txt unreadable (http {status})"
    assert net.paths() == ["/robots.txt"]
    assert f.hosts_down() == {FACULTY: f"http {status}"}


def test_robots_unreachable_disallows_everything():
    def down(request):
        raise httpx.ConnectError("down", request=request)

    net = Net({f"http://{FACULTY}/robots.txt": down})
    f = fetcher(net, retries=1)
    result = f.get(HUB)
    assert isinstance(result, Skipped) and result.reason == "robots.txt unreadable (unreachable)"
    assert net.paths() == ["/robots.txt"] * 2  # the retry
    assert f.hosts_down() == {FACULTY: "unreachable"}


def test_robots_html_body_with_status_200_is_not_allow_all():
    net = Net(robots_served(200, "<!doctype html><html><body>Seite nicht gefunden</body></html>", "text/html"))
    result = fetcher(net).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "robots.txt unreadable (an HTML page)"


def test_robots_that_disallow_do_not_make_a_host_down():
    f = fetcher(Net())
    assert isinstance(f.get(f"http://{FACULTY}{pages.LOGIN_PAGE}"), Skipped) and f.hosts_down() == {}


def test_robots_kept_one_per_host():
    net = Net()
    f = fetcher(net)
    f.get(HUB)
    f.get(f"http://{FACULTY}{pages.STUDIENBUERO}/index.de.jsp")
    assert net.paths().count("/robots.txt") == 1
    assert isinstance(f.robots(HUB), Robots)


def test_a_server_error_is_retried_then_skipped():
    calls = []

    def flaky(request):
        calls.append(1)
        return (
            httpx.Response(503)
            if len(calls) < 3
            else httpx.Response(200, text="<html></html>", headers={"content-type": "text/html"})
        )

    net = Net({HUB: flaky})
    assert isinstance(fetcher(net, retries=2).get(HUB), Response)
    assert len(calls) == 3

    result = fetcher(Net({HUB: lambda r: httpx.Response(503)}), retries=1).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "http 503"


@pytest.mark.parametrize(
    ("status", "retry_after", "wait"), [(429, "7", 7.0), (503, None, 30.0), (429, "9999", 120.0), (503, "soon", 30.0)]
)
def test_a_host_that_asks_to_slow_down_keeps_that_interval_for_the_crawl(status, retry_after, wait):
    calls = []
    headers = {"retry-after": retry_after} if retry_after else {}

    def slow(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(status, headers=headers)
        return httpx.Response(200, text="<html></html>", headers={"content-type": "text/html"})

    clock = Clock()
    f = fetcher(Net({HUB: slow}, clock=clock), clock, retries=1)
    assert isinstance(f.get(HUB), Response)
    assert wait in clock.slept  # the retry waited
    clock.slept.clear()
    assert isinstance(f.get(f"http://{FACULTY}{pages.STUDIENBUERO}/index.de.jsp"), Response)
    assert clock.slept == [pytest.approx(wait)]  # and so does every later request to the host


def test_a_missing_page_is_skipped():
    result = fetcher(Net()).get(f"http://{FACULTY}/nothing.de.jsp")
    assert isinstance(result, Skipped) and result.reason == "http 404"


def test_a_redirect_to_a_host_outside_the_allow_list_is_not_followed():
    net = Net({HUB: lambda r: httpx.Response(302, headers={"location": "https://www.darmstadt.de/x"})})
    result = fetcher(net).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "host never fetched"
    assert all(r.url.host == FACULTY for r in net.log)


def test_a_redirect_inside_the_allow_list_is_followed():
    target = f"http://{INTERN}{pages.APB_ORIGINAL_PDF}"
    net = Net({HUB: lambda r: httpx.Response(301, headers={"location": target})})
    result = fetcher(net).get(HUB)
    assert isinstance(result, Response) and result.final_url == target and result.url == HUB


# the cache, conditional requests, interrupted PDFs, resume


def test_the_cache_keeps_body_date_validators_and_hash(tmp_path):
    net = Net(
        {
            HUB: lambda r: httpx.Response(
                200,
                text="<html><body>x</body></html>",
                headers={"content-type": "text/html", "etag": '"v1"', "last-modified": "modified-1"},
            )
        }
    )
    got = caching(net, tmp_path).get(HUB)
    assert isinstance(got, Response)
    assert got.content_hash == hashlib.sha256(got.body).hexdigest() and got.fetched == "2026-11-17"
    entry = Cache(tmp_path / "crawl").entry(HUB)
    assert entry is not None
    assert (entry.fetched, entry.etag, entry.last_modified) == ("2026-11-17", '"v1"', "modified-1")
    assert entry.content_hash == got.content_hash
    assert (
        got.path is not None and got.path.parent == tmp_path / "crawl" / FACULTY and got.path.read_bytes() == got.body
    )


def test_the_cache_lives_under_the_given_folder_not_the_working_directory(tmp_path, monkeypatch):
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    caching(Net(), tmp_path).get(HUB)
    assert list(elsewhere.iterdir()) == []
    assert (tmp_path / "crawl" / FACULTY).is_dir()


def test_a_validator_makes_the_next_request_conditional(tmp_path):
    served = []

    def page(request):
        served.append(dict(request.headers))
        if request.headers.get("if-none-match") == '"v1"' and request.headers.get("if-modified-since") == "Mon":
            return httpx.Response(304)
        return httpx.Response(
            200, text="<html>one</html>", headers={"content-type": "text/html", "etag": '"v1"', "last-modified": "Mon"}
        )

    net = Net({HUB: page})
    first = caching(net, tmp_path).get(HUB)
    second = caching(net, tmp_path, date(2026, 11, 18)).get(HUB)
    assert "if-none-match" not in served[0]
    assert isinstance(first, Response) and isinstance(second, Response)
    assert second.from_cache and not second.changed and second.body == first.body
    assert second.fetched == "2026-11-18" and Cache(tmp_path / "crawl").entry(HUB).fetched == "2026-11-18"


def test_without_validators_the_hash_decides(tmp_path):
    net = Net()
    first = caching(net, tmp_path).get(HUB)
    again = caching(net, tmp_path, date(2026, 11, 18)).get(HUB)
    assert "if-none-match" not in net.log[-1].headers and "if-modified-since" not in net.log[-1].headers
    assert isinstance(again, Response) and not again.from_cache and not again.changed
    assert again.content_hash == first.content_hash

    changed = Net({HUB: lambda r: httpx.Response(200, text="<html>new</html>", headers={"content-type": "text/html"})})
    new = caching(changed, tmp_path, date(2026, 11, 19)).get(HUB)
    assert isinstance(new, Response) and new.changed and new.content_hash != first.content_hash
    assert Cache(tmp_path / "crawl").entry(HUB).content_hash == new.content_hash


def test_a_page_that_turns_noai_leaves_the_cache(tmp_path):
    cache = Cache(tmp_path / "crawl")
    url = f"http://{FACULTY}{pages.NOAI_PAGE}"
    cache.store(url, url, b"<html>old</html>", "text/html", None, None, "2026-11-01")
    assert isinstance(caching(Net(), tmp_path).get(url), Skipped)
    assert cache.entry(url) is None
    assert not any(p.suffix == ".body" for p in (tmp_path / "crawl").rglob("*"))


def test_a_pdf_is_kept_as_a_complete_file(tmp_path):
    got = caching(Net(), tmp_path).get(PO_PDF)
    assert isinstance(got, Response) and got.path is not None
    assert got.content_hash == PO_PDF_SHA256 == hashlib.sha256(pages.PO_2023).hexdigest()
    assert got.path.suffix == ".pdf" and hashlib.sha256(got.path.read_bytes()).hexdigest() == PO_PDF_SHA256
    assert sorted(p.suffix for p in got.path.parent.iterdir()) == [".json", ".pdf"]  # the entry and the file


@pytest.mark.parametrize("fail", [True, False], ids=["connection dropped", "short body"])
def test_interrupted_pdf_is_not_registered(tmp_path, fail):
    net = Net({PO_PDF: truncated(pages.PO_2023, fail)})
    result = caching(net, tmp_path, retries=1).get(PO_PDF)
    assert isinstance(result, Skipped) and result.reason == "unreachable"
    assert len([r for r in net.log if str(r.url) == PO_PDF]) == 2
    crawl = tmp_path / "crawl"
    assert Cache(crawl).entry(PO_PDF) is None
    assert [p for p in crawl.rglob("*") if p.is_file() and p.suffix in (".pdf", ".part", ".json")] == []


def test_an_interrupted_download_keeps_the_pdf_it_would_have_replaced(tmp_path):
    ok = caching(Net(), tmp_path).get(PO_PDF)
    assert isinstance(ok, Response) and ok.path is not None
    broken = Net({PO_PDF: truncated(pages.PO_2023, True)})
    assert isinstance(caching(broken, tmp_path, date(2026, 11, 18), retries=0).get(PO_PDF), Skipped)
    entry = Cache(tmp_path / "crawl").entry(PO_PDF)
    assert entry is not None and entry.content_hash == PO_PDF_SHA256 and ok.path.read_bytes() == pages.PO_2023


def test_a_pdf_url_that_serves_a_page_is_not_kept(tmp_path):
    net = Net(
        {PO_PDF: lambda r: httpx.Response(200, text="<html>Fehler</html>", headers={"content-type": "application/pdf"})}
    )
    result = caching(net, tmp_path).get(PO_PDF)
    assert isinstance(result, Skipped) and result.reason == "not a pdf"
    assert Cache(tmp_path / "crawl").entry(PO_PDF) is None


def test_resume_does_not_refetch_what_the_interrupted_crawl_fetched(tmp_path):
    urls = [HUB, STUDIENBUERO, PRUEFUNGEN, PO_PDF]
    first = Net()
    f = caching(first, tmp_path)
    [f.get(u) for u in urls[:2]]  # the crawl stops here

    second = Net()
    resumed = caching(second, tmp_path, since=TODAY)
    got = [resumed.get(u) for u in urls]
    assert all(isinstance(g, Response) for g in got)
    assert [g.from_cache for g in got] == [True, True, False, False]
    assert [p for p in second.paths() if p != "/robots.txt"] == [
        f"{pages.STUDIENBUERO}/pruefungen/index.de.jsp",
        pages.PO_2023_PDF,
    ]


def test_an_offline_fetcher_answers_from_the_cache_alone_and_sends_nothing(tmp_path):
    caching(Net(), tmp_path).get(HUB)
    net = Net()
    offline = caching(net, tmp_path, offline=True)
    cached = offline.get(HUB)
    assert isinstance(cached, Response) and cached.from_cache and "Prüfungsordnung 2023".encode() in cached.body
    missing = offline.get(STUDIENBUERO)
    assert isinstance(missing, Skipped) and missing.reason == "not cached"
    refused = offline.get("https://www.darmstadt.de/")
    assert isinstance(refused, Skipped) and refused.reason == "host never fetched"
    assert net.log == []  # not even robots.txt


def test_resume_still_applies_the_robots_rules(tmp_path):
    url = f"http://{FACULTY}{pages.LOGIN_PAGE}"
    Cache(tmp_path / "crawl").store(url, url, b"<html>x</html>", "text/html", None, None, "2026-11-17")
    result = caching(Net(), tmp_path, since=TODAY).get(url)
    assert isinstance(result, Skipped) and result.reason == "robots"


def test_a_cache_entry_without_its_file_is_a_miss(tmp_path):
    cache = Cache(tmp_path / "crawl")
    entry = cache.store(HUB, HUB, b"<html>x</html>", "text/html", None, None, "2026-11-17")
    cache.file(entry).unlink()
    assert cache.entry(HUB) is None
    meta = next((tmp_path / "crawl").rglob("*.json"))
    meta.write_text("{not json", encoding="utf-8")
    assert cache.entry(HUB) is None


# the snapshot manifest


def crawl(f, urls):
    return [
        ManifestDoc.of(got, {"kind": "pdf" if got.url.endswith(".pdf") else "page"})
        for u in urls
        if isinstance(got := f.get(u), Response)
    ]


def test_resume_gives_same_manifest(tmp_path):
    urls = [HUB, STUDIENBUERO, PRUEFUNGEN, PO_PDF, APB]
    straight = write_manifest("g1", crawl(caching(Net(), tmp_path / "a"), urls), directory=tmp_path / "a" / "snapshots")

    interrupted = caching(Net(), tmp_path / "b")
    crawl(interrupted, urls[:3])  # the process dies here
    net = Net()
    resumed = write_manifest(
        "g1", crawl(caching(net, tmp_path / "b", since=TODAY), urls), directory=tmp_path / "b" / "snapshots"
    )

    assert resumed.read_bytes() == straight.read_bytes()
    assert [p for p in net.paths() if p != "/robots.txt"] == [pages.PO_2023_PDF]  # only the rest was requested
    assert {u["content_hash"] for u in json.loads(resumed.read_text(encoding="utf-8"))["documents"]} >= {PO_PDF_SHA256}


def test_the_manifest_lists_url_hash_date_and_metadata_by_url(tmp_path):
    docs = [
        ManifestDoc("https://b.example/x", "h2", "2026-11-17", {"programme": None}),
        ManifestDoc("https://a.example/y", "h1", "2026-11-16", {"title": "Prüfungsordnung", "po_version": [2023]}),
    ]
    path = write_manifest("2026-11-17-001", docs, directory=tmp_path)
    assert path == tmp_path / "2026-11-17-001.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["generation"] == "2026-11-17-001"
    assert data["documents"] == [
        {
            "url": "https://a.example/y",
            "content_hash": "h1",
            "fetched": "2026-11-16",
            "metadata": {"title": "Prüfungsordnung", "po_version": [2023]},
        },
        {"url": "https://b.example/x", "content_hash": "h2", "fetched": "2026-11-17", "metadata": {"programme": None}},
    ]
    assert "Prüfungsordnung" in path.read_text(encoding="utf-8")  # written as UTF-8, not escaped


def test_a_failed_manifest_write_leaves_the_previous_one(tmp_path, monkeypatch):
    first = write_manifest("g", [ManifestDoc("https://a.example/", "h1", "2026-11-17", {})], directory=tmp_path)
    before = first.read_bytes()

    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        write_manifest("g", [ManifestDoc("https://a.example/", "h2", "2026-11-18", {})], directory=tmp_path)
    assert first.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == ["g.json"]


@pytest.mark.parametrize("generation", ["", "../x", "a/b", "."])
def test_a_generation_that_is_not_a_file_name_is_refused(tmp_path, generation):
    with pytest.raises(ValueError, match="generation"):
        write_manifest(generation, [], directory=tmp_path)


def test_snapshots_sit_beside_the_crawl_folder():
    assert snapshots_dir("/x/.rag/crawl") == Path("/x/.rag/snapshots")


# bounds on one response, hosts written oddly, malformed headers, redirects


class Drip(httpx.SyncByteStream):
    """A body that never ends: one chunk every `every` seconds of the fake clock."""

    def __init__(self, clock, every, chunk=b"x"):
        self.clock, self.every, self.chunk, self.sent = clock, every, chunk, 0

    def __iter__(self):
        while self.sent < 10_000:
            self.sent += 1
            self.clock.now += self.every
            yield self.chunk


class Unread(httpx.SyncByteStream):
    def __iter__(self):
        raise AssertionError("the body was read")
        yield b""


def served(content_type, stream, **headers):
    return lambda r: httpx.Response(200, headers={"content-type": content_type, **headers}, stream=stream)


def test_a_slow_drip_response_hits_the_total_deadline(tmp_path):
    clock = Clock()
    drip = Drip(clock, every=10.0)
    net = Net({HUB: served("text/html", drip)})
    result = caching(net, tmp_path, clock=clock, deadline=35.0, retries=2).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "too slow"
    assert drip.sent == 4  # the fourth chunk arrived at 40 s
    assert len([r for r in net.log if str(r.url) == HUB]) == 1  # no retry
    assert Cache(tmp_path / "crawl").entry(HUB) is None


def test_a_slow_drip_pdf_leaves_no_part_file(tmp_path):
    clock = Clock()
    net = Net({PO_PDF: served("application/pdf", Drip(clock, 10.0, b"%PDF-1.4 "))})
    result = caching(net, tmp_path, clock=clock, deadline=35.0).get(PO_PDF)
    assert isinstance(result, Skipped) and result.reason == "too slow"
    assert [p for p in (tmp_path / "crawl").rglob("*") if p.is_file()] == []


def test_a_content_length_over_the_limit_is_refused_before_the_body_is_read(tmp_path):
    net = Net({HUB: served("text/html", Unread(), **{"content-length": "5000"})})
    result = caching(net, tmp_path, page_limit=1000).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "too large"
    assert Cache(tmp_path / "crawl").entry(HUB) is None


def test_a_body_over_the_limit_without_a_content_length_is_cut_off(tmp_path):
    clock = Clock()
    drip = Drip(clock, 0.0, b"y" * 400)
    net = Net({HUB: served("text/html", drip)})
    result = caching(net, tmp_path, clock=clock, page_limit=1000).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "too large"
    assert drip.sent == 3  # 1200 bytes > 1000
    assert Cache(tmp_path / "crawl").entry(HUB) is None


def test_pdfs_have_their_own_larger_limit(tmp_path):
    ok = caching(Net(), tmp_path, page_limit=1000, pdf_limit=10_000).get(PO_PDF)  # 2406 bytes
    assert isinstance(ok, Response) and ok.content_hash == PO_PDF_SHA256 and ok.body == pages.PO_2023
    over = caching(Net(), tmp_path / "other", pdf_limit=1000).get(PO_PDF)
    assert isinstance(over, Skipped) and over.reason == "too large"
    assert [p for p in (tmp_path / "other" / "crawl").rglob("*") if p.is_file()] == []


def test_a_pdf_streams_into_the_cache_and_leaves_no_part_file(tmp_path):
    got = caching(Net(), tmp_path).get(PO_PDF)
    assert isinstance(got, Response) and got.path is not None
    assert not list((tmp_path / "crawl").rglob("*.part"))
    again = caching(Net(), tmp_path, date(2026, 11, 18)).get(PO_PDF)
    assert isinstance(again, Response) and not again.changed and again.path == got.path
    assert not list((tmp_path / "crawl").rglob("*.part"))


def test_a_robots_txt_over_the_limit_disallows_the_host(tmp_path):
    net = Net(robots_served(200, "User-agent: *\n" + "# padding\n" * 500))
    result = caching(net, tmp_path, page_limit=1000).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "robots.txt unreadable (too large)"


@pytest.mark.parametrize("host", ["www.darmstadt.de.", "WWW.Darmstadt.DE.", "h-da.de.", "www.h-da.de."])
def test_a_trailing_dot_does_not_get_past_the_never_fetched_hosts(host):
    net = Net()
    f = Fetcher((host, host.rstrip(".")), client=net.client, clock=Clock(), sleep=Clock().sleep)
    result = f.get(f"https://{host}/anything")
    assert isinstance(result, Skipped) and result.reason == "host never fetched"
    assert net.log == []


def test_a_redirect_to_a_trailing_dot_host_is_refused():
    net = Net({HUB: lambda r: httpx.Response(302, headers={"location": "https://www.darmstadt.de./x"})})
    result = fetcher(net, hosts=(FACULTY, "www.darmstadt.de.")).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "host never fetched"
    assert {r.url.host for r in net.log} == {FACULTY}


def test_a_trailing_dot_on_an_allowed_host_still_matches():
    assert isinstance(fetcher(Net(), hosts=(FACULTY + ".",)).get(HUB), Response)
    assert isinstance(fetcher(Net()).get(HUB.replace(FACULTY, FACULTY + ".")), Response)


def test_a_malformed_content_length_is_skipped():
    net = Net({HUB: served("text/html", httpx.ByteStream(b"<html></html>"), **{"content-length": "12x"})})
    result = fetcher(net).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "bad content-length"


@pytest.mark.parametrize("location", ["http://[::1", "http://faculty.test:notaport/"])
def test_a_malformed_location_is_skipped(location):
    net = Net({HUB: lambda r: httpx.Response(302, headers={"location": location})})
    result = fetcher(net).get(HUB)
    assert isinstance(result, Skipped)


def test_a_malformed_url_is_skipped():
    result = fetcher(Net()).get("http://[::1")
    assert isinstance(result, Skipped) and result.reason == "not an http url"


def test_a_malformed_location_in_robots_txt_disallows_the_host():
    net = Net({f"http://{FACULTY}/robots.txt": lambda r: httpx.Response(302, headers={"location": "http://[::1"})})
    result = fetcher(net).get(HUB)
    assert isinstance(result, Skipped) and result.reason.startswith("robots.txt unreadable")


def test_a_redirect_to_a_path_robots_disallows_is_not_followed():
    login = f"http://{FACULTY}{pages.LOGIN_PAGE}"
    net = Net({HUB: lambda r: httpx.Response(302, headers={"location": login})})
    result = fetcher(net).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "robots" and result.url == HUB
    assert login not in {str(r.url) for r in net.log}


def test_the_rate_limit_holds_across_a_redirect():
    clock = Clock()
    target = f"http://{FACULTY}{pages.STUDIENBUERO}/index.de.jsp"
    net = Net({HUB: lambda r: httpx.Response(302, headers={"location": target})}, clock=clock)
    assert isinstance(fetcher(net, clock).get(HUB), Response)
    assert [r.url.path for r in net.log] == [
        "/robots.txt",
        f"{pages.BSC_HUB}.de.jsp",
        f"{pages.STUDIENBUERO}/index.de.jsp",
    ]
    assert net.times == [1000.0, 1001.0, 1002.0]
