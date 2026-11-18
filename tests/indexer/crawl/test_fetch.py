"""The fetcher against the fixture mini-site served in-process: host policy, robots.txt handling, the `<meta name="robots">`
page rule, the rate limit (a fake clock, nothing sleeps) and the user agent. No request leaves the process."""

import httpx
import pytest
from fastapi.testclient import TestClient
from rag_testing import serve
from rag_testing.minisite import pages

from indexer.crawl.fetch import Fetcher, Response, Skipped, user_agent
from indexer.crawl.robots import Robots

FACULTY = "faculty.test"
INTERN = "intern.test"
CONTACT = "dummy@example.invalid"
HUB = f"http://{FACULTY}{pages.BSC_HUB}.de.jsp"
PO_PDF = f"http://{FACULTY}{pages.PO_2023_PDF}"
APB = f"http://{INTERN}{pages.APB_ORIGINAL_PDF}"


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
        app = self.apps.get(request.url.host)
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
    return Fetcher(hosts, contact=CONTACT, client=net.client, clock=clock, sleep=clock.sleep, **kw)


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
    result = fetcher(net, retries=0).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "robots"
    assert net.paths() == ["/robots.txt"]


def test_robots_unreachable_disallows_everything():
    def down(request):
        raise httpx.ConnectError("down", request=request)

    net = Net({f"http://{FACULTY}/robots.txt": down})
    result = fetcher(net, retries=1).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "robots"
    assert net.paths() == ["/robots.txt"] * 2  # the retry


def test_robots_html_body_with_status_200_is_not_allow_all():
    net = Net(robots_served(200, "<!doctype html><html><body>Seite nicht gefunden</body></html>", "text/html"))
    result = fetcher(net).get(HUB)
    assert isinstance(result, Skipped) and result.reason == "robots"


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
