"""The crawl's routing table: a routed host is asked at its route's origin with its own name in the request, and a host
without a route is not asked at all."""

import dataclasses

import httpx
import pytest

from indexer import build
from indexer.connectors.crawl_setup import make_fetcher
from indexer.crawl.fetch import Skipped
from indexer.crawl.routes import Routed

ROUTES = {"www.faculty.test": "http://minisite:8091", "www.intern.test": "https://minisite-intern"}


def routed(seen: list[httpx.Request]) -> httpx.Client:
    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"User-agent: *\nAllow: /\n")

    return httpx.Client(transport=Routed(ROUTES, httpx.MockTransport(answer)))


def test_a_request_goes_to_the_routes_origin_with_the_hosts_name():
    seen: list[httpx.Request] = []
    client = routed(seen)
    client.get("https://www.faculty.test/a/b.jsp?x=1")
    client.get("http://www.intern.test:8080/c")
    assert [str(r.url) for r in seen] == ["http://minisite:8091/a/b.jsp?x=1", "https://minisite-intern/c"]
    assert [r.headers["host"] for r in seen] == ["www.faculty.test", "www.intern.test:8080"]


def test_a_host_without_a_route_is_not_asked():
    seen: list[httpx.Request] = []
    with pytest.raises(httpx.ConnectError, match="no route"):
        routed(seen).get("https://www.tu-darmstadt.de/")
    assert seen == []


def test_the_fetcher_of_a_context_with_routes_fetches_nothing_from_an_unrouted_host(settings):
    ctx = build.context(settings, "web", None, False)
    ctx = dataclasses.replace(ctx, options={**ctx.options, "routes": ROUTES})
    reply = make_fetcher(ctx, ["www.faculty.test", "www.other.test"], interval=0).get("https://www.other.test/page.jsp")
    assert (
        isinstance(reply, Skipped) and reply.reason == "robots.txt unreadable (unreachable)"
    )  # robots.txt could not be read: nothing is fetched
