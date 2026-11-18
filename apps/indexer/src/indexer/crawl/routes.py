"""The crawl's routing table: where the requests for some hosts are sent instead of the host.

The overlay's `crawl.routes` maps a host to the origin (`scheme://host[:port]`) that answers for it. The harness uses it to
send the profile's hosts to the fixture mini-site on its own network; the URLs, the host allow-list, robots.txt rules and
the cache keys all stay those of the real hosts. With a table, a host that has no route is unreachable, so nothing leaves
for the internet."""

from collections.abc import Mapping

import httpx


class Routed(httpx.BaseTransport):
    def __init__(self, routes: Mapping[str, str], network: httpx.BaseTransport | None = None) -> None:
        self.routes = {host.lower(): httpx.URL(origin) for host, origin in routes.items()}
        self._network = network or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        route = self.routes.get(request.url.host)
        if route is None:
            raise httpx.ConnectError(f"{request.url.host}: no route (crawl.routes lists the hosts that can be reached)")
        request.url = request.url.copy_with(scheme=route.scheme, host=route.host, port=route.port)
        return self._network.handle_request(request)

    def close(self) -> None:
        self._network.close()
