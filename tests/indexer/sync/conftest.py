"""Fakes for the sync tests: a fetcher that answers from a dict of bodies and records every call, and a context on a
temporary folder around a real JsonStateStore."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import requests

from indexer.sources import DownloadRefused, JsonStateStore

Body = bytes | Callable[[], bytes]


class FakeResponse:
    def __init__(self, url: str, body: bytes | None) -> None:
        self.url, self._body = url, body

    def raise_for_status(self) -> None:
        if self._body is None:
            raise requests.HTTPError(f"404 Not Found: {self.url}")

    def iter_content(self, chunk_size: int = 8192):
        body = self._body or b""
        for start in range(0, len(body), chunk_size):
            yield body[start : start + chunk_size]


@dataclass
class FakeFetcher:
    """`get` serves bodies[url] (absent: HTTP 404) unless allowed_hosts refuses the host, which records no call. A
    callable body is called per request."""

    bodies: dict[str, Body] = field(default_factory=dict)
    allowed_hosts: list[str] | None = None
    calls: list[tuple] = field(default_factory=list)

    def endpoint(self, *, timeout):
        raise AssertionError("the sync never reads the connector's own endpoint")

    def get(self, url: str, *, timeout):
        host = urlparse(url).hostname
        if self.allowed_hosts is not None and host not in self.allowed_hosts:
            raise DownloadRefused(f"{host} is not an allowed host")
        self.calls.append(("get", url, timeout))
        body = self.bodies.get(url)
        return FakeResponse(url, body() if callable(body) else body)


@dataclass(frozen=True)
class FakeContext:
    fetch: Any
    state: JsonStateStore
    cache_dir: Path
    settings: Mapping[str, Any] = field(default_factory=dict)
    data_dir: Path | None = None
    endpoint: str | None = None
    paths: Mapping[str, str] = field(default_factory=dict)
    limit: int | None = None
    reuse: bool = False


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    return tmp_path / "pdfs"


@pytest.fixture
def make_ctx(tmp_path: Path):
    """make_ctx(bodies, allowed_hosts, **fields): a FakeContext on tmp_path whose FakeFetcher serves bodies; the cache folder
    exists, the state folder is created on first save. Variants of one context: dataclasses.replace(ctx, limit=…)."""

    def make(bodies: dict[str, Body] | None = None, allowed_hosts: list[str] | None = None, **fields: Any):
        values: dict[str, Any] = {
            "fetch": FakeFetcher(bodies or {}, allowed_hosts),
            "state": JsonStateStore(tmp_path / "state"),
            "cache_dir": tmp_path / "extracted",
        }
        values.update(fields)
        values["cache_dir"].mkdir(parents=True, exist_ok=True)
        return FakeContext(**values)

    return make
