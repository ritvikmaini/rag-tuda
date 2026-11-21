"""The web connectors' fixtures: a context on temporary folders whose profile folder is a copy of the TU profile's crawl
files (a test edits the copy), and an httpx client that serves a dict of URLs in-process. No request leaves the process."""

import shutil
from pathlib import Path

import httpx
import pytest

from indexer.sources import Context, HttpFetcher, JsonStateStore

TU = Path(__file__).resolve().parents[3] / "profiles" / "tu-informatik"
CONTACT = "dummy@example.invalid"


class Served:
    """An httpx client over `handler(request) -> httpx.Response | None` (None: 404); `log` holds every request."""

    def __init__(self, handler):
        self.log: list[httpx.Request] = []
        self.handler = handler
        self.client = httpx.Client(transport=httpx.MockTransport(self._serve))

    def _serve(self, request: httpx.Request) -> httpx.Response:
        self.log.append(request)
        return self.handler(request) or httpx.Response(404)

    def urls(self) -> list[str]:
        return [str(r.url) for r in self.log]


@pytest.fixture
def served():
    """The Served class: served(handler) is an in-process client."""
    return Served


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "profile"
    folder.mkdir()
    for name in ("sources.yaml", "roles.yaml"):
        shutil.copy(TU / name, folder / name)
    return folder


@pytest.fixture
def make_context(tmp_path: Path, profile_dir: Path):
    """make_context(**changes) -> Context; `options=` are merged over the defaults (core set, fresh crawl, the dummy contact),
    any other keyword replaces the context's field of that name."""

    def make(options=None, **changes) -> Context:
        paths = {
            "crawl_dir": str(tmp_path / "crawl"),
            "state_dir": str(tmp_path / "state"),
            "extracted_dir": str(tmp_path / "extracted"),
            "pdf_dir": str(tmp_path / "pdfs"),
            "skipped_pdfs_log": str(tmp_path / "skipped.json"),
        }
        values = {
            "fetch": HttpFetcher(None, None),
            "state": JsonStateStore(paths["state_dir"]),
            "data_dir": None,
            "endpoint": None,
            "settings": {},
            "paths": paths,
            "cache_dir": Path(paths["extracted_dir"]),
            "limit": None,
            "reuse": False,
            "options": {
                "sources": "core",
                "resume": False,
                "contact": CONTACT,
                "profile_dir": str(profile_dir),
                **(options or {}),
            },
        }
        Path(paths["extracted_dir"]).mkdir(parents=True, exist_ok=True)
        return Context(**{**values, **changes})

    return make
