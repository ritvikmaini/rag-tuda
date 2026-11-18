"""What the indexer gives each connector: the context, a fetcher that holds data-derived downloads to the overlay's
allowed hosts, an atomic JSON state store under the settings' state_dir, and discovery of the profile's connectors by entry point (the app imports no connector package)."""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from rag_core.domain.sources import CorruptStateError
from rag_core.ports.sources import DocumentSource, Timeout
from rag_core.settings import SettingsError
from rag_core.settings.profile import Profile

from indexer.extract.quality import write_json_atomic
from indexer.settings import Settings

GROUP = "rag_core.connectors"


class DownloadRefused(RuntimeError):
    """A data-derived URL whose host the overlay does not allow (connectors.<name>.allowed_hosts)."""


class HttpFetcher:
    def __init__(self, endpoint: str | None, allowed_hosts: list[str] | None) -> None:
        self._endpoint, self._allowed = endpoint, allowed_hosts

    def endpoint(self, *, timeout: Timeout) -> requests.Response:
        if not self._endpoint:
            raise RuntimeError("the connector has no endpoint (overlay connectors.<name>.url or manifest_url)")
        return requests.get(self._endpoint, stream=True, timeout=timeout)

    def get(self, url: str, *, timeout: Timeout) -> requests.Response:
        host = urlparse(url).hostname
        if self._allowed is not None and host not in self._allowed:
            raise DownloadRefused(f"{host} is not an allowed host (overlay connectors.<name>.allowed_hosts)")
        return requests.get(url, stream=True, timeout=timeout)


class JsonStateStore:
    """load(): an absent file is None; an unreadable one or one that is not a JSON object raises CorruptStateError."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / name

    def load(self, name: str) -> Any:
        path = self.path(name)
        if not path.exists():
            logging.info(f"Downloaded URLs file does not yet exist locally: {path}")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CorruptStateError(
                f"{path} exists but cannot be read ({exc}); fix or remove it before the next build"
            ) from exc
        if not isinstance(data, dict):
            raise CorruptStateError(f"{path} does not hold a JSON object; fix or remove it before the next build")
        return data

    def save(self, name: str, data: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(str(self.path(name)), data)


@dataclass(frozen=True)
class Context:
    fetch: HttpFetcher
    state: JsonStateStore
    data_dir: Path | None
    endpoint: str | None
    settings: Mapping[str, Any]
    paths: Mapping[str, str]
    cache_dir: Path
    limit: int | None
    reuse: bool
    options: Mapping[str, Any]  # see SourceContext.options


def context(
    settings: Settings, name: str, limit: int | None, reuse: bool, options: Mapping[str, Any] | None = None
) -> Context:
    """The context of one source. `options` are the build's (`sources`: core or full, `resume`); the crawl contact and the
    profile's folder come from the settings."""
    endpoint = settings.endpoints[name]
    paths = settings.overlay.paths
    url = endpoint.url or endpoint.manifest_url
    return Context(
        fetch=HttpFetcher(url, endpoint.allowed_hosts),
        state=JsonStateStore(paths.state_dir),
        data_dir=Path(endpoint.dir) if endpoint.dir else None,
        endpoint=url,
        settings=settings.profile.ingest.sources[name].model_dump(),
        paths=paths.model_dump(),
        cache_dir=Path(paths.extracted_dir),
        limit=limit,
        reuse=reuse,
        options={
            "sources": "core",
            "resume": False,
            **(options or {}),
            "contact": settings.overlay.crawl.contact or None,
            "routes": settings.overlay.crawl.routes,
            "profile_dir": str(settings.profile_dir),
        },
    )


def discover(profile: Profile) -> dict[str, DocumentSource]:
    """The profile's sources, in its order, each an instance of the connector its entry names."""
    found = {ep.name: ep for ep in entry_points(group=GROUP)}
    sources: dict[str, DocumentSource] = {}
    for name, entry in profile.ingest.sources.items():
        if entry.connector not in found:
            raise SettingsError(
                f"ingest.sources.{name}.connector: no connector {entry.connector!r} is installed "
                f"(entry points {GROUP}: {sorted(found)})"
            )
        sources[name] = found[entry.connector].load()()
    return sources
