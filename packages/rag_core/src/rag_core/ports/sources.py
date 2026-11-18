"""The connector interface. The indexer gives each source a context and receives items; extraction, chunking and the gates
stay in the indexer. The context carries the extraction cache, the overlay paths and the build options, and a source names
the state files the indexer reads before any download."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from rag_core.domain.sources import SourceResult

Timeout = float | tuple[float, float]


class Fetcher(Protocol):
    def endpoint(self, *, timeout: Timeout) -> Any:
        """A streaming GET of the connector's own endpoint (the overlay's url or manifest_url); a requests.Response."""
        ...

    def get(self, url: str, *, timeout: Timeout) -> Any:
        """A streaming GET of a URL that came from data (a list's download URL); refused unless its host is allowed."""
        ...


class StateStore(Protocol):
    def path(self, name: str) -> Path:
        """Where the state file lives (the paths are stable: an older image shares them)."""
        ...

    def load(self, name: str) -> Any:
        """The parsed JSON object; None when absent; CorruptStateError when it cannot be read or is no object."""
        ...

    def save(self, name: str, data: Any) -> None:
        """Atomic: a temporary file and os.replace."""
        ...


class SourceContext(Protocol):
    @property
    def fetch(self) -> Fetcher: ...
    @property
    def state(self) -> StateStore: ...
    @property
    def data_dir(self) -> Path | None: ...  # overlay connectors.<name>.dir
    @property
    def endpoint(self) -> str | None: ...  # overlay connectors.<name>.url or .manifest_url
    @property
    def settings(self) -> Mapping[str, Any]: ...  # the source's profile entry (ingest.sources.<name>)
    @property
    def paths(self) -> Mapping[str, str]: ...  # overlay paths (crawl_dir, manifest_dir)
    @property
    def cache_dir(self) -> Path: ...  # the indexer's extraction cache (a changed PDF's entry is deleted)
    @property
    def limit(self) -> int | None: ...
    @property
    def reuse(self) -> bool: ...  # frozen inputs: nothing is fetched


class DocumentSource(Protocol):
    name: str
    state_files: Sequence[str]

    def collect(self, ctx: SourceContext) -> SourceResult: ...
