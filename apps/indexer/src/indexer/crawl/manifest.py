"""The snapshot manifest: `<snapshots>/<generation>.json` lists, per document of a build, its URL, content hash, fetch date
and metadata. Documents are sorted by URL, so the same crawl gives the same file whether it ran straight through or was
resumed. The file is written to a `.part` file and renamed, so a reader sees the old manifest or the new one."""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from indexer.crawl.cache import write_atomic
from indexer.crawl.fetch import Response

__all__ = ["ManifestDoc", "snapshots_dir", "write_manifest"]


@dataclass(frozen=True)
class ManifestDoc:
    url: str
    content_hash: str
    fetched: str  # ISO date
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, response: Response, metadata: dict[str, Any] | None = None) -> "ManifestDoc":
        return cls(response.url, response.content_hash, response.fetched, metadata or {})


def snapshots_dir(crawl_dir: str | Path) -> Path:
    """The snapshots folder beside the crawl folder (`.rag/crawl` and `.rag/snapshots`)."""
    return Path(crawl_dir).parent / "snapshots"


def write_manifest(generation: str, docs: Iterable[ManifestDoc], *, directory: str | Path) -> Path:
    """Write `<directory>/<generation>.json` and return its path. `generation` must be a plain file name."""
    if not generation or Path(generation).name != generation or generation in (".", ".."):
        raise ValueError(f"generation {generation!r} is not a file name")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    documents = [asdict(d) for d in sorted(docs, key=lambda d: d.url)]
    text = json.dumps({"generation": generation, "documents": documents}, indent=1, ensure_ascii=False, sort_keys=True)
    path = folder / f"{generation}.json"
    write_atomic(path, text.encode("utf-8") + b"\n")
    return path
