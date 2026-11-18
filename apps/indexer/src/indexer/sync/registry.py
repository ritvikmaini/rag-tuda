"""The registry of downloaded PDFs: which hash each URL was last fetched under (the list's own, else the file's), and the
names and hashes of the files in the PDF folder."""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rag_core.ports.sources import StateStore

REGISTRY = "pdf_downloaded_urls.json"  # in the state store


@dataclass(frozen=True)
class PdfEntry:
    """One PDF of a source's list: its download URL, its title, the hash the list publishes for it (None when it
    publishes none) and whatever else the connector keeps with it."""

    url: str
    title: str
    sha256: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def local_path(url: str, pdf_dir: str | Path) -> Path:
    """Where the PDF of a URL lives: its cleaned file name plus a digest of the whole URL, so equal names never collide."""
    raw_name = Path(unquote(urlparse(url).path)).name or "document.pdf"
    cleaned = re.sub(r"\s+", "_", re.sub(r"[^\w.\- ]+", "_", raw_name, flags=re.ASCII).strip())
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return Path(pdf_dir) / f"{Path(cleaned).stem or 'document'}__{digest}.pdf"


def drop_cache_entry(sha256: str | None, cache_dir: str | Path) -> None:
    """Delete the extraction cache entry of a file that no longer exists in that form."""
    if sha256:
        (Path(cache_dir) / f"{sha256}.json").unlink(missing_ok=True)


def load(state: StateStore) -> dict[str, str]:
    """The registry; empty on the first run. CorruptStateError when the file cannot be read."""
    registry = state.load(REGISTRY)
    if registry is None:
        return {}
    logging.info(f"Downloaded URLs file loaded: {state.path(REGISTRY)} ({len(registry)} entries)")
    return registry


def save(state: StateStore, registry: dict[str, str]) -> None:
    """A write error is logged, not raised: the downloads the registry records have happened."""
    try:
        state.save(REGISTRY, registry)
        logging.info(f"Downloaded URLs saved locally: {state.path(REGISTRY)}")
    except OSError as e:
        logging.error(f"Error saving downloaded URLs to local file: {e}")
