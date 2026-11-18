"""What a connector hands the indexer: pure data."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class CorruptStateError(RuntimeError):
    """A safety-net state file exists but cannot be read: the build refuses instead of reading it as empty and then
    overwriting it."""


@dataclass(frozen=True)
class PdfItem:
    """A PDF on disk for the indexer to extract; metadata is the connector's entry for it (url, download_url, titles,
    origin, section, doc_type, public_url)."""

    path: str
    url: str
    file_hash: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PageItem:
    """One document: its text and metadata (a web page, a record)."""

    url: str
    text: str
    metadata: Mapping[str, Any]


SourceItem = PdfItem | PageItem


@dataclass(frozen=True)
class SourceResult:
    items: Sequence[SourceItem]
    report: Mapping[str, Any] = field(
        default_factory=dict[str, Any]
    )  # counts, retained, list check: into the build reply
    expected_minimum: int = 0  # the per-source gate: PageItems, or PdfItems that produced a document
