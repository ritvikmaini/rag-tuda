"""The page type between the extractors and chunking."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageDoc:
    """One document page as the extractors and sources hand it to chunking."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
