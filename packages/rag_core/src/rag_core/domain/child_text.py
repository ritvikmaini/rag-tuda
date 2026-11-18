"""The child's own text: children are indexed with a prefix, '<title>: ' or the context header stored in the parent's
metadata['context_header'], that the citation quote and the window anchor must not show."""

from collections.abc import Mapping
from typing import Any


def child_body(metadata: Mapping[str, Any] | None, text: str | None) -> str:
    """text without the parent's context_header, else without '<title>: ', else unchanged."""
    meta = metadata or {}
    text = text or ""
    header = str(meta.get("context_header") or "")
    if header and text.startswith(header):
        return text[len(header) :]
    title = str(meta.get("title") or "")
    if title and text.startswith(title + ": "):
        return text[len(title) + 2 :]
    return text
