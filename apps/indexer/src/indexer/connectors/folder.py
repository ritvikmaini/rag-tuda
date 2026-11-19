"""The folder source: every Markdown (.md), text (.txt) and PDF (.pdf) file under the overlay's connectors.<source>.dir is
one document, read in path order.

A page is titled by the first "# " line of its text (else the file name without its suffix). It may open with a front matter
block ("---" lines around YAML) whose keys doc_type, date (YYYY-MM-DD) and url override the defaults; the block itself is not
indexed. A PDF is handed to the indexer's extraction as it is.

url: the front matter url, else the profile entry's base_url plus the relative path without its suffix (each part
percent-encoded), else "folder:<relative path>" (never an absolute path: citations must not carry where the folder lives).
Two files with one url are refused, naming both. A symlink to a file outside the folder is skipped.
origin: "website" for a folder that copies a site (the entry has a base_url), else "documents" (the user's own files).
doc_type: the front matter, else the profile entry's doc_type, else "Dokument".
The profile entry's other keys: base_url, doc_type, min_documents (the source gate)."""

import datetime
import urllib.parse
from pathlib import Path
from typing import Any

import yaml
from rag_core.domain.sources import PageItem, PdfItem, SourceResult
from rag_core.ports.sources import SourceContext

from indexer.sync.registry import file_hash

SUFFIXES = (".md", ".txt", ".pdf")
FENCE = "---"


def split_front_matter(text: str, name: str) -> tuple[dict[str, Any], str]:
    """The front matter block as a dict and the text after it; ({}, text) without a block."""
    lines = text.split("\n")
    if lines[0].rstrip() != FENCE:
        return {}, text
    for end in range(1, len(lines)):
        if lines[end].rstrip() == FENCE:
            try:
                meta = yaml.safe_load("\n".join(lines[1:end])) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"{name}: the front matter is not YAML ({exc})") from exc
            if not isinstance(meta, dict):
                raise ValueError(f"{name}: the front matter is not a mapping")
            return meta, "\n".join(lines[end + 1 :]).lstrip("\n")
    raise ValueError(f"{name}: the front matter block is not closed")


def title_of(text: str, fallback: str) -> str:
    for line in text.split("\n"):
        if line.startswith("# ") and line[2:].strip():
            return line[2:].strip()
    return fallback


def day(value: Any, name: str) -> str | None:
    """The front matter date as an ISO day; None when the file has none."""
    if value is None:
        return None
    try:
        return datetime.date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name}: date must be YYYY-MM-DD, not {value!r}") from exc


class Folder:
    name = "folder"
    state_files: tuple[str, ...] = ()

    def collect(self, ctx: SourceContext) -> SourceResult:
        if ctx.data_dir is None:
            raise RuntimeError("folder: the overlay names no directory (connectors.<source>.dir)")
        root = ctx.data_dir
        real_root = root.resolve()
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUFFIXES and p.resolve().is_relative_to(real_root)
        )
        files = files[: ctx.limit or None]
        items = [self.item(path, path.relative_to(root), ctx.settings) for path in files]
        seen: dict[str, Path] = {}
        for path, item in zip(files, items):
            if item.url in seen:
                raise ValueError(
                    f"{seen[item.url].relative_to(root).as_posix()} and {path.relative_to(root).as_posix()} "
                    f"both have the url {item.url}"
                )
            seen[item.url] = path
        return SourceResult(
            items,
            {"files": len(items)},
            expected_minimum=int(ctx.settings.get("min_documents", 0)),
        )

    def item(self, path: Path, rel: Path, settings: Any) -> PageItem | PdfItem:
        name = rel.as_posix()
        meta: dict[str, Any] = {}
        text = ""
        if path.suffix.lower() != ".pdf":
            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{name}: not UTF-8 ({exc.reason} at byte {exc.start})") from exc
            meta, text = split_front_matter(raw, name)
        base_url = settings.get("base_url")
        origin = "website" if base_url else "documents"
        url = meta.get("url") or (
            f"{base_url.rstrip('/')}/{urllib.parse.quote(rel.with_suffix('').as_posix())}"
            if base_url
            else f"folder:{name}"
        )
        doc_type = meta.get("doc_type") or settings.get("doc_type") or "Dokument"
        if path.suffix.lower() == ".pdf":
            return PdfItem(
                str(path),
                url,
                file_hash(path),
                {"url": url, "download_url": url, "doc_type": doc_type, "origin": origin},
            )
        date = day(meta.get("date"), name)
        return PageItem(
            url,
            text,
            {
                "title": title_of(text, rel.stem),
                "url": url,
                "source_type": "page",
                "doc_type": doc_type,
                "origin": origin,
                "section": "",
                "doc_year": int(date[:4]) if date else None,
                "doc_date": date,
                "page_number": -1,  # not a PDF page, as for a crawled web page
                "doc_key": url,
            },
        )
