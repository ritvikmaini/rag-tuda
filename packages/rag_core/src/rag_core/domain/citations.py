# pyright: standard
"""Citation map for the page (clickable citations).

entries(sources, docs) turns the numbered sources of one answer (the answer step's context_sources: n, title, origin, type,
date, url, page, section_id) into one entry per number, so the page can link every [n] of the answer to its document and
page. The quote is the passage that matched the search (matched_child_text of the retrieved parent with the same
section_id). The SSE frame 'data: {"type": "citations", "content": [...]}' made of them is assistant_api.sse.events'; it is
sent once after the answer."""

import re

from rag_core.domain.child_text import child_body
from rag_core.domain.source_meta import strip_source_text

QUOTE_CHARS = 200
WHITESPACE = re.compile(r"\s+")
HEADING_MARKER = re.compile(r"^[ \t]*#+[ \t]+", re.MULTILINE)  # a Markdown heading's '## ' at the start of a line


def _metadata(doc):
    meta = doc.get("metadata") if isinstance(doc, dict) else getattr(doc, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def _cut(text):
    if len(text) <= QUOTE_CHARS:
        return text
    cut = text[:QUOTE_CHARS]
    space = cut.rfind(" ")
    if space > QUOTE_CHARS // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"


def _quotes(docs):
    """{section_id: quote} for the IDs that occur exactly once among the retrieved parents."""
    by_section = {}
    for doc in docs or []:
        meta = _metadata(doc)
        section_id = meta.get("section_id")
        if isinstance(section_id, str) and section_id:
            by_section.setdefault(section_id, []).append(meta)
    quotes = {}
    for section_id, metas in by_section.items():
        if len(metas) != 1:
            continue  # the degraded path (parent lookup failed) can repeat IDs
        text = child_body(metas[0], str(metas[0].get("matched_child_text") or ""))  # no '<title>: ' or context header
        text = WHITESPACE.sub(" ", HEADING_MARKER.sub("", strip_source_text(text))).strip()
        if text:
            quotes[section_id] = _cut(text)
    return quotes


def _page(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def entries(sources, docs):
    quotes = _quotes(docs)
    out, seen = [], set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        n = source.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1 or n in seen:
            continue
        seen.add(n)
        section_id = source.get("section_id")
        out.append(
            {
                "n": n,
                "url": str(source.get("url") or ""),
                "page": _page(source.get("page")),
                "title": str(source.get("title") or ""),
                "type": str(source.get("type") or ""),
                "origin": str(source.get("origin") or ""),
                "date": str(source.get("date") or ""),
                "quote": quotes.get(section_id, "") if isinstance(section_id, str) else "",
            }
        )
    out.sort(key=lambda entry: entry["n"])
    return out
