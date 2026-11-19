# pyright: standard
"""
Source description for prompt headers and citations. Works with the
corpus metadata (title, origin, source_type, doc_type, doc_year, doc_date, url,
page_number). Standard library only.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date as _date
from urllib.parse import unquote, urlparse

YEAR_PATTERN = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")
PAGE_MARKER = re.compile(r"\[Page \d+\]\s*")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


@dataclass(frozen=True)
class SourceLabels:
    """Where a source comes from, as its header and its citation name it, and the words that frame the sources in the prompt: the
    profile's messages, labels.* (one set per language)."""

    source_types: Mapping[str, str]  # metadata source_type -> label (checked first)
    origins: Mapping[str, str]  # metadata origin -> label
    hosts: Mapping[str, str]  # the url's host -> label, for a source without a known origin
    scope: str = ""  # the label of the header's scope line; "" prints no line
    sources_lead: str = field(kw_only=True)  # the line before the sources, {n}: how many
    nothing_found: str = field(kw_only=True)  # the sources of a search that found nothing
    date: str = field(kw_only=True)  # the header's date field reads "<date>: 2021"
    page: str = field(kw_only=True)  # and its page field "<page> 4"
    link: str = field(kw_only=True)  # the line before a source's url reads "<link>: https://..."


def strip_source_text(text):
    """Remove page markers and Docling placeholders."""
    text = HTML_COMMENT.sub("", text or "")
    return PAGE_MARKER.sub("", text).strip()


def _origin(metadata, url, labels):
    """The label of a source's type, else of its origin, else of its url's host; "" when none fits."""
    kind = metadata.get("source_type")
    if kind in labels.source_types:
        return labels.source_types[kind]
    origin = metadata.get("origin")
    if origin in labels.origins:
        return labels.origins[origin]
    return labels.hosts.get(urlparse(url or "").netloc, "")


def _german_date(iso):
    try:
        y, m, d = iso.split("-")
        return "%s.%s.%s" % (d, m, y)
    except (ValueError, AttributeError):
        return ""


def describe_source(metadata, labels):
    """{"title", "origin", "type", "date", "url"} for a parent document's metadata; empty strings when unknown.
    origin: the profile's label (labels.*)."""
    m = metadata or {}
    url = m.get("download_url") or m.get("url") or ""
    origin = _origin(m, url, labels)
    title = m.get("title") or "Dokument"
    doc_type = m.get("doc_type") or ""
    if m.get("doc_date"):
        date = _german_date(m["doc_date"])
    elif m.get("news_date"):
        date = m["news_date"]
    else:
        max_year = _date.today().year + 1
        year = m.get("doc_year")
        if not year:
            # decoded URL ("%2045" is no year, a fragment's "11. Juli 2026" is), no year after next year
            found = [int(y) for y in YEAR_PATTERN.findall(unquote(url) + " " + title) if int(y) <= max_year]
            year = found[-1] if found else None
        elif str(year).isdigit() and int(year) > max_year:
            year = None
        date = str(year) if year else ""
    return {"title": title, "origin": origin, "type": doc_type, "date": date, "url": url}


CITATION = re.compile(r"\[(\d{1,2})\]")


def select_sources_for_display(sources, answer, mode):
    """Sources to send to the page after an answer: all the model saw ("seen"), or only
    those cited as [n] ("cited", falling back to all seen when the answer cites none)."""
    if mode != "cited":
        return list(sources)
    cited = {int(n) for n in CITATION.findall(answer or "")}
    chosen = [s for s in sources if s.get("n") in cited]
    return chosen or list(sources)
