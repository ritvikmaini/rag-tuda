"""Text rules shared by the connectors and the extraction."""

import html
import re

YEAR_PATTERN = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
HTML_TAG = re.compile(r"<[^>]+>")
MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str | None, tags: bool = True) -> str:
    """Remove Docling placeholders/HTML comments, unescape entities, tidy blank lines; tags=True
    (HTML sources: crawl pages, record fields) also replaces HTML tags by a space. PDF text
    keeps its angle brackets ("< Naturschutz >", "< 3.300 Kelvin)")."""
    text = HTML_COMMENT.sub("", text or "")
    if tags:
        text = HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return MANY_NEWLINES.sub("\n\n", text).strip()


def extract_year(*texts: str | None) -> int | None:
    """The last plausible year (1950-2049) in the first text that has one, else None."""
    for text in texts:
        years = YEAR_PATTERN.findall(text or "")
        if years:
            return int(years[-1])
    return None
