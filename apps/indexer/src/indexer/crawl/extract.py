"""Extraction of a fetched HTML page: Markdown documents (one per section of the page) and the PDFs it links to.

The main content is the first element matching `Selectors.main`; elements matching `Selectors.drop` (navigation, header,
footer, scripts, ...) are removed from it before anything is read, as are elements hidden from readers (`hidden`,
`aria-hidden="true"`, an inline style with `display:none` or `visibility:hidden`), so that invisible text is never indexed. Each outermost element matching `Selectors.section`
becomes one `PageDocument` with the section's id and heading, so a hub page with several regulation sections gives one
document per section; text outside any section is a document of its own (first), unless it is only headings. Without
sections the whole main content is one document. The Markdown keeps headings (`h1`-`h6` as `#`..`######`), paragraphs,
lists, tables, emphasis and links (absolute URLs).

PDFs are the anchors in the main content whose `href` path ends in `.pdf` (any case; query, fragment and `;` path parameters ignored), as
absolute normalised URLs in page order, each once, with its link text, `title` attribute and the id of its section. Which
hosts may be fetched stays with the fetcher.

A page whose main selector matches nothing, or only elements without text, comes back with `problem` set (for the build
summary) and no documents. Selectors are a small CSS subset: `tag`, `.class`, `#id` compounds, descendant (space) and child
(`>`) combinators and `,` alternatives. The bytes are decoded in the charset of the HTTP `Content-Type`, else of the page's
`<meta>`, else as UTF-8 (Windows-1252 when that fails). HTML is parsed without network access or entity resolution."""

import codecs
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import lxml.etree as etree
from lxml import html

from indexer.crawl.discover import normalise

__all__ = ["Link", "PageDocument", "Extraction", "Selectors", "extract"]

DEFAULT_DROP = "script, style, noscript, template, svg, iframe, nav, header, footer, aside, form"
LOWER = "translate({}, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ \t\r\n', 'abcdefghijklmnopqrstuvwxyz')"
HIDDEN = (
    f"[@hidden or {LOWER.format('@aria-hidden')}='true' or "
    f"contains({LOWER.format('@style')}, 'display:none') or contains({LOWER.format('@style')}, 'visibility:hidden')]"
)
HTTP_CHARSET = re.compile(r"charset\s*=\s*[\"']?([\w.:-]+)", re.IGNORECASE)
META_CHARSET = re.compile(r"<meta[^>]{0,300}?charset\s*=\s*[\"']?([\w.:-]+)", re.IGNORECASE)
PATH_PARAMETERS = re.compile(r";[^/]*")
XML_DECLARATION = re.compile(r"\A\s*<\?xml[^>]*\?>")
HTML = html.HTMLParser(no_network=True, remove_comments=True, remove_pis=True)
HEADINGS = {f"h{n}": n for n in range(1, 7)}
BLOCKS = frozenset(
    "p div section article main figure figcaption blockquote address dl dt dd fieldset details summary hr body".split()
)
LINE_BREAK = "\x00"
COMPOUND = re.compile(r"([A-Za-z][\w-]*|\*)?((?:[.#][\w-]+)*)")
SIMPLE = re.compile(r"[.#][\w-]+")


def _xpath(selector: str) -> str:
    """The XPath (relative to a context element's descendants) of a selector in the supported CSS subset."""
    alternatives = []
    for alternative in selector.split(","):
        path, axis = ".", "//"
        tokens = re.findall(r">|[^\s>]+", alternative)
        for token in tokens:
            if token == ">":
                axis = "/"
                continue
            match = COMPOUND.fullmatch(token)
            if not match or (not match.group(1) and not match.group(2)):
                raise ValueError(f"unsupported selector {selector!r} (at {token!r})")
            tests = [
                f"contains(concat(' ', normalize-space(@class), ' '), ' {part[1:]} ')"
                if part[0] == "."
                else f"@id='{part[1:]}'"
                for part in SIMPLE.findall(match.group(2))
            ]
            path += f"{axis}{match.group(1) or '*'}" + (f"[{' and '.join(tests)}]" if tests else "")
            axis = "//"
        if not tokens or tokens[-1] == ">":
            raise ValueError(f"unsupported selector {selector!r}")
        alternatives.append(path)
    return " | ".join(alternatives)


@dataclass(frozen=True)
class Selectors:
    """Where the content is on a host's pages: the main content, the sections it splits into and the chrome to drop."""

    main: str = ".main-column"
    section: str = "section.content-section"
    drop: str = DEFAULT_DROP

    def __post_init__(self) -> None:
        for selector in (self.main, self.section, self.drop):
            _xpath(selector)


@dataclass(frozen=True)
class Link:
    """An anchor: absolute `url`, its text, its `title` attribute ('' if none) and the id of the section it is in."""

    url: str
    text: str
    title: str
    section_id: str | None


@dataclass(frozen=True)
class PageDocument:
    """One document of a page: the section's id (None outside an id-carrying section), its first heading and its Markdown."""

    section_id: str | None
    heading: str | None
    markdown: str


@dataclass(frozen=True)
class Extraction:
    """What a page yields: its title, its documents, the PDFs it links to and, when there is nothing to index, why."""

    url: str
    title: str
    documents: tuple[PageDocument, ...]
    pdfs: tuple[Link, ...]
    problem: str | None = None


def _text(element: etree._Element) -> str:
    return " ".join(element.text_content().split()) if isinstance(element, html.HtmlElement) else ""


def _squash(text: str) -> str:
    lines = " ".join(text.split()).split(LINE_BREAK)
    return "\n".join(line.strip() for line in lines if line.strip())


def _absolute(base: str, href: str) -> str:
    try:
        url = normalise(urljoin(base, href.strip()))
    except ValueError:
        return ""
    return url if urlsplit(url).scheme in ("http", "https") else ""


def _markup(element: html.HtmlElement, base: str) -> str:
    """An inline element as Markdown: its content, wrapped for links, emphasis and code."""
    inner = _inline(element, base)
    if element.tag == "br":
        return LINE_BREAK
    if not inner.strip():
        return inner
    if element.tag == "a":
        url = _absolute(base, element.get("href", ""))
        return f"[{' '.join(inner.split())}]({url})" if url else inner
    if element.tag in ("strong", "b"):
        return f"**{inner.strip()}**"
    if element.tag in ("em", "i"):
        return f"*{inner.strip()}*"
    if element.tag == "code":
        return f"`{inner.strip()}`"
    return inner


def _inline(element: html.HtmlElement, base: str) -> str:
    """The content of an element with its inline markup as Markdown (whitespace not yet collapsed)."""
    parts = [element.text or ""]
    for child in element:
        if isinstance(child.tag, str):
            parts.append(_markup(child, base))
        parts.append(child.tail or "")
    return "".join(parts)


def _indent(block: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else line for line in block.split("\n"))


def _list(element: html.HtmlElement, base: str) -> str:
    items = []
    for number, item in enumerate((c for c in element if c.tag == "li"), start=1):
        parts: list[str] = []
        _blocks(item, base, parts)
        marker = f"{number}. " if element.tag == "ol" else "- "
        if parts:
            items.append(marker + parts[0] + "".join("\n" + _indent(p, "  ") for p in parts[1:]))
    return "\n".join(items)


def _table(element: html.HtmlElement, base: str) -> str:
    rows = [
        "| "
        + " | ".join(_squash(_inline(cell, base)).replace("\n", " ") for cell in row if cell.tag in ("td", "th"))
        + " |"
        for row in element.iter("tr")
    ]
    columns = rows[0].count("|") - 1 if rows else 0
    return "\n".join(rows[:1] + ["|" + " --- |" * columns] + rows[1:]) if columns > 0 else ""


def _blocks(element: html.HtmlElement, base: str, out: list[str]) -> None:
    """Append the blocks (paragraphs, headings, lists, tables) of an element's content to `out`."""
    pending = [element.text or ""]

    def flush() -> None:
        text = _squash("".join(pending))
        if text:
            out.append(text)
        pending.clear()

    for child in element:
        tag = child.tag if isinstance(child.tag, str) else ""
        if tag in HEADINGS:
            flush()
            if heading := _squash(_inline(child, base)).replace("\n", " "):
                out.append("#" * HEADINGS[tag] + " " + heading)
        elif tag in ("ul", "ol"):
            flush()
            out.append(_list(child, base))
        elif tag == "table":
            flush()
            out.append(_table(child, base))
        elif tag == "pre":
            flush()
            out.append("```\n" + child.text_content().strip("\n") + "\n```")
        elif tag in BLOCKS or tag == "li":
            flush()
            _blocks(child, base, out)
        elif tag:
            pending.append(_markup(child, base))
        pending.append(child.tail or "")
    flush()
    out[:] = [block for block in out if block]


def _document(section_id: str | None, blocks: list[str]) -> PageDocument:
    heading = next((b.lstrip("#").strip() for b in blocks if b.startswith("#")), None)
    return PageDocument(section_id, heading, "\n\n".join(blocks))


def _pdfs(main: html.HtmlElement, base: str, sections: list[html.HtmlElement]) -> tuple[Link, ...]:
    found: dict[str, Link] = {}
    outer = set(sections)
    for anchor in main.iter("a"):
        url = _absolute(base, anchor.get("href", ""))
        if url and PATH_PARAMETERS.sub("", urlsplit(url).path).lower().endswith(".pdf") and url not in found:
            title = " ".join(anchor.get("title", "").split())
            section = next((a for a in anchor.iterancestors() if a in outer), None)
            found[url] = Link(url, _text(anchor), title, section.get("id") if section is not None else None)
    return tuple(found.values())


def _decode(body: bytes, content_type: str) -> str:
    """The page text: in the charset of the HTTP `Content-Type`, else of the page's `<meta>`, else UTF-8, else (when the
    bytes are not UTF-8) Windows-1252. An unknown charset name counts as none."""
    for pattern, source in ((HTTP_CHARSET, content_type), (META_CHARSET, body[:4096].decode("ascii", "ignore"))):
        if found := pattern.search(source):
            name = found.group(1)
            try:
                return body.decode(codecs.lookup(name).name, "replace")
            except LookupError:
                continue
    try:
        return body.decode("utf-8-sig")
    except UnicodeDecodeError:
        return body.decode("cp1252", "replace")


def extract(body: bytes, url: str, selectors: Selectors = Selectors(), content_type: str = "") -> Extraction:
    """The documents and PDF links of the page `body` fetched from `url` (after redirects) with the response's
    `Content-Type` header (its charset is used first)."""
    try:
        tree = html.document_fromstring(XML_DECLARATION.sub("", _decode(body, content_type)), parser=HTML)
    except (ValueError, etree.ParserError):
        return Extraction(url, "", (), (), "page is not parseable HTML")
    title = " ".join((tree.findtext(".//title") or "").split())
    found = [m for m in tree.xpath(_xpath(selectors.main)) if not m.xpath(f"ancestor-or-self::*{HIDDEN}")]
    if not found:
        return Extraction(url, title, (), (), f"main content selector {selectors.main!r} matched nothing")
    main: html.HtmlElement = found[0]
    for element in main.xpath(f"{_xpath(selectors.drop)} | .//*{HIDDEN}"):  # a union lists each element once
        element.drop_tree()
    matched = main.xpath(_xpath(selectors.section))
    sections = [s for s in matched if not any(a in matched for a in s.iterancestors())]
    pdfs = _pdfs(main, url, sections)
    documents: list[PageDocument] = []
    for section in sections:
        blocks: list[str] = []
        _blocks(section, url, blocks)
        if blocks:
            documents.append(_document(section.get("id"), blocks))
        section.drop_tree()
    rest: list[str] = []
    _blocks(main, url, rest)
    if rest and (not documents or not all(b.startswith("#") for b in rest)):
        documents.insert(0, _document(None, rest))
    if not documents:
        return Extraction(url, title, (), (), f"main content selector {selectors.main!r} matched no text")
    return Extraction(url, title, tuple(documents), pdfs)
