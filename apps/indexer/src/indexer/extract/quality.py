"""
PDF text quality.

Page assessment of extracted PDF text and the repair of one extraction: OCR the pages
that have no text or only substituted glyph codes (through a function passed in), drop
what stays unreadable.
Standard library only (tests: tests/indexer/test_extract_quality.py).
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

QUALITY_VERSION = 1
ENCODED_SHARE = 0.5  # a page is "encoded" when this share of its characters are U+F000-U+F0FF
EMPTY_BELOW = 20  # fewer non-space characters: the page has no usable text (often only "2 / 142")
READABLE_SHARE = 0.6  # share of letters, digits and punctuation in readable text

_PAGE_MARK = re.compile(r"\[Page (\d+)\]\n")
_PUA_LO, _PUA_HI = 0xF000, 0xF0FF
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_PUNCTUATION = set(".,;:!?-–—/()[]{}%€$&+*=<>'\"#§@|_°²³…«»„“”‚‘’~^`\\")


def split_pages(text):
    """{page_number: text} from the cache format '[Page N]\\n...' (pages joined by a blank
    line); text without any page mark counts as page 1."""
    text = text or ""
    parts = _PAGE_MARK.split(text)
    if len(parts) == 1:
        return {1: text.strip()} if text.strip() else {}
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def join_pages(pages):
    """The cache format written by extract.pdf._page_docs_to_text."""
    return "\n\n".join(f"[Page {n}]\n{t}" for n, t in sorted(pages.items()) if t and t.strip())


def _visible(text):
    return [c for c in text or "" if not c.isspace()]


def classify_page(text):
    """'ok', 'encoded' (mostly substituted glyph codes) or 'empty' (no usable text)."""
    chars = _visible(text)
    if len(chars) < EMPTY_BELOW:
        return "empty"
    encoded = sum(1 for c in chars if _PUA_LO <= ord(c) <= _PUA_HI)
    return "encoded" if encoded >= ENCODED_SHARE * len(chars) else "ok"


def is_readable(text):
    """True when most visible characters are letters, digits or punctuation."""
    chars = _visible(text)
    if not chars:
        return False
    good = sum(1 for c in chars if c.isalnum() or c in _PUNCTUATION)
    return good >= READABLE_SHARE * len(chars)


def clean_ocr_text(text):
    """OCR output (Docling markdown) without HTML comments (Docling's picture placeholder)
    and without table padding: space runs become one space, dash runs '---'."""
    text = _COMMENT_RE.sub("", text or "")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"-{3,}", "---", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def repair_pages(pages, page_count, ocr_fn=None, ocr_budget=None, ocr_missing=True, known_empty=()):
    """Repair one PDF's pages.

    pages: {page_number: text} as extracted (pages without text may be missing);
    page_count: pages in the PDF; ocr_fn(page_numbers) -> {page_number: text};
    ocr_budget: at most this many pages go to OCR (None: no limit);
    ocr_missing: False for extractions that already ran OCR (Docling), whose unreadable
    pages are recorded as empty instead of being sent to OCR again;
    known_empty: pages an earlier run already found empty (never sent to OCR again).
    Docling's picture placeholder does not count as text when a page is assessed.
    Returns (readable pages, quality)."""
    quality = {
        "version": QUALITY_VERSION,
        "ocr_pages": [],
        "empty_pages": [],
        "pending_ocr_pages": [],
    }
    result, need_ocr, known_empty = {}, [], set(known_empty)
    for n in range(1, max(page_count, max(pages, default=0)) + 1):
        text = pages.get(n, "")
        if classify_page(_COMMENT_RE.sub("", text)) == "ok":
            result[n] = text
            continue
        if n in known_empty:
            quality["empty_pages"].append(n)
            continue
        need_ocr.append(n)
    if not ocr_missing:
        quality["empty_pages"] = sorted(quality["empty_pages"] + need_ocr)
        return result, quality
    if ocr_fn is None:
        quality["pending_ocr_pages"] = need_ocr
        return result, quality
    limit = len(need_ocr) if ocr_budget is None else max(0, ocr_budget)
    todo, quality["pending_ocr_pages"] = need_ocr[:limit], need_ocr[limit:]
    if not todo:
        return result, quality
    try:
        ocr = ocr_fn(todo) or {}
    except Exception as exc:
        logger.warning(f"[PdfTextQuality] OCR failed for pages {todo[:5]}...: {exc}")
        quality["pending_ocr_pages"] = todo + quality["pending_ocr_pages"]
        return result, quality
    for n in todo:
        text = clean_ocr_text(ocr.get(n))
        if classify_page(text) == "ok" and is_readable(text):
            result[n] = text
            quality["ocr_pages"].append(n)
        else:
            quality["empty_pages"].append(n)
    quality["empty_pages"].sort()
    return result, quality


def repair_entry(entry, page_count, ocr_fn=None, ocr_budget=None, before_ocr=None):
    """A cache entry (dict) with its text rebuilt from readable pages and a 'quality' block.
    Missing pages go to OCR only for fast-reader entries ('pypdfium2...') and near-empty
    entries. before_ocr(pages) runs just before OCR (the pipeline writes a crash marker);
    pages of an earlier OCR attempt that never finished are recorded as empty, not retried.
    Returns (new_entry, pages_sent_to_ocr), counting pages sent even when OCR fails."""
    method = entry.get("extraction_method") or ""
    pages = split_pages(entry.get("text") or "")
    near_empty = sum(len(_visible(_COMMENT_RE.sub("", t))) for t in pages.values()) < 200
    ocr_missing = method.startswith("pypdfium2") or near_empty
    previous = entry.get("quality")
    known_empty = []
    if isinstance(previous, dict) and previous.get("version") == QUALITY_VERSION:
        known_empty = list(previous.get("empty_pages") or [])
        if previous.get("ocr_attempt"):
            logger.warning(
                f"[PdfTextQuality] OCR of pages {previous['ocr_attempt'][:5]}... did not finish "
                f"in an earlier run; recorded as empty"
            )
            known_empty += previous["ocr_attempt"]
    sent_pages = []

    def counted(numbers):
        sent_pages.extend(numbers)
        if before_ocr:
            before_ocr(numbers)
        assert ocr_fn is not None  # counted is passed only with an ocr_fn
        return ocr_fn(numbers)

    repaired, quality = repair_pages(
        pages, page_count, counted if ocr_fn else None, ocr_budget, ocr_missing, known_empty
    )
    new = dict(entry)
    new["text"] = join_pages(repaired)
    new["quality"] = quality
    if quality["ocr_pages"] and not method.endswith("+repair"):
        new["extraction_method"] = method + "+repair"
    return new, len(sent_pages)


def ocr_attempt_marker(entry, pages):
    """The entry as it is, marked 'OCR of these pages started': written before OCR, so a run
    killed inside OCR (out of memory, crash) leaves a mark and the next run does not retry."""
    marker = dict(entry)
    marker["quality"] = dict(entry.get("quality") or {}, version=QUALITY_VERSION, ocr_attempt=list(pages))
    return marker


def write_json_atomic(path, data):
    """JSON to path through a temporary file and os.replace: a failed or interrupted write never
    leaves a truncated cache file (the build reads every cache file)."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def quality_is_current(quality):
    """True when an entry was assessed with this version and nothing waits for OCR."""
    return (
        isinstance(quality, dict)
        and quality.get("version") == QUALITY_VERSION
        and not quality.get("pending_ocr_pages")
        and not quality.get("ocr_attempt")
    )


def clean_docling_markup(text):
    """Cache text without Docling's markup artefacts:
    HTML comments (the picture placeholder) removed; in markdown table rows, runs of three or
    more spaces become one and runs of four or more dashes '---'; three or more newlines
    become two, and no blank line stays right after a page mark (a removed placeholder would
    otherwise cut the mark off from its page in chunking). Runs of one or two spaces stay. Idempotent; the result is not stripped."""
    text = _COMMENT_RE.sub("", text or "")
    lines = []
    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            line = re.sub(r" {3,}", " ", line)
            line = re.sub(r"-{4,}", "---", line)
        lines.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return re.sub(r"(\[Page \d+\]\n)\n+", r"\1", text)
