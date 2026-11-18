"""Page documents of the PDFs: titles, pages and their metadata."""

import json
import os
import re
from collections import Counter
from collections.abc import Sequence

from indexer.extract.pdf import EXTRACTED_DIR
from indexer.extract.quality import clean_docling_markup
from rag_core.domain.sources import PdfItem
from rag_core.domain.text import clean_text, extract_year

PAGE_MARKER_LINE = re.compile(r"^\[Page (\d+)\]\n", re.M)

MIN_PDF_PAGE_TEXT = 20


def split_pages(text):
    """[(page_number, cleaned_text)] from extraction-cache text with '[Page N]' marker lines."""
    parts = PAGE_MARKER_LINE.split(text or "")
    pages = []
    if clean_text(parts[0], tags=False):
        pages.append((1, parts[0]))
    for i in range(1, len(parts) - 1, 2):
        pages.append((int(parts[i]), parts[i + 1]))
    out = []
    for number, page_text in pages:
        cleaned = clean_text(page_text, tags=False)
        if len(cleaned) >= MIN_PDF_PAGE_TEXT:
            out.append((number, cleaned))
    return out


SLUG_LIKE = re.compile(r"(\.(pdf|docx?|xlsx?|pptx?)\)?$)|^microsoft (word|excel)|^untitled|^dokument\d*$", re.I)
TRAILING_YMD = re.compile(r" (\d{4}) (\d{2}) (\d{2})$")
TRAILING_DMY = re.compile(r" (\d{2}) (\d{2}) (\d{4})$")
FILE_EXTENSION = re.compile(r"\.(pdf|docx?|xlsx?|pptx?)$", re.I)


def clean_download_title(title):
    """'Mustervereinbarung 2001 03 15.pdf' -> 'Mustervereinbarung (15.03.2001)'."""
    t = " ".join((title or "").split())
    t = TRAILING_YMD.sub(lambda m: " (%s.%s.%s)" % (m.group(3), m.group(2), m.group(1)), t)
    t = TRAILING_DMY.sub(lambda m: " (%s.%s.%s)" % (m.group(1), m.group(2), m.group(3)), t)
    return FILE_EXTENSION.sub("", t)


def resolve_title(download_title, pdf_meta_title="", first_page_text="", generic_titles=()):
    """Best human title: the PDF's own title if it looks real and is not shared by
    several PDFs (generic_titles, e.g. a letterhead's "Ihr Zeichen"), else the cleaned
    manifest title, else the first heading-like line of page 1."""
    meta = " ".join((pdf_meta_title or "").split())
    if (
        8 <= len(meta) <= 150
        and not SLUG_LIKE.search(meta)
        and re.search(r"[a-zäöü]{3}", meta)
        and meta not in generic_titles
    ):
        return meta
    cleaned = clean_download_title(download_title)
    if cleaned:
        return cleaned
    for line in (first_page_text or "").splitlines()[:8]:
        line = " ".join(line.strip("#* ").split())
        if 8 <= len(line) <= 120 and re.search(r"[A-Za-zÄÖÜäöü]{4}", line) and not line.lower().startswith("seite"):
            return line
    return "Dokument"


def pdf_page_documents(entry, cache, title):
    """One document per non-empty page of a PDF from its extraction-cache entry."""
    docs = []
    year = extract_year(entry.get("download_url", ""), entry.get("download_title", ""), title)
    for number, text in split_pages(cache.get("text", "")):
        docs.append(
            {
                "page_content": text,
                "metadata": {
                    "title": title,
                    "url": entry.get("download_url"),
                    "download_url": entry.get("download_url"),
                    "page_url": entry.get("url"),
                    "public_url": entry.get("public_url"),
                    "source_type": "pdf",
                    "doc_type": entry.get("doc_type", "Dokument"),
                    "origin": entry.get("origin", "web"),
                    "section": entry.get("section", ""),
                    "doc_year": year,
                    "doc_date": None,
                    "page_number": number,
                    "doc_key": entry.get("download_url"),
                    "file_hash": cache.get("file_hash"),
                    "extraction_method": cache.get("extraction_method"),
                },
            }
        )
    return docs


def _pdf_meta_title(path):
    try:
        import pypdfium2 as pdfium

        return (pdfium.PdfDocument(path).get_metadata_dict() or {}).get("Title", "") or ""
    except Exception:
        return ""


def pdf_documents(items: Sequence[PdfItem], seen_hashes, extracted_dir=EXTRACTED_DIR, clean=True):
    """Page documents for PDF items from the extraction cache, deduplicated by file hash across calls (the first
    item with a hash wins: items are processed in list order). clean: Docling markup cleanup (no
    picture placeholders, no table padding)."""
    # A PDF metadata title shared by several PDFs is a template's, not the document's
    meta_titles = {item.path: _pdf_meta_title(item.path) for item in items if item.file_hash not in seen_hashes}
    shared = Counter(" ".join(t.split()) for t in meta_titles.values() if t.strip())
    generic_titles = {t for t, n in shared.items() if n > 1}
    docs, missing = [], 0
    for item in items:
        if item.file_hash in seen_hashes:
            continue
        cache_path = os.path.join(extracted_dir, item.file_hash + ".json")
        if not os.path.exists(cache_path):
            missing += 1
            continue
        seen_hashes.add(item.file_hash)
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
        if clean:
            cache["text"] = clean_docling_markup(cache.get("text", ""))
        entry = item.metadata
        pages = split_pages(cache.get("text", ""))
        first_page = pages[0][1] if pages else ""
        if entry.get("origin") == "website" and entry.get("download_title"):
            title = entry["download_title"]
        else:
            title = resolve_title(
                entry.get("download_title", ""),
                meta_titles.get(item.path, ""),
                first_page,
                generic_titles=generic_titles,
            )
        docs.extend(pdf_page_documents(entry, cache, title))
    return docs, missing
