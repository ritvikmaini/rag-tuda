import json
import logging
import os
from datetime import datetime
from typing import Any, cast

from rag_core.settings.profile import Extraction

from indexer.documents import PageDoc
from indexer.extract.ocr import ocr_session
from indexer.extract.quality import (
    ocr_attempt_marker,
    quality_is_current,
    repair_entry,
    write_json_atomic,
)


class PageLimitExceededError(RuntimeError):
    """Raised when a PDF has more pages than MAX_OCR_PAGES allows for full Docling OCR."""

    def __init__(self, pdf_url: str, page_count: int, max_pages: int):
        self.pdf_url = pdf_url
        self.page_count = page_count
        self.max_pages = max_pages
        super().__init__(
            f"PDF has {page_count} pages, exceeding MAX_OCR_PAGES={max_pages}; "
            f"skipping full Docling OCR to avoid OOM: {pdf_url}"
        )


def _extract_text_with_pypdfium2(pdf_path: str) -> list[PageDoc]:
    """Fast per-page text extraction using pypdfium2 (no ML models). Works for born-digital PDFs."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    documents = []
    for i in range(len(pdf)):
        page = pdf[i]
        text = page.get_textpage().get_text_range()
        if text and text.strip():
            documents.append(
                PageDoc(
                    page_content=text.strip(),
                    metadata={"source": pdf_path, "page_number": i + 1},
                )
            )
    pdf.close()
    return documents


def _load_pdf_with_docling_full(pdf_url: str, max_ocr_pages: int) -> list[PageDoc]:
    """
    Full Docling pipeline with layout model + EasyOCR (best quality, slow on CPU).
    Returns one PageDoc per page with page_number in metadata, so chunks carry page references.
    """
    from collections import defaultdict
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        EasyOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Guard against pathologically large documents: Docling's full pipeline
    # (layout + table structure + EasyOCR) buffers per-page results in memory
    # for the whole document until conversion finishes, so a document with an
    # excessive page count can exceed the container's memory limit and get
    # SIGKILL'd (exit 137), taking down the whole service. Skip such files
    # instead so the rest of the batch can keep processing.
    if max_ocr_pages:
        try:
            import pypdfium2 as pdfium

            page_count = len(pdfium.PdfDocument(pdf_url))
        except Exception as exc:
            logging.warning(f"Could not determine page count for {pdf_url}: {exc}")
            page_count = None

        if page_count is not None and page_count > max_ocr_pages:
            raise PageLimitExceededError(pdf_url, page_count, max_ocr_pages)
    device = "cpu"  # the indexer never gets a GPU

    ocr_options = EasyOcrOptions(
        lang=["de", "en"],
        use_gpu=(device == "cuda"),
        download_enabled=True,
    )

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_options,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
        accelerator_options=AcceleratorOptions(num_threads=4, device=device),
    )

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    result = converter.convert(pdf_url)
    doc = result.document

    # Group content by page using Docling's provenance info
    page_texts = defaultdict(list)
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text:
            # Try markdown export for items like tables/pictures
            # Newer Docling versions require the doc argument
            if hasattr(item, "export_to_markdown"):
                try:
                    text = cast(Any, item).export_to_markdown(doc)
                except TypeError:
                    try:
                        text = cast(Any, item).export_to_markdown()
                    except Exception:
                        pass
        if not text:
            continue
        prov = getattr(item, "prov", None)
        page_no = prov[0].page_no if prov else 1
        page_texts[page_no].append(text)

    if page_texts:
        total_chars = sum(len("\n\n".join(texts)) for texts in page_texts.values())
        logging.info(f"PDF Docling full extraction OK ({total_chars} chars, {len(page_texts)} pages): {pdf_url}")
        documents = []
        for page_no in sorted(page_texts.keys()):
            text = "\n\n".join(page_texts[page_no])
            if text.strip():
                documents.append(
                    PageDoc(
                        page_content=text,
                        metadata={"source": pdf_url, "page_number": page_no},
                    )
                )
        return documents

    # Fallback: if iterate_items produced nothing, use full markdown export
    md_content = doc.export_to_markdown()
    if not md_content or not md_content.strip():
        return []

    logging.info(f"PDF Docling full extraction fallback ({len(md_content)} chars): {pdf_url}")
    return [PageDoc(page_content=md_content, metadata={"source": pdf_url, "page_number": 1})]


def _page_docs_to_text(page_docs: list[PageDoc]) -> str:
    pages: list[str] = []
    for page_doc in page_docs:
        text = page_doc.page_content.strip()
        if not text:
            continue
        page_number = page_doc.metadata.get("page_number")
        prefix = f"[Page {page_number}]\n" if isinstance(page_number, int) and page_number > 0 else ""
        pages.append(f"{prefix}{text}")
    return "\n\n".join(pages).strip()


def _pdf_page_count(pdf_path: str) -> int:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def _assess_cached_entry(
    cache_path: str, pdf_path: str, filename: str, ocr_budget: int, ocr_for, pending: dict[str, int]
) -> int:
    """Quality check of one cached extraction: decode substituted glyph codes, OCR pages without text within the run's page budget,
    never keep unreadable text. Returns the number of pages sent to OCR; the pages still waiting for OCR are put into
    `pending` under the file name."""
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        if quality_is_current(entry.get("quality")):
            return 0
        ocr_fn = ocr_for(pdf_path) if ocr_budget > 0 else None
        new, sent = repair_entry(
            entry,
            _pdf_page_count(pdf_path),
            ocr_fn=ocr_fn,
            ocr_budget=max(ocr_budget, 0),
            before_ocr=lambda pages: write_json_atomic(cache_path, ocr_attempt_marker(entry, pages)),
        )
        write_json_atomic(cache_path, new)
        quality = new["quality"]
        if quality["pending_ocr_pages"]:
            pending[filename] = len(quality["pending_ocr_pages"])
        if quality["ocr_pages"] or quality["pending_ocr_pages"]:
            logging.info(
                f"[PdfQuality] {filename}: "
                f"ocr {len(quality['ocr_pages'])}, empty {len(quality['empty_pages'])}, "
                f"pending {len(quality['pending_ocr_pages'])}"
            )
        return sent
    except Exception as exc:
        logging.warning(f"[PdfQuality] check failed for {filename}: {exc}")
        return 0


def extract_pdfs(*, inventory: list[dict], extraction: Extraction, extracted_dir: str, skipped_log: str) -> dict:
    """
    Extract text from PDFs in pdf_dir, cache results in extracted_dir.

    Returns summary dict with counts and file details.
    """
    os.makedirs(extracted_dir, exist_ok=True)

    results = {"processed": 0, "cached": 0, "failed": 0, "skipped_page_limit": 0, "files": [], "ocr_pending": {}}
    skipped_page_limit_files = []
    # Quality check of every extraction: at most
    # ocr_pages_per_run pages go to OCR per run, so a run keeps its time window
    ocr_budget = extraction.ocr_pages_per_run
    ocr_for = ocr_session(device="cpu", num_threads=4)

    for pdf_entry in inventory:
        filename = pdf_entry["filename"]
        pdf_path = pdf_entry["path"]
        file_hash = pdf_entry["file_hash"]
        cache_path = os.path.join(extracted_dir, f"{file_hash}.json")

        if os.path.exists(cache_path):
            ocr_budget -= _assess_cached_entry(
                cache_path, pdf_path, filename, ocr_budget, ocr_for, results["ocr_pending"]
            )
            logging.info(f"Cached: {filename} ({file_hash[:12]}...)")
            results["cached"] += 1
            results["files"].append(
                {
                    "file": filename,
                    "status": "cached",
                    "hash": file_hash,
                    "url": pdf_entry.get("download_url"),
                }
            )
            continue

        text = ""
        method = "none"
        pdf_mode = extraction.processing

        try:
            if pdf_mode == "full":
                docs = _load_pdf_with_docling_full(pdf_path, extraction.max_ocr_pages)
                if docs:
                    text = _page_docs_to_text(docs)
                    method = "docling_full"
            else:
                try:
                    text = _page_docs_to_text(_extract_text_with_pypdfium2(pdf_path))
                    method = "pypdfium2"
                except Exception as exc:
                    logging.warning(f"pypdfium2 failed for {filename}: {exc}")

                if len(text) < extraction.min_text_length:
                    logging.info(f"Fast extraction insufficient for {filename}, trying Docling OCR...")
                    try:
                        docs = _load_pdf_with_docling_full(pdf_path, extraction.max_ocr_pages)
                        if docs:
                            text = _page_docs_to_text(docs)
                            method = "docling_ocr"
                    except PageLimitExceededError:
                        raise
                    except Exception as exc:
                        logging.warning(f"Docling OCR failed for {filename}: {exc}")

        except PageLimitExceededError as exc:
            logging.warning(
                f"Skipping {filename}: {exc.page_count} pages exceeds MAX_OCR_PAGES="
                f"{exc.max_pages} — would risk OOM-crashing the container."
            )
            results["skipped_page_limit"] += 1
            skip_record = {
                "file": filename,
                "status": "skipped_page_limit",
                "pages": exc.page_count,
                "max_pages": exc.max_pages,
                "url": pdf_entry.get("download_url"),
            }
            results["files"].append(skip_record)
            skipped_page_limit_files.append(skip_record)
            continue

        except Exception as exc:
            logging.error(f"Failed to extract {filename}: {exc}")
            results["failed"] += 1
            results["files"].append({"file": filename, "status": "failed", "error": str(exc)})
            continue

        # A fast-reader result with little text (pages drawn as outlines) is cached anyway:
        # the quality check below sends its pages to OCR
        if (not text or len(text.strip()) < extraction.min_text_length) and method != "pypdfium2":
            logging.warning(f"No usable text extracted from {filename}")
            results["failed"] += 1
            results["files"].append({"file": filename, "status": "failed", "error": "no text extracted"})
            continue

        cache_entry = {
            "source_pdf": filename,
            "file_hash": file_hash,
            "source_url": pdf_entry.get("url"),
            "download_url": pdf_entry.get("download_url"),
            "download_hash": pdf_entry.get("download_hash"),
            "download_title": pdf_entry.get("download_title"),
            "extracted_at": datetime.now().isoformat(),
            "extraction_method": method,
            "text": text,
        }
        write_json_atomic(cache_path, cache_entry)
        ocr_budget -= _assess_cached_entry(cache_path, pdf_path, filename, ocr_budget, ocr_for, results["ocr_pending"])

        logging.info(f"Extracted: {filename} ({len(text)} chars, {method})")
        results["processed"] += 1
        results["files"].append(
            {
                "file": filename,
                "status": "extracted",
                "url": pdf_entry.get("download_url"),
                "method": method,
                "chars": len(text),
            }
        )

    if skipped_page_limit_files:
        logging.warning(
            f"Skipped {len(skipped_page_limit_files)} PDF(s) due to MAX_OCR_PAGES limit: "
            + ", ".join(f"{f['file']} ({f['pages']} pages)" for f in skipped_page_limit_files)
        )
        try:
            os.makedirs(os.path.dirname(skipped_log) or ".", exist_ok=True)
            with open(skipped_log, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_at": datetime.now().isoformat(),
                        "max_ocr_pages": extraction.max_ocr_pages,
                        "skipped": skipped_page_limit_files,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logging.info(f"Wrote skipped-PDF summary to {skipped_log}")
        except Exception as exc:
            logging.error(f"Failed to write skipped-PDF summary to {skipped_log}: {exc}")

    return results
