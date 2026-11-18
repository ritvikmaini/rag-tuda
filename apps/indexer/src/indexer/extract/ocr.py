"""
OCR of selected PDF pages with Docling.

The pages are copied into temporary sub-PDFs of at most batch_size pages, so memory stays
bounded for any page count (MAX_OCR_PAGES guards only the whole-document path). Docling's
full pipeline runs with full-page OCR, because the text of these pages exists only as drawn
outlines; tables come out as markdown, as in extract.pdf._load_pdf_with_docling_full.
Docling and pypdfium2 are imported lazily; tests use a fake converter (tests/indexer/test_extract_ocr.py).
"""

import os
import tempfile
from collections import defaultdict


def texts_by_page(doc, batch):
    """{original page number: text} from a Docling document converted from a sub-PDF whose
    page k (1-based) is original page batch[k - 1]."""
    texts = defaultdict(list)
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text and hasattr(item, "export_to_markdown"):
            try:
                text = item.export_to_markdown(doc)
            except TypeError:
                text = item.export_to_markdown()
        if not text:
            continue
        prov = getattr(item, "prov", None)
        k = prov[0].page_no if prov else 1
        if 1 <= k <= len(batch):
            texts[batch[k - 1]].append(text)
    return {n: "\n\n".join(texts[n]) for n in batch if texts.get(n)}


def make_converter(engine="easyocr", device="cpu", num_threads=4):
    """A Docling converter with layout, table structure and full-page OCR."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if engine == "ocrmac":
        from docling.datamodel.pipeline_options import OcrMacOptions

        ocr_options = OcrMacOptions(lang=["de-DE", "en-US"], force_full_page_ocr=True)
    else:
        from docling.datamodel.pipeline_options import EasyOcrOptions

        ocr_options = EasyOcrOptions(lang=["de", "en"], use_gpu=(device == "cuda"), force_full_page_ocr=True)
    options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_options,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
        accelerator_options=AcceleratorOptions(num_threads=num_threads, device=device),
    )
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def docling_ocr_pages(
    pdf_path, page_numbers, batch_size=20, engine="easyocr", device="cpu", num_threads=4, converter=None
):
    """{page number: text} for the given pages (1-based) of pdf_path; pages without any
    recognised text are left out."""
    import pypdfium2 as pdfium

    converter = converter or make_converter(engine, device, num_threads)
    pages = sorted(set(page_numbers))
    out = {}
    src = pdfium.PdfDocument(pdf_path)
    try:
        for i in range(0, len(pages), batch_size):
            batch = pages[i : i + batch_size]
            sub = pdfium.PdfDocument.new()
            try:
                sub.import_pages(src, [n - 1 for n in batch])
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "pages.pdf")
                    sub.save(path)
                    doc = converter.convert(path).document
            finally:
                sub.close()
            out.update(texts_by_page(doc, batch))
    finally:
        src.close()
    return out


def ocr_session(make=None, **options):
    """ocr_for(pdf_path) -> ocr_fn(page_numbers) for one ingestion run: one converter, created
    on first use; after the first failure every later call fails at once, so a broken OCR
    costs one attempt per run, not one per PDF."""
    state = {}

    def ocr_for(pdf_path):
        def run(pages):
            if state.get("failed"):
                raise RuntimeError("OCR disabled for this run after an earlier failure")
            try:
                if "converter" not in state:
                    state["converter"] = (make or make_converter)(**options)
                return docling_ocr_pages(pdf_path, pages, converter=state["converter"])
            except Exception:
                state["failed"] = True
                raise

        return run

    return ocr_for
