"""OCR of selected pages on sub-PDFs: page
mapping and batching, tested with a fake Docling converter and small generated PDFs."""

from types import SimpleNamespace

import pypdfium2 as pdfium
import pytest

from indexer.extract import ocr as pdf_ocr  # noqa: E402


def make_pdf(path, pages):
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(595, 842)
    doc.save(path)
    doc.close()


class Item:
    def __init__(self, page, text=None, table=None):
        self.text, self._table = text, table
        self.prov = [SimpleNamespace(page_no=page)]
        if table is not None:
            self.export_to_markdown = lambda doc=None: self._table


class FakeConverter:
    """Returns, for every page k of the sub-PDF it gets, one text item 'sub page k of n'."""

    def __init__(self):
        self.batches = []

    def convert(self, path):
        n = len(pdfium.PdfDocument(path))
        self.batches.append(n)
        items = [(Item(k, text=f"sub page {k} of {n}"), 0) for k in range(1, n + 1)]
        return SimpleNamespace(document=SimpleNamespace(iterate_items=lambda: iter(items)))


def test_texts_by_page_maps_sub_pdf_pages_back_and_keeps_tables():
    doc = SimpleNamespace(
        iterate_items=lambda: iter(
            [
                (Item(1, text="Kopf"), 0),
                (Item(2, table="| a | b |\n|---|---|\n| 1 | 2 |"), 0),
                (Item(2, text="Fuss"), 0),
            ]
        )
    )
    assert pdf_ocr.texts_by_page(doc, [7, 9]) == {7: "Kopf", 9: "| a | b |\n|---|---|\n| 1 | 2 |\n\nFuss"}


def test_docling_ocr_pages_runs_bounded_batches_on_the_asked_pages(tmp_path):
    pdf = str(tmp_path / "book.pdf")
    make_pdf(pdf, 6)
    conv = FakeConverter()
    out = pdf_ocr.docling_ocr_pages(pdf, [5, 2, 4], batch_size=2, converter=conv)
    assert conv.batches == [2, 1]
    assert out == {2: "sub page 1 of 2", 4: "sub page 2 of 2", 5: "sub page 1 of 1"}


def test_ocr_session_shares_one_converter_and_stops_after_the_first_failure(tmp_path):
    pdf = str(tmp_path / "book.pdf")
    make_pdf(pdf, 3)
    made = []

    def make(**options):
        made.append(options)
        return FakeConverter()

    ocr_for = pdf_ocr.ocr_session(make=make, device="cpu", num_threads=4)
    assert ocr_for(pdf)([1]) == {1: "sub page 1 of 1"}
    assert ocr_for(pdf)([2, 3]) == {2: "sub page 1 of 2", 3: "sub page 2 of 2"}
    assert made == [{"device": "cpu", "num_threads": 4}]

    class Broken:
        def convert(self, path):
            raise RuntimeError("model download failed")

    broken = pdf_ocr.ocr_session(make=lambda **options: Broken())
    with pytest.raises(RuntimeError, match="model download failed"):
        broken(pdf)([1])
    with pytest.raises(RuntimeError, match="disabled"):
        broken(pdf)([2])
