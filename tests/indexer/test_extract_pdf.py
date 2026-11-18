"""The extraction takes its values from the profile and reads no config file."""

import importlib
import json
import sys

from indexer.extract import pdf
from rag_core.settings.profile import Extraction


def _extraction(**changes):
    values = dict(processing="fast", max_ocr_pages=500, ocr_pages_per_run=100, clean_markup=True, min_text_length=50)
    return Extraction(**{**values, **changes})


def _entry(tmp_path, name):
    return {
        "filename": name + ".pdf",
        "path": str(tmp_path / (name + ".pdf")),
        "file_hash": name,
        "download_url": "https://docs.example.org/" + name,
    }


def test_extraction_reads_no_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config/ here
    assert not (tmp_path / "config").exists()
    monkeypatch.delitem(sys.modules, "indexer.extract.pdf")
    reloaded = importlib.import_module("indexer.extract.pdf")
    assert not hasattr(reloaded, "cfg")


def test_the_ocr_budget_and_the_page_limit_come_from_the_profile(tmp_path, monkeypatch):
    budgets = []
    monkeypatch.setattr(pdf, "ocr_session", lambda **options: None)
    monkeypatch.setattr(pdf, "_assess_cached_entry", lambda *args: budgets.append(args[3]) or 0)
    (tmp_path / "cached.json").write_text("{}", encoding="utf-8")
    pdf.extract_pdfs(
        inventory=[_entry(tmp_path, "cached")], extraction=_extraction(ocr_pages_per_run=7), extracted_dir=str(tmp_path)
    )
    assert budgets == [7]

    import pypdfium2

    document = pypdfium2.PdfDocument.new()
    for _ in range(3):
        document.new_page(200, 200)  # no text: the fast reader finds nothing and the Docling OCR is tried
    document.save(str(tmp_path / "big.pdf"))
    monkeypatch.setattr(pdf, "SKIPPED_PDFS_LOG", str(tmp_path / "skipped.json"))
    result = pdf.extract_pdfs(
        inventory=[_entry(tmp_path, "big")], extraction=_extraction(max_ocr_pages=2), extracted_dir=str(tmp_path)
    )
    assert result["skipped_page_limit"] == 1 and result["failed"] == 0
    assert json.loads((tmp_path / "skipped.json").read_text(encoding="utf-8"))["max_ocr_pages"] == 2


def test_a_cached_entry_is_counted_and_not_extracted_again(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf, "ocr_session", lambda **options: None)
    monkeypatch.setattr(pdf, "_assess_cached_entry", lambda *args: 0)
    monkeypatch.setattr(pdf, "_extract_text_with_pypdfium2", lambda path: (_ for _ in ()).throw(AssertionError(path)))
    (tmp_path / "done.json").write_text("{}", encoding="utf-8")
    result = pdf.extract_pdfs(
        inventory=[_entry(tmp_path, "done")], extraction=_extraction(), extracted_dir=str(tmp_path)
    )
    assert (result["cached"], result["processed"]) == (1, 0)
    assert result["files"][0]["status"] == "cached"
