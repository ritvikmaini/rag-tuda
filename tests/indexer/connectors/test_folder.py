"""The folder source: Markdown, text and PDF files of a directory become one document each, in path order; the title, url,
doc_type and date follow the file, its front matter and the profile entry, and a PDF's item carries what the indexer's
extraction reads."""

import json
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace

import pypdfium2 as pdfium
import pytest
from rag_core.domain.sources import PageItem, PdfItem

from indexer.connectors.folder import Folder
from indexer.extract.documents import pdf_documents

BASE = "https://docs.example.org/"


def ctx(folder: Path | None, limit: int | None = None, **settings):
    return SimpleNamespace(data_dir=folder, limit=limit, settings=settings)


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def tiny_pdf(path: Path) -> None:
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 200)
    pdf.save(str(path))


def test_folder_reads_markdown_and_pdf(tmp_path: Path) -> None:
    write(tmp_path, "b/zweites.md", "# Zweites Dokument\n\n## Abschnitt\n\nText zwei.\n")
    write(tmp_path, "erstes.txt", "Nur Text, keine Überschrift.\n")
    tiny_pdf(tmp_path / "a.pdf")
    result = Folder().collect(ctx(tmp_path, base_url=BASE, doc_type="Gesetz", min_documents=3))
    pdf, page, text = result.items
    assert [type(i) for i in result.items] == [PdfItem, PageItem, PageItem]
    assert result.expected_minimum == 3 and result.report == {"files": 3}

    assert page.url == BASE + "b/zweites" and page.text.startswith("# Zweites Dokument")
    assert dict(page.metadata) == {
        "title": "Zweites Dokument",
        "url": BASE + "b/zweites",
        "source_type": "page",
        "doc_type": "Gesetz",
        "origin": "website",
        "section": "",
        "doc_year": None,
        "doc_date": None,
        "page_number": -1,
        "doc_key": BASE + "b/zweites",
    }
    assert text.metadata["title"] == "erstes"  # no "# " line: the file name

    assert (pdf.url, pdf.path) == (BASE + "a", str(tmp_path / "a.pdf"))
    assert len(pdf.file_hash) == 64
    # what the extraction's page documents read from the item
    cache = {
        "text": "[Page 1]\nEin Absatz Text, der lang genug ist.",
        "file_hash": pdf.file_hash,
        "extraction_method": "x",
    }
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / (pdf.file_hash + ".json")).write_text(json.dumps(cache), encoding="utf-8")
    docs, missing = pdf_documents([pdf], set(), str(extracted))
    assert missing == 0 and len(docs) == 1
    meta = docs[0]["metadata"]
    assert (meta["url"], meta["download_url"], meta["doc_type"], meta["source_type"], meta["page_number"]) == (
        BASE + "a",
        BASE + "a",
        "Gesetz",
        "pdf",
        1,
    )


def test_front_matter_sets_url_doc_type_and_date_and_is_not_indexed(tmp_path: Path) -> None:
    write(
        tmp_path,
        "po.md",
        '---\ndoc_type: Ordnung\ndate: 2026-03-01\nurl: "https://elsewhere.example/po"\n---\n\n# Prüfungsordnung\n\nText.\n',
    )
    (item,) = Folder().collect(ctx(tmp_path, base_url=BASE, doc_type="Gesetz")).items
    assert item.url == "https://elsewhere.example/po" and item.text.startswith("# Prüfungsordnung")
    assert "doc_type:" not in item.text
    assert (item.metadata["doc_type"], item.metadata["doc_date"], item.metadata["doc_year"]) == (
        "Ordnung",
        "2026-03-01",
        2026,
    )


def test_without_base_url_or_doc_type_the_url_is_relative_and_the_type_generic(tmp_path: Path) -> None:
    write(tmp_path, "sub/a.md", "# A\n")
    (item,) = Folder().collect(ctx(tmp_path)).items
    assert item.url == "folder:sub/a.md" and item.metadata["doc_type"] == "Dokument"
    assert str(tmp_path) not in item.url


def test_other_files_are_ignored_and_a_limit_takes_the_first_files(tmp_path: Path) -> None:
    for name in ("a.md", "b.md", "c.md", "notiz.csv", "bild.png"):
        write(tmp_path, name, f"# {name}\n")
    assert [i.url for i in Folder().collect(ctx(tmp_path, limit=2)).items] == ["folder:a.md", "folder:b.md"]
    assert len(Folder().collect(ctx(tmp_path)).items) == 3


def test_without_a_directory_the_source_refuses() -> None:
    with pytest.raises(RuntimeError, match=r"connectors\.<source>\.dir"):
        Folder().collect(ctx(None))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("---\ndate: gestern\n---\n# T\n", "YYYY-MM-DD"),
        ("---\ndoc_type: x\n# T\n", "not closed"),
        ("---\n- a\n---\n# T\n", "not a mapping"),
    ],
)
def test_a_bad_front_matter_is_refused_with_the_file_name(tmp_path: Path, text: str, message: str) -> None:
    write(tmp_path, "kaputt.md", text)
    with pytest.raises(ValueError, match=f"kaputt.md.*{message}"):
        Folder().collect(ctx(tmp_path))


def test_the_connector_is_found_by_its_entry_point() -> None:
    found = {ep.name: ep for ep in entry_points(group="rag_core.connectors")}
    assert found["folder"].load() is Folder


def test_an_own_folder_is_labelled_as_documents_and_a_site_copy_as_website(tmp_path: Path) -> None:
    write(tmp_path, "a.md", "# A\n")
    tiny_pdf(tmp_path / "b.pdf")
    own = Folder().collect(ctx(tmp_path)).items
    assert [i.metadata["origin"] for i in own] == ["documents", "documents"]
    site = Folder().collect(ctx(tmp_path, base_url=BASE)).items
    assert [i.metadata["origin"] for i in site] == ["website", "website"]


def test_two_files_with_one_url_are_refused_naming_both(tmp_path: Path) -> None:
    write(tmp_path, "a.md", "# A\n")
    write(tmp_path, "a.txt", "Text\n")
    with pytest.raises(ValueError, match=r"a\.md.*a\.txt|a\.txt.*a\.md"):
        Folder().collect(ctx(tmp_path, base_url=BASE))


def test_a_front_matter_url_repeated_by_another_file_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "a.md", "---\nurl: https://x.example/p\n---\n# A\n")
    write(tmp_path, "b.md", "---\nurl: https://x.example/p\n---\n# B\n")
    with pytest.raises(ValueError, match="https://x.example/p"):
        Folder().collect(ctx(tmp_path))


def test_a_symlink_out_of_the_folder_is_skipped_and_one_inside_is_kept(tmp_path: Path) -> None:
    folder, outside = tmp_path / "docs", tmp_path / "secret.md"
    write(folder, "a.md", "# A\n")
    outside.write_text("# Geheim\n", encoding="utf-8")
    (folder / "leak.md").symlink_to(outside)
    (folder / "same.md").symlink_to(folder / "a.md")
    assert [i.url for i in Folder().collect(ctx(folder)).items] == ["folder:a.md", "folder:same.md"]


def test_a_file_that_is_not_utf8_is_named(tmp_path: Path) -> None:
    (tmp_path / "alt.txt").write_bytes("Prüfung".encode("latin-1"))
    with pytest.raises(ValueError, match=r"alt\.txt.*UTF-8"):
        Folder().collect(ctx(tmp_path))


@pytest.mark.parametrize("base", ["https://docs.example.org", "https://docs.example.org/"])
def test_the_base_url_joins_with_one_slash_and_the_path_is_quoted(tmp_path: Path, base: str) -> None:
    write(tmp_path, "Studien ordnung/Teil 1.md", "# T\n")
    (item,) = Folder().collect(ctx(tmp_path, base_url=base)).items
    assert item.url == "https://docs.example.org/Studien%20ordnung/Teil%201"
