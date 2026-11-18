import os
import json
from indexer.extract import documents
from rag_core.domain.sources import PdfItem


def test_split_pages_strips_placeholders():
    text = "[Page 1]\nTitel der Satzung\n<!-- 🖼️❌ Image not available. Please use `PdfPipelineOptions(generate_picture_images=True)` -->\n\n[Page 2]\n§ 1 Geltungsbereich der Satzung über ...\n\n[Page 3]\nx"
    pages = documents.split_pages(text)
    assert [p for p, _ in pages] == [2]
    assert "PdfPipelineOptions" not in pages[0][1]


def test_resolve_title():
    assert documents.resolve_title(
        "Richtlinie Pflege Von Anlagen Mit Grnflchen 2031 02",
        "Richtlinie zur Pflege von Anlagen mit Grünflächen",
    ).startswith("Richtlinie zur")
    assert documents.resolve_title("Formblatt X", "Microsoft Word - form.docx", "Seite 1\nAntrag") == "Formblatt X"
    assert (
        documents.resolve_title("", "", "Seite 1\nAntrag auf Anwohnerparkausweis") == "Antrag auf Anwohnerparkausweis"
    )
    assert documents.resolve_title("", "", "") == "Dokument"


def test_clean_download_title():
    assert documents.clean_download_title("Beispielregelung 1991 03 07") == "Beispielregelung (07.03.1991)"
    assert documents.clean_download_title("Musterordnung 14 06 2012") == "Musterordnung (14.06.2012)"
    assert documents.clean_download_title("420 3 2 Satzung") == "420 3 2 Satzung"  # numbers are kept as written
    assert documents.clean_download_title("Haushaltsbroschüre 2023") == "Haushaltsbroschüre 2023"
    assert documents.clean_download_title("Amtsblatt Nr. 51_28") == "Amtsblatt Nr. 51_28"


def test_generic_or_file_name_titles_fall_back_to_the_manifest_title():
    # PDF metadata titles shared by several PDFs ("Ihr Zeichen") or naming a source
    # file are not titles; manifest titles lose a trailing file extension.
    assert (
        documents.resolve_title("Fachstelle 05 Vollzug Beispielgesetz", "Ihr Zeichen", generic_titles={"Ihr Zeichen"})
        == "Fachstelle 05 Vollzug Beispielgesetz"
    )
    assert (
        documents.resolve_title(
            "Übersicht Schwellenwerte 2031", "(2031-01-01 - Übersicht Schwellenwerte öffentlicher Auftraggeber.xlsx)"
        )
        == "Übersicht Schwellenwerte 2031"
    )
    assert documents.clean_download_title("Richtlinie 5200.pdf") == "Richtlinie 5200"
    assert (
        documents.clean_download_title("2220_Dokumentation_Musterverfahren.PDF") == "2220_Dokumentation_Musterverfahren"
    )
    assert documents.resolve_title(
        "Richtlinie Anlagen 2031 02",
        "Richtlinie zur Pflege von Anlagen mit Grünflächen",
        generic_titles={"Ihr Zeichen"},
    ).startswith("Richtlinie zur")


def test_split_pages_keeps_angle_bracket_text_from_pdfs():
    # The tag regex must not delete real PDF text between "<" and ">":
    # organisational units in the Haushaltsplan, "less than" figures, form placeholders, table cells.
    text = (
        "[Page 1]\nUnterabschnitt 5678 < Naturschutz >\nPersonalausgaben 18.000 Euro\n"
        "[Page 2]\nWarmweißes Licht (< 3.300 Kelvin) wirkt gemütlich. Rechnungsnummer: <Rechnungsnummer>\n"
        "| Anforderung | Wert |\n|---|---|\n| Beleuchtung | <2.000 | <1.000 |\n"
    )
    pages = dict(documents.split_pages(text))
    assert "< Naturschutz >" in pages[1]
    assert "(< 3.300 Kelvin)" in pages[2] and "<Rechnungsnummer>" in pages[2]
    assert "| <2.000 | <1.000 |" in pages[2]


def _item(file_hash, path, download_title, url="https://docs.example.org/budget/x/?layout=file"):
    metadata = {"download_url": url, "download_title": download_title, "origin": "pdf"}
    return PdfItem(path=str(path), url=url, file_hash=file_hash, metadata=metadata)


def _cache(tmp_path, file_hash, text):
    (tmp_path / (file_hash + ".json")).write_text(json.dumps({"file_hash": file_hash, "text": text}), encoding="utf-8")


def test_the_first_pdf_with_a_hash_wins_and_the_rest_are_skipped(tmp_path):
    _cache(tmp_path, "same", "[Page 1]\nText der ersten Seite mit Inhalt.")
    first = _item("same", tmp_path / "a.pdf", "Erste Fassung", "https://docs.example.org/a/?layout=file")
    second = _item("same", tmp_path / "b.pdf", "Zweite Fassung", "https://docs.example.org/b/?layout=file")
    seen = set()
    docs, missing = documents.pdf_documents([first, second], seen, extracted_dir=str(tmp_path))
    assert missing == 0 and seen == {"same"}
    assert [(d["metadata"]["title"], d["metadata"]["download_url"]) for d in docs] == [
        ("Erste Fassung", "https://docs.example.org/a/?layout=file")
    ]


def test_pdf_documents_clean_docling_markup(tmp_path):
    text = (
        "[Page 1]\n<!-- 🖼️❌ Image not available. Please use `PdfPipelineOptions(generate_picture_images=True)` -->\n\n"
        "| Gr     | Ansatz 2025   |\n|--------|---------------|\n| 7302   | 45.000        |"
    )
    (tmp_path / "abc.json").write_text(json.dumps({"file_hash": "abc", "text": text}), encoding="utf-8")
    item = _item("abc", tmp_path / "missing.pdf", "Haushalt 2025")
    docs, missing = documents.pdf_documents([item], set(), extracted_dir=str(tmp_path))
    assert missing == 0
    assert [d["page_content"] for d in docs] == ["| Gr | Ansatz 2025 |\n|---|---|\n| 7302 | 45.000 |"]
    raw, _ = documents.pdf_documents([item], set(), extracted_dir=str(tmp_path), clean=False)
    assert "|--------|" in raw[0]["page_content"]


def test_pdf_documents_keep_angle_bracket_text(tmp_path):
    # the Docling cleanup removes the picture placeholder, the page keeps "< Naturschutz >".
    text = (
        "[Page 1]\n<!-- 🖼️❌ Image not available. Please use `PdfPipelineOptions(generate_picture_images=True)` -->\n\n"
        "Unterabschnitt 5678 < Naturschutz >\nPersonalausgaben 18.000 Euro"
    )
    (tmp_path / "abc.json").write_text(json.dumps({"file_hash": "abc", "text": text}), encoding="utf-8")
    item = _item("abc", tmp_path / "missing.pdf", "Haushalt 2025")
    docs, missing = documents.pdf_documents([item], set(), extracted_dir=str(tmp_path))
    assert missing == 0
    assert [d["page_content"] for d in docs] == ["Unterabschnitt 5678 < Naturschutz >\nPersonalausgaben 18.000 Euro"]


def test_a_template_title_shared_by_two_pdfs_is_not_used(tmp_path, monkeypatch):
    items = []
    for n, title in (
        (1, "Fachstelle 05 Vollzug Beispielgesetz"),
        (2, "Rundschreiben Kantinenkarte"),
        (3, "Dienstanweisung Datenschutz"),
    ):
        (tmp_path / ("h%d.json" % n)).write_text(
            json.dumps({"file_hash": "h%d" % n, "text": "[Page 1]\nText der Seite %d mit Inhalt." % n}),
            encoding="utf-8",
        )
        items.append(
            _item(
                "h%d" % n,
                tmp_path / ("d%d.pdf" % n),
                title,
                "https://docs.example.org/x/d%d/?layout=file" % n,
            )
        )
    meta = {
        "d1.pdf": "Ihr Zeichen",
        "d2.pdf": "Ihr Zeichen",
        "d3.pdf": "Dienstanweisung über den Datenschutz bei der Behörde",
    }
    monkeypatch.setattr(documents, "_pdf_meta_title", lambda path: meta[os.path.basename(path)])
    docs, _ = documents.pdf_documents(items, set(), extracted_dir=str(tmp_path))
    assert [d["metadata"]["title"] for d in docs] == [
        "Fachstelle 05 Vollzug Beispielgesetz",
        "Rundschreiben Kantinenkarte",
        "Dienstanweisung über den Datenschutz bei der Behörde",
    ]
