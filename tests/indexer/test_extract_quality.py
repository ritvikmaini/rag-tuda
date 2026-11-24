"""PDF text quality: page assessment and page repair."""

import json
import os

from indexer.extract import quality as q  # noqa: E402

ENCODED = "\uf061" * 40  # substituted glyph codes: no text a reader could use


def test_classify_pages():
    assert q.classify_page("Die Kernarbeitszeit endet um 15:30 Uhr.") == "ok"
    assert q.classify_page(ENCODED) == "encoded"
    assert q.classify_page("2 / 142") == "empty"
    assert q.classify_page("") == "empty"


def stamped(n, body=""):
    return f"Satzungsbeilage 2023-IV,\nSeite {n} von 225" + (f"\n{body}" if body else "")


def test_a_page_of_page_stamps_alone_is_empty_and_goes_to_ocr():
    pages = {n: stamped(n + 29) for n in range(1, 13)}
    pages[8] = stamped(37, "o\no")  # stray marks of the scan
    assert q.classify_page(pages[1]) == "ok"  # without the document's other pages it looks like text
    asked = []

    def ocr(numbers):
        asked.append(list(numbers))
        return {n: f"§ {n} Der Studiengang gliedert sich in Module und Prüfungen. " + stamped(n + 29) for n in numbers}

    repaired, quality = q.repair_pages(pages, page_count=12, ocr_fn=ocr)
    assert asked == [list(range(1, 13))] and quality["ocr_pages"] == list(range(1, 13))
    assert repaired[3].startswith("§ 3 Der Studiengang")


def test_ocr_that_returns_only_the_stamp_leaves_the_document_without_text():
    pages = {n: stamped(n) for n in range(1, 5)}
    repaired, quality = q.repair_pages(pages, page_count=4, ocr_fn=lambda numbers: {n: stamped(n) for n in numbers})
    assert repaired == {} and quality["empty_pages"] == [1, 2, 3, 4] and quality["ocr_pages"] == []


def test_a_label_on_some_pages_is_no_page_stamp():
    handbook = {
        n: f"Modulhandbuch Informatik {n}\n{word} vermittelt Grundlagen." for n, word in enumerate("ABCDEFGHIJ", 1)
    }
    for n in range(1, 7):  # the form label returns on the module pages only
        handbook[n] += "\n10 Kommentar"
    for n in range(1, 4):
        handbook[n] += "\nLiteratur\nVorlesungsskript und Übungsblätter"
    assert q.stamp_lines(handbook, 10) == {"Modulhandbuch Informatik 0"}
    repaired, quality = q.repair_pages(handbook, page_count=10, ocr_fn=None)
    assert sorted(repaired) == list(range(1, 11)) and quality["pending_ocr_pages"] == []


def test_symbol_font_bullets_are_not_encoded_pages():
    page = " Erste Hilfe leisten\n Notruf 112 wählen\n Unfallstelle absichern"
    assert q.classify_page(page) == "ok"


def test_is_readable():
    assert q.is_readable("Unterabschnitt: 7911 Sonstige Förderung der Wirtschaft | 30.000")
    assert not q.is_readable("" * 3)


def test_split_and_join_round_trip():
    text = "[Page 1]\nErste Seite\n\n[Page 3]\nDritte Seite"
    assert q.split_pages(text) == {1: "Erste Seite", 3: "Dritte Seite"}
    assert q.join_pages(q.split_pages(text)) == text
    assert q.split_pages("Text ohne Seitenmarke") == {1: "Text ohne Seitenmarke"}


def test_repair_ocrs_encoded_and_empty_pages_and_drops_what_stays_unreadable():
    good = "Die Dienstvereinbarung gilt ab dem 01.06.2024 für alle Beschäftigten."
    pages = {1: good, 2: ENCODED, 3: "5 / 142"}
    asked = []

    def ocr(numbers):
        asked.append(list(numbers))
        return {
            2: "A. Betriebshaushalt 2025 Einzelplan 7 Forschungsförderung",
            3: "Vorbericht: 27.481 Beschäftigte",
            4: "",
        }

    repaired, quality = q.repair_pages(pages, page_count=4, ocr_fn=ocr)
    assert asked == [[2, 3, 4]]
    assert repaired[1] == good
    assert repaired[2].startswith("A. Betriebshaushalt")
    assert repaired[3].startswith("Vorbericht")
    assert 4 not in repaired
    assert quality == {
        "version": q.QUALITY_VERSION,
        "ocr_pages": [2, 3],
        "empty_pages": [4],
        "pending_ocr_pages": [],
    }


def test_ocr_budget_leaves_the_rest_pending_and_no_ocr_means_no_junk():
    pages = {1: "", 2: "", 3: "\uf06f" * 30}  # empty pages and an encoded one
    repaired, quality = q.repair_pages(
        pages, page_count=3, ocr_fn=lambda n: {k: "Seite mit genug lesbarem Text " + str(k) for k in n}, ocr_budget=1
    )
    assert list(repaired) == [1] and quality["ocr_pages"] == [1] and quality["pending_ocr_pages"] == [2, 3]
    repaired2, quality2 = q.repair_pages(pages, page_count=3, ocr_fn=None)
    assert repaired2 == {} and quality2["pending_ocr_pages"] == [1, 2, 3]


def test_a_failing_ocr_leaves_the_pages_pending():
    def broken(numbers):
        raise RuntimeError("docling crashed")

    repaired, quality = q.repair_pages({1: "Genug lesbarer Text auf der ersten Seite."}, page_count=2, ocr_fn=broken)
    assert list(repaired) == [1] and quality["pending_ocr_pages"] == [2] and quality["ocr_pages"] == []


def test_repair_entry_only_ocrs_missing_pages_of_fast_reader_entries():
    docling_entry = {
        "extraction_method": "docling_full",
        "text": "[Page 1]\n" + ("Lesbarer Text der ersten Seite. " * 10).strip(),
    }
    new, sent = q.repair_entry(docling_entry, page_count=2, ocr_fn=lambda n: {k: "neu" for k in n})
    assert sent == 0 and new["text"] == docling_entry["text"] and new["quality"]["empty_pages"] == [2]
    assert new["extraction_method"] == "docling_full"
    fast_entry = {"extraction_method": "pypdfium2", "text": "[Page 1]\nLesbarer Text der ersten Seite hier."}
    new2, sent2 = q.repair_entry(
        fast_entry, page_count=2, ocr_fn=lambda n: {k: "Lesbarer OCR-Text der zweiten Seite." for k in n}
    )
    assert sent2 == 1 and new2["text"].endswith("[Page 2]\nLesbarer OCR-Text der zweiten Seite.")
    assert new2["extraction_method"] == "pypdfium2+repair"


def test_an_older_assessment_keeps_what_ocr_found_empty():
    entry = {
        "extraction_method": "pypdfium2",
        "text": "[Page 1]\nLesbarer Text der ersten Seite hier.",
        "quality": {"version": 1, "ocr_pages": [], "empty_pages": [2], "pending_ocr_pages": []},
    }
    assert not q.quality_is_current(entry["quality"])
    asked = []
    new, _sent = q.repair_entry(entry, page_count=3, ocr_fn=lambda n: asked.append(list(n)) or {})
    assert asked == [[3]] and new["quality"]["empty_pages"] == [2, 3]


def test_quality_is_current():
    assert q.quality_is_current({"version": q.QUALITY_VERSION, "pending_ocr_pages": []})
    assert not q.quality_is_current({"version": q.QUALITY_VERSION, "pending_ocr_pages": [4]})
    assert not q.quality_is_current(None)


PLACEHOLDER = "<!-- 🖼️❌ Image not available. Please use `PdfPipelineOptions(generate_picture_images=True)` -->"


def test_clean_ocr_text_drops_placeholders_and_table_padding():
    padded = "| Gr     | Ansatz 2025   |\n|--------|---------------|\n| 5101   | 30.000        |"
    assert (
        q.clean_ocr_text(PLACEHOLDER + "\n\n" + padded + "\n\n\n\nEnde  der   Seite")
        == "| Gr | Ansatz 2025 |\n|---|---|\n| 5101 | 30.000 |\n\nEnde der Seite"
    )


def test_repair_cleans_ocr_output_and_a_placeholder_alone_is_no_text():
    table = "| Gr     | Bezeichnung                     | Ansatz 2025   |\n|--------|---------------------------------|---------------|\n| 5101   | Unterhalt des sonstigen Vermögens | 30.000        |"
    repaired, quality = q.repair_pages(
        {}, page_count=2, ocr_fn=lambda n: {1: PLACEHOLDER + "\n\n" + table, 2: PLACEHOLDER}
    )
    assert (
        repaired[1]
        == "| Gr | Bezeichnung | Ansatz 2025 |\n|---|---|---|\n| 5101 | Unterhalt des sonstigen Vermögens | 30.000 |"
    )
    assert quality["ocr_pages"] == [1] and quality["empty_pages"] == [2]


def test_placeholder_only_docling_entry_is_near_empty_and_goes_to_ocr():
    entry = {"extraction_method": "docling_full", "text": "[Page 1]\n" + PLACEHOLDER + "\n\n[Page 2]\n" + PLACEHOLDER}
    new, sent = q.repair_entry(
        entry, page_count=3, ocr_fn=lambda n: {k: "8908 .5000 Ansatz 2025: 121.500 siehe Erläuterungen" for k in n}
    )
    assert sent == 3 and new["quality"]["ocr_pages"] == [1, 2, 3] and "<!--" not in new["text"]
    assert new["extraction_method"] == "docling_full+repair"


def test_a_placeholder_alone_is_an_empty_page_of_a_docling_entry():
    repaired, quality = q.repair_pages(
        {1: PLACEHOLDER, 2: "Genug lesbarer Text auf der zweiten Seite."}, page_count=2, ocr_missing=False
    )
    assert list(repaired) == [2] and quality["empty_pages"] == [1]


def test_pages_ocr_found_empty_are_not_sent_again_on_later_runs():
    entry = {"extraction_method": "pypdfium2", "text": "[Page 1]\nLesbarer Text der ersten Seite hier."}
    asked = []

    def ocr(numbers):
        asked.append(list(numbers))
        return {k: ("" if k < 4 else "Lesbarer OCR-Text der Seite " + str(k)) for k in numbers}

    for _night in range(3):
        entry, _sent = q.repair_entry(entry, page_count=6, ocr_fn=ocr, ocr_budget=2)
    assert asked == [[2, 3], [4, 5], [6]]
    assert entry["quality"]["empty_pages"] == [2, 3] and q.quality_is_current(entry["quality"])
    assert entry["text"].endswith("[Page 6]\nLesbarer OCR-Text der Seite 6")


def test_write_json_atomic_keeps_the_old_file_when_writing_fails(tmp_path):
    path = tmp_path / "entry.json"
    q.write_json_atomic(str(path), {"text": "alt"})
    try:
        q.write_json_atomic(str(path), {"text": "neu", "bad": object()})
    except TypeError:
        pass
    assert json.loads(path.read_text(encoding="utf-8")) == {"text": "alt"}
    assert os.listdir(tmp_path) == ["entry.json"]


def test_pages_sent_to_ocr_are_counted_even_when_ocr_fails():
    def broken(numbers):
        raise RuntimeError("docling crashed")

    entry = {"extraction_method": "pypdfium2", "text": "[Page 1]\nLesbarer Text der ersten Seite hier."}
    new, sent = q.repair_entry(entry, page_count=4, ocr_fn=broken, ocr_budget=2)
    assert sent == 2 and new["quality"]["pending_ocr_pages"] == [2, 3, 4]


def test_an_ocr_attempt_that_never_finished_is_not_retried():
    entry = {"extraction_method": "pypdfium2", "text": "[Page 1]\nLesbarer Text der ersten Seite hier."}
    markers = []

    def killed(numbers):  # the process dies inside OCR: only the marker written before remains
        raise SystemExit

    try:
        q.repair_entry(
            entry,
            page_count=3,
            ocr_fn=killed,
            before_ocr=lambda pages: markers.append(q.ocr_attempt_marker(entry, pages)),
        )
    except SystemExit:
        pass
    marker = markers[-1]
    assert marker["text"] == entry["text"] and not q.quality_is_current(marker["quality"])
    asked = []
    new, sent = q.repair_entry(marker, page_count=3, ocr_fn=lambda n: asked.append(n) or {})
    assert asked == [] and sent == 0 and new["quality"]["empty_pages"] == [2, 3]
    assert q.quality_is_current(new["quality"]) and "ocr_attempt" not in new["quality"]


def test_clean_docling_markup_removes_placeholders_and_table_padding():
    text = (
        "[Page 1]\n" + PLACEHOLDER + "\n\n| 1411         | Nach Vereinbarung      | 65.000 €   |\n"
        "|:-------------|-----------------------:|------------|\n\n\n\nText  mit   Abstand"
    )
    assert q.clean_docling_markup(text) == (
        "[Page 1]\n| 1411 | Nach Vereinbarung | 65.000 € |\n|:---|---:|---|\n\nText  mit   Abstand"
    )


def test_clean_docling_markup_keeps_two_space_columns_and_prose():
    decoded = "| 5101  Unterhalt des sonstigen  30.000  -  - | Budg.=5.12.600,  0600 |"
    prose = "Die Kernarbeitszeit endet um 15:30 Uhr.     Ab 01.06.2024 gilt die neue Regelung."
    assert q.clean_docling_markup(decoded) == decoded
    assert q.clean_docling_markup(prose) == prose
    assert q.clean_docling_markup(None) == ""


def test_clean_docling_markup_is_idempotent_and_keeps_page_marks():
    text = "[Page 2]\n| a      | b |\n|------|---|\n\n[Page 3]\n" + PLACEHOLDER
    once = q.clean_docling_markup(text)
    assert q.clean_docling_markup(once) == once
    assert once.startswith("[Page 2]\n| a | b |\n|---|---|") and "[Page 3]" in once


def test_clean_docling_markup_leaves_no_blank_line_after_a_page_mark():
    text = "[Page 1]\n" + PLACEHOLDER + "\n\nErster Absatz\n\n[Page 2]\n" + PLACEHOLDER + "\n\n[Page 3]\nText"
    assert q.clean_docling_markup(text) == "[Page 1]\nErster Absatz\n\n[Page 2]\n[Page 3]\nText"
