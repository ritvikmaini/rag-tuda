"""The chunker records every "§ n" of a parent, because small sections are merged and the section path keeps only the
prefix the merged sections share."""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

import pypdfium2  # noqa: E402
from rag_testing.minisite import pages  # noqa: E402

from indexer.chunking import structure as sc  # noqa: E402
from indexer.chunking import units  # noqa: E402


def _po_pages():
    pdf = pypdfium2.PdfDocument(pages.PO_2023)
    return [(n + 1, page.get_textpage().get_text_range()) for n, page in enumerate(pdf)]


def test_merged_paragraph_is_recorded():
    parents = sc.structure_parents(_po_pages(), 1200)
    [(first, _, path)] = [p for p in parents if "§ 2 Studienziel" in p[0]]
    assert "Das Studium vermittelt die Grundlagen der Informatik." in first
    assert len(first) < 1200 and "§ 3 Regelstudienzeit" in first  # the short § 2 shares a parent with its neighbours
    assert not any("§ 2" in step for step in path)  # the section path alone cannot find it
    assert sc.paragraphs(first)[:3] == ["§ 1", "§ 2", "§ 3"]

    docs = [
        {"page_content": t, "metadata": {"title": "PO", "doc_key": "po", "source_type": "pdf", "page_number": n}}
        for n, t in _po_pages()
    ]
    chunked = units.chunk_documents(docs, mode="structure")
    holder = next(p for p in chunked if "§ 2 Studienziel" in p.text)
    assert "§ 2" in holder.metadata["paragraphs"]
    assert holder.metadata["paragraphs"] == sc.paragraphs(holder.text)


def test_only_paragraph_headings_count_and_text_without_any_has_none():
    text = "§ 1 Zweck\n\nSiehe § 22 und § 23 APB.\n\n§ 2a Geltung\n\nText\n§ 24 Absatz 1 gilt.\n\nArtikel 3 Ende\n\n§ 4"
    assert sc.paragraphs(text) == ["§ 1", "§ 2a", "§ 4"]
    assert sc.paragraphs("[Page 2]\n§ 5 Frist Die Frist") == ["§ 5"]
    assert sc.paragraphs("Erster Satz.\n§ 6 Frist Die Frist\n§ 7 Ende") == ["§ 6", "§ 7"]  # after a paragraph start
    assert sc.paragraphs("§ 7 Frist Die Frist beträgt\r\n§ 7 Frist") == ["§ 7"]
    assert sc.paragraphs("Nur Text ohne Gliederung.") == []
    parents = units.chunk_documents(
        [{"page_content": "Nur Text.", "metadata": {"title": "T", "doc_key": "t", "page_number": -1}}], mode="structure"
    )
    assert "paragraphs" not in parents[0].metadata


def test_a_reference_wrapped_to_a_line_start_is_not_a_paragraph():
    wrapped = (
        "Nicht bestandene Prüfungen werden nach\n§ 22 APB gilt entsprechend.\n"
        "Die Dauer regelt\n§ 5 HHG regelt das weitere.\n"
        "Die Wiederholung ergibt sich aus\n§ 3\nder Satzung.\n"
        "Zur Frist siehe\n§ 24 Absatz 1 und\n§ 9 Anlage 1"
    )
    assert sc.paragraphs(wrapped) == []
