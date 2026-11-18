"""Structure-aware chunking. The module changes to the repo root before importing the indexer."""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

from indexer.chunking import structure as sc  # noqa: E402

DOC = """[Page 1]
DIENSTANWEISUNG ARBEITSZEIT

§ 1 Geltungsbereich

Diese Dienstanweisung gilt für alle Beschäftigten.

§ 2 Kernzeit

Die Kernzeit beginnt um 9 Uhr
[Page 2]
und endet um 15 Uhr.

| Tag | Beginn | Ende |
| --- | --- | --- |
| Mo | 9 | 15 |
| Fr | 9 | 12 |
"""


def test_heading_rules():
    assert sc.heading("5. Funktions- und Transferkontrolle") == (1, "5. Funktions- und Transferkontrolle")
    assert sc.heading("5.1 Kontrolle durch die Scan-Arbeitskraft")[0] == 2
    assert sc.heading("§ 4 Arbeitszeit")[0] == 2
    assert sc.heading("Anlage 2")[0] == 1
    assert sc.heading("## Geltungsbereich") == (2, "Geltungsbereich")
    assert sc.heading("ALLGEMEINE BESTIMMUNGEN") == (1, "ALLGEMEINE BESTIMMUNGEN")
    assert sc.heading("1. Die Belege müssen bis zum 13. Werktag vorliegen.") is None  # a sentence
    assert sc.heading("5.1 Kontrolle .......... 12") is None  # table of contents
    assert sc.heading("1.200 Euro im Jahr") is None
    assert sc.heading("Die Kernzeit beginnt um 9 Uhr") is None


def test_blocks_pages_and_section_paths():
    blocks = sc.assign_paths(sc.parse_blocks(DOC))
    assert [(b.kind, b.page) for b in blocks] == [
        ("heading", 1),
        ("heading", 1),
        ("text", 1),
        ("heading", 1),
        ("text", 1),
        ("text", 2),
        ("table", 2),
    ]
    assert blocks[2].path == ("DIENSTANWEISUNG ARBEITSZEIT", "§ 1 Geltungsbereich")
    assert blocks[6].path == ("DIENSTANWEISUNG ARBEITSZEIT", "§ 2 Kernzeit")


def test_parent_across_pages_starts_with_its_page_mark():
    pages = [(1, "§ 2 Kernzeit\n\nDie Kernzeit beginnt um 9 Uhr"), (2, "und endet um 15 Uhr.")]
    [(text, page, path)] = sc.structure_parents(pages, 1200)
    assert page == 1 and path == ("§ 2 Kernzeit",)
    assert text == "[Page 1]\n§ 2 Kernzeit\n\nDie Kernzeit beginnt um 9 Uhr\n\n[Page 2]\nund endet um 15 Uhr."
    [(text, page, _)] = sc.structure_parents([(3, "Nur eine Seite mit Text.")], 1200)
    assert page == 3 and "[Page" not in text
    [(text, page, _)] = sc.structure_parents([(-1, "Eine Webseite ohne Seiten.")], 1200)
    assert page is None and "[Page" not in text


def test_small_sections_merge_long_sections_split_and_headings_never_end_a_parent():
    small = "\n\n".join(f"§ {n} Titel {n}\n\nKurzer Text {n}." for n in range(1, 6))
    parents = sc.structure_parents([(1, small)], 1200)
    assert len(parents) == 1 and parents[0][2] == ()  # five sections, no common path
    paragraph = "Ein Absatz mit genau einigen Wörtern darin, der sich wiederholt. " * 6
    long = "§ 1 Lang\n\n" + "\n\n".join(paragraph for _ in range(10))
    parents = sc.structure_parents([(1, long)], 1200)
    assert len(parents) == 4 and all(len(t) <= 1200 for t, _, _ in parents)
    assert parents[0][0].startswith("§ 1 Lang") and all(p[2] == ("§ 1 Lang",) for p in parents)
    assert all(sc.heading(t.split("\n")[-1]) is None for t, _, _ in parents)
    huge = "§ 9 Riesig\n\n" + ("Ein sehr langer Absatz ohne Ende " * 200).strip()
    parents = sc.structure_parents([(1, huge)], 1200)
    assert len(parents) > 4 and all(0 < len(t) <= 1200 + sc.MAX_HEADING_CHARS for t, _, _ in parents)


def test_long_markdown_table_split_by_rows_with_header():
    rows = "\n".join(f"| Zeile {n} | {'x' * 40} |" for n in range(60))
    parents = sc.structure_parents([(1, "| Name | Wert |\n| --- | --- |\n" + rows)], 1200)
    assert len(parents) > 1
    for text, _, _ in parents:
        lines = text.split("\n")
        assert lines[:2] == ["| Name | Wert |", "| --- | --- |"] and len(text) <= 1200
        assert all(line.startswith("| ") and line.endswith(" |") for line in lines)


def test_box_table_rows_never_cut():
    rows = "\n".join(" │ %s │ %s │" % ("Position %03d" % n, "1.234,00") for n in range(200))
    parents = sc.structure_parents([(5, rows)], 1200)
    assert all(t.split("\n")[0].lstrip().startswith("│") for t, _, _ in parents)
    assert sum(len(t.split("\n")) for t, _, _ in parents) == 200 and all(p == 5 for _, p, _ in parents)


def test_headings_only_document():
    [(text, page, _)] = sc.structure_parents([(1, "INHALT\n\n§ 1 Zweck\n\n§ 2 Geltung")], 1200)
    assert text == "INHALT\n\n§ 1 Zweck\n\n§ 2 Geltung" and page == 1


def test_children_stay_inside_one_page_and_are_substrings():
    text = "[Page 1]\n" + "Satz eins ist hier. " * 30 + "\n\n[Page 2]\n" + "Satz zwei ist dort. " * 30
    for mode in ("recursive", "sentences"):
        kids = sc.child_texts(text, mode, 450, 50)
        assert kids and all("[Page" not in k and k in text for k in kids)
    assert all(k.count("Satz") <= 3 for k in sc.child_texts(text, "sentences", 450, 50))


def test_context_header():
    meta = {"title": "Dienstanweisung KI", "doc_type": "Dienstanweisung/Dienstvereinbarung", "doc_year": 2025}
    assert (
        sc.context_header(meta, ("§ 3 Erlaubte Anwendungen",))
        == "Dienstanweisung KI · Dienstanweisung/Dienstvereinbarung · Stand 2025 · § 3 Erlaubte Anwendungen: "
    )
    assert sc.context_header({"title": "E-Kennzeichen", "doc_type": "Webseite"}) == "E-Kennzeichen · Webseite: "
    assert (
        sc.context_header({"title": "T", "doc_date": "2024-03-01T00:00:00"}, ("A", "B", "C", "D"))
        == "T · Stand 2024-03-01 · B > C > D: "
    )


def test_numbered_headings_close_unrelated_numbered_sections():
    text = "3. Aufgaben\n\n3.1 Kasse\n\nText eins.\n\n5.1.6 Onlinebanking\n\nText zwei."
    assert sc.assign_paths(sc.parse_blocks(text))[-1].path == ("5.1.6 Onlinebanking",)  # "5." and "5.1" unrecognised
    text = "3. Aufgaben\n\n3.1 Kasse\n\n3.1.2 Belege\n\nText."
    assert sc.assign_paths(sc.parse_blocks(text))[-1].path == ("3. Aufgaben", "3.1 Kasse", "3.1.2 Belege")
