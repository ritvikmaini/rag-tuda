"""Extraction of fetched pages into Markdown documents and PDF links, run on the fixture mini-site's markup and on small
inline pages. Nothing is fetched."""

import pytest
from fastapi.testclient import TestClient
from rag_testing import serve
from rag_testing.minisite import pages

from indexer.crawl.extract import Selectors, extract

FACULTY = "http://faculty.test"
INTERN = "http://intern.test"
HUB_URL = f"{FACULTY}{pages.BSC_HUB}.de.jsp"
SITE = TestClient(serve.minisite_app("fb20", INTERN), base_url=FACULTY)


def _hub():
    return extract(SITE.get(f"{pages.BSC_HUB}.de.jsp").content, HUB_URL)


def _inline(main: str, url: str = f"{FACULTY}/a/b/page.de.jsp", **selectors):
    html = f'<html><head><title>T</title></head><body><div class="main-column">{main}</div></body></html>'
    return extract(html.encode(), url, Selectors(**selectors))


def test_hub_splits_into_one_document_per_po_section():
    result = _hub()
    assert result.problem is None
    assert result.title == pages.BSC_HUB_TITLE
    assert [d.section_id for d in result.documents] == [
        "pruefungsordnung_2023",
        "pruefungsordnung_2015",
        "allg_pruefungsbest___dateiliste",
    ]
    assert [d.heading for d in result.documents] == [
        "Prüfungsordnung 2023",
        "Prüfungsordnung 2015",
        "Allgemeine Prüfungsbestimmungen",
    ]
    po_2023, po_2015, _ = result.documents
    assert po_2023.markdown.startswith("### Prüfungsordnung 2023\n\n")
    assert "ab dem Wintersemester 2023/24" in po_2023.markdown
    assert "2015" not in po_2023.markdown
    assert "Ordnung von 2015" in po_2015.markdown
    assert "2023" in po_2015.markdown and "Studiengangsordnung Bachelor" not in po_2015.markdown
    assert f"[Studiengangsordnung Bachelor Informatik 2023]({FACULTY}{pages.PO_2023_PDF})" in po_2023.markdown
    assert "(PDF-Datei," in po_2023.markdown


def test_page_without_sections_is_one_document():
    result = _inline("<h1>Willkommen</h1><p>Erster Absatz.</p><p>Zweiter Absatz.</p>")
    assert [(d.section_id, d.heading) for d in result.documents] == [(None, "Willkommen")]
    assert result.documents[0].markdown == "# Willkommen\n\nErster Absatz.\n\nZweiter Absatz."


def test_text_around_sections_is_a_document_of_its_own():
    result = _inline(
        '<h1>Titel</h1><p>Vorwort.</p><section class="content-section" id="a"><h3>A</h3><p>x</p></section>'
        '<section class="content-section"><h3>B</h3><p>y</p></section>'
    )
    assert [(d.section_id, d.heading) for d in result.documents] == [(None, "Titel"), ("a", "A"), (None, "B")]
    assert "Vorwort." in result.documents[0].markdown


def test_chrome_dropped():
    result = _hub()
    text = "\n".join(d.markdown for d in result.documents)
    for chrome in ("Seitenfuß", "Studienbüro", "EN", "Datenschutz"):
        assert chrome not in text
    result = _inline(
        "<nav><a href='/x'>Menü</a></nav><script>var x = 1;</script><style>p {}</style><h1>H</h1>"
        "<aside>Seitenleiste</aside><p>Inhalt</p><footer><a href='/footer.pdf'>Fuß</a></footer>"
    )
    assert result.documents[0].markdown == "# H\n\nInhalt"
    assert result.pdfs == ()


def test_chrome_selectors_are_configurable():
    result = _inline('<div class="teaser">Werbung</div><p>Inhalt</p>', drop="div.teaser")
    assert [d.markdown for d in result.documents] == ["Inhalt"]


def test_pdf_links_by_href_suffix():
    result = _hub()
    assert [(p.url, p.section_id) for p in result.pdfs] == [
        (f"{FACULTY}{pages.PO_2023_PDF}", "pruefungsordnung_2023"),
        (f"{FACULTY}{pages.APB_COPY_PDF}", "allg_pruefungsbest___dateiliste"),
        (f"{INTERN}{pages.APB_ORIGINAL_PDF}", "allg_pruefungsbest___dateiliste"),
    ]
    first = result.pdfs[0]
    assert (first.text, first.title) == ("Studiengangsordnung Bachelor Informatik 2023", "in Kraft seit 01.10.2023")

    result = _inline(
        "<a href='a.PDF'>1</a><a href='/b.pdf?download=1'>2</a><a href='../c.pdf#page=2'>3</a>"
        "<a href='https://other.test/d.pdf'>4</a><a href='a.PDF'>again</a>"
        "<a href='e.pdfx'>no</a><a href='/f.jsp?file=g.pdf'>no</a><a href='mailto:h.pdf'>no</a><a name='i'>no</a>"
    )
    assert [p.url for p in result.pdfs] == [
        f"{FACULTY}/a/b/a.PDF",
        f"{FACULTY}/b.pdf?download=1",
        f"{FACULTY}/a/c.pdf",
        "https://other.test/d.pdf",
    ]
    assert result.pdfs[0].section_id is None


def test_pdf_links_of_an_aside_beside_the_main_content_follow_those_of_the_main_content():
    html = (
        "<html><body><nav><a href='/menu.pdf'>Menu</a></nav><div class='main-column'><section class='content-section' "
        "id='a'><a href='/main.pdf'>Main</a></section><aside><a href='/inner.pdf'>Inner</a></aside></div>"
        "<aside><section id='links___downloads'><h2>Links &amp; Downloads</h2><a href='/main.pdf'>again</a>"
        "<a href='/check.pdf' title='Antrag'>Checkliste</a><a href='/page.jsp'>Seite</a>"
        "<div hidden><a href='/hidden.pdf'>x</a></div></section></aside><footer><aside><a href='/foot.pdf'>x</a></aside>"
        "</footer></body></html>"
    )
    result = extract(html.encode(), f"{FACULTY}/p.de.jsp", Selectors())
    assert [(p.url, p.text, p.section_id, p.aside) for p in result.pdfs] == [
        (f"{FACULTY}/main.pdf", "Main", "a", False),
        (f"{FACULTY}/check.pdf", "Checkliste", None, True),
    ]
    assert result.pdfs[1].title == "Antrag"
    assert "Downloads" not in "".join(d.markdown for d in result.documents)  # the aside is no content


def test_markdown_keeps_lists_emphasis_links_and_tables():
    result = _inline(
        "<h2>Kopf</h2><p>Ein <strong>wichtiger</strong> und <em>kurzer</em> Text<br>mit Umbruch und "
        "<a href='/x?a=1#f'>Link</a>.</p><ul><li>eins<ul><li>innen</li></ul></li><li>zwei</li></ul>"
        "<ol><li>erst</li><li>dann</li></ol>"
        "<table><tr><th>Modul</th><th>CP</th></tr><tr><td>GDI</td><td>10</td></tr></table>"
    )
    assert result.documents[0].markdown == (
        "## Kopf\n\n"
        f"Ein **wichtiger** und *kurzer* Text\nmit Umbruch und [Link]({FACULTY}/x?a=1).\n\n"
        "- eins\n  - innen\n- zwei\n\n"
        "1. erst\n2. dann\n\n"
        "| Modul | CP |\n| --- | --- |\n| GDI | 10 |"
    )


def test_main_selector_is_configurable():
    html = b"<html><body><div id='x'><p>Gewollt</p></div><div class='main-column'><p>Anderes</p></div></body></html>"
    result = extract(html, HUB_URL, Selectors(main="div#x"))
    assert [d.markdown for d in result.documents] == ["Gewollt"]
    with pytest.raises(ValueError, match="selector"):
        Selectors(main="div:hover")


def test_empty_main_content_reported():
    html = b"<html><body><div class='other'><p>Text</p></div></body></html>"
    result = extract(html, HUB_URL)
    assert result.documents == () and result.pdfs == ()
    assert result.problem == "main content selector '.main-column' matched nothing"

    result = _inline("<nav>nur Menü</nav><script>x</script>")
    assert result.documents == ()
    assert result.problem == "main content selector '.main-column' matched no text"

    for body in (b"", b"   "):
        result = extract(body, HUB_URL)
        assert result.documents == () and result.problem is not None


def test_hidden_elements_are_dropped():
    result = _inline(
        "<p>Sichtbar</p><div style='color: red; Display : None'>A</div><p hidden>B</p><span aria-hidden='true'>C</span>"
        "<div style='VISIBILITY:hidden'><a href='/h.pdf'>D</a></div><p style='display:block'>Auch sichtbar</p>"
    )
    assert result.documents[0].markdown == "Sichtbar\n\nAuch sichtbar"
    assert result.pdfs == ()


def test_element_that_is_chrome_and_hidden_is_dropped_once():
    result = _inline(
        "<p>Text</p><nav hidden>A</nav><aside aria-hidden='true'>B</aside><form style='display:none'>C</form>"
        "<script hidden>D</script><svg aria-hidden='true'><path/></svg>"
        "<nav><p hidden>E</p><footer>F</footer></nav><div hidden><nav>G</nav><p hidden>H</p></div>"
    )
    assert result.documents[0].markdown == "Text"


def test_hidden_markers_match_in_any_case_and_whitespace():
    result = _inline(
        "<p>Sichtbar</p><span aria-hidden='TRUE'>A</span><div style='color:red;\n\tdisplay:\r\n none'>B</div>"
        "<div style='Visibility :\thidden'>C</div>"
    )
    assert result.documents[0].markdown == "Sichtbar"


def test_a_main_element_inside_a_hidden_ancestor_is_skipped_for_the_visible_one():
    page = (
        "<html><body><div style='display:none'><div class='main-column'><p>Versteckt</p></div></div>"
        "<div class='main-column'><p>Sichtbar</p></div></body></html>"
    )
    assert extract(page.encode(), HUB_URL).documents[0].markdown == "Sichtbar"
    own = page.replace("<div class='main-column'><p>Sichtbar", "<div class='main-column' hidden><p>Sichtbar")
    assert extract(own.replace("display:none", "display:block").encode(), HUB_URL).documents[0].markdown == "Versteckt"


def test_comments_noscript_and_template_never_reach_the_markdown():
    result = _inline(
        "<p>Text</p><!-- Kommentar --><noscript>Ohne Skript</noscript><template><p>Vorlage</p></template><?pi x?>"
    )
    assert result.documents[0].markdown == "Text"


def test_pdf_suffix_ignores_path_parameters():
    result = _inline("<a href='/x.pdf;jsessionid=ABC?a=1'>1</a><a href='/y.jsp;a=b.pdf'>2</a>")
    assert [p.url for p in result.pdfs] == [f"{FACULTY}/x.pdf;jsessionid=ABC?a=1"]


@pytest.mark.parametrize(
    ("head", "encoded", "content_type"),
    [
        ("", "utf-8", ""),  # no declaration: UTF-8
        ("", "utf-8", "text/html"),
        ('<meta charset="utf-8">', "utf-8", ""),
        ('<meta charset="iso-8859-1">', "latin-1", ""),
        ('<meta http-equiv="Content-Type" content="text/html; charset=windows-1252">', "cp1252", ""),
        ("", "latin-1", "text/html; charset=ISO-8859-1"),
        ('<meta charset="utf-8">', "latin-1", 'text/html; charset="latin1"'),  # the HTTP header wins
        ('<meta charset="nonsense">', "utf-8", "text/html; charset=nonsense"),  # unknown names count as none
        ("", "cp1252", ""),  # not UTF-8: Windows-1252
    ],
)
def test_text_is_decoded_in_the_declared_charset(head, encoded, content_type):
    page = f"<html><head>{head}</head><body><div class='main-column'><h1>Prüfungsordnung für Übungen €</h1></div></body></html>"
    text = page.replace("€", "") if encoded == "latin-1" else page
    result = extract(text.encode(encoded), HUB_URL, content_type=content_type)
    assert result.documents[0].markdown == "# Prüfungsordnung für Übungen" + ("" if encoded == "latin-1" else " €")


def test_alternatives_of_the_main_selector_are_tried_in_the_order_written():
    page = (
        b"<html><body><main><p>Rahmen</p><div class='main-column'><p>Inhalt</p></div></main></body></html>",
        b"<html><body><main><p>Nur Rahmen</p></main></body></html>",
    )
    url = f"{FACULTY}/a/page.de.jsp"
    texts = [
        extract(body, url, Selectors(main=main)).documents[0].markdown
        for body, main in (
            (page[0], ".main-column, main"),
            (page[0], "main, .main-column"),
            (page[1], ".main-column, main"),
        )
    ]
    assert texts == ["Inhalt", "Rahmen\n\nInhalt", "Nur Rahmen"]
