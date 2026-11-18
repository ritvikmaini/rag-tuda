"""The fixture mini-site through FastAPI's test client: every page is served, the two hosts keep their own files, the two PDFs
are valid and carry "§ n" references, and the copy of the general regulation equals its original."""

import pypdfium2
import pytest
from fastapi.testclient import TestClient

from rag_testing import serve
from rag_testing.minisite import pages
from rag_testing.minisite.pdf import make_pdf

INTERN = "http://intern.test"
FB20 = TestClient(serve.minisite_app("fb20", INTERN), base_url="http://faculty.test")
SECOND = TestClient(serve.minisite_app("intern", INTERN), base_url=INTERN)

FB20_PAGES = [
    (pages.BSC_HUB + ".de.jsp", 'id="pruefungsordnung_2023"', "Prüfungsordnung 2023"),
    (pages.BSC_HUB + ".de.jsp", 'id="pruefungsordnung_2015"', "ab dem Wintersemester 2023/24"),
    (pages.BSC_HUB + ".de.jsp", 'id="allg_pruefungsbest___dateiliste"', 'class="serif-h3"'),
    (pages.BSC_HUB + ".de.jsp", 'name="generator" content="FirstSpirit', "(PDF-Datei,"),
    (pages.BSC_HUB + ".en.jsp", "The content of this page is only available in German", "Prüfungsordnung 2023"),
    (pages.ORDNUNGEN + "/index.de.jsp", "ordnungen_msc_visual_computing.de.jsp", "ordnungen_bsc_informatik.de.jsp"),
    (pages.EXCLUDED_HUB + ".de.jsp", "Visual Computing", "nicht indexiert"),
    (pages.STUDIENBUERO + "/index.de.jsp", "Sprechstunde", "pruefungen/index.de.jsp?view=print"),
    (pages.STUDIENBUERO + "/index.en.jsp", "Office hours", 'lang="en"'),
    (pages.STUDIENBUERO + "/pruefungen/index.de.jsp", "Prüfungsanmeldung", 'lang="de"'),
    (pages.STUDIENBUERO + "/pruefungen/index.en.jsp", "Prüfungsanmeldung", 'lang="en"'),
    (pages.NOAI_PAGE, '<meta name="robots" content="noai">', "Hinweise"),
    (pages.LOGIN_PAGE, "Anmeldung", "Mitarbeitende"),
    ("/sitemap.de.xml", "<urlset", "http://faculty.test" + pages.NOAI_PAGE),
]


@pytest.mark.parametrize(("path", "first", "second"), FB20_PAGES)
def test_the_faculty_host_serves_each_page(path, first, second):
    resp = FB20.get(path)
    assert resp.status_code == 200
    assert first in resp.text and second in resp.text
    assert "@SELF@" not in resp.text and "@INTERN@" not in resp.text


def test_the_german_body_without_a_notice_is_the_other_kind_of_fallback_twin():
    assert "only available in German" not in FB20.get(pages.STUDIENBUERO + "/pruefungen/index.en.jsp").text


def test_robots_txt_carries_the_wildcard_and_query_rules_and_the_sitemap():
    resp = FB20.get("/robots.txt")
    assert resp.headers["content-type"].startswith("text/plain")
    lines = resp.text.splitlines()
    assert "User-agent: *" in lines and "Disallow: /*view=" in lines and "Disallow: /*.login" in lines
    assert "Sitemap: http://faculty.test/sitemap.de.xml" in lines
    assert SECOND.get("/robots.txt").text.splitlines() == ["User-agent: *", "Disallow: /*.login"]


def test_a_host_reached_through_a_route_names_itself_in_its_sitemap_and_robots_txt():
    routed = TestClient(serve.minisite_app("fb20", "https://intern.example", "https://faculty.example"))
    assert "Sitemap: https://faculty.example/sitemap.de.xml" in routed.get("/robots.txt").text.splitlines()
    assert "<loc>https://faculty.example" + pages.BSC_HUB + ".de.jsp</loc>" in routed.get("/sitemap.de.xml").text


def test_the_hub_links_the_copy_on_its_own_host_and_the_original_on_the_second():
    html = FB20.get(pages.BSC_HUB + ".de.jsp").text
    assert f'href="{pages.APB_COPY_PDF}"' in html and f'href="{INTERN}{pages.APB_ORIGINAL_PDF}"' in html
    assert pages.APB_ORIGINAL_PDF in SECOND.get(pages.INTERN_INDEX).text


def test_the_two_hosts_keep_their_own_files():
    assert FB20.get(pages.APB_ORIGINAL_PDF).status_code == 404
    assert SECOND.get(pages.APB_COPY_PDF).status_code == 404
    assert SECOND.get(pages.BSC_HUB + ".de.jsp").status_code == 404
    assert SECOND.get(pages.APB_ORIGINAL_PDF).status_code == 200


@pytest.mark.parametrize(
    "path", ["/nope", "/robots.txt/", pages.ORDNUNGEN + "/ordnungen_bsc_aim.de.jsp", "/media/x.pdf"]
)
def test_unknown_paths_are_404_on_both_hosts(path):
    assert FB20.get(path).status_code == 404 and SECOND.get(path).status_code == 404


def _text(content: bytes) -> list[str]:
    doc = pypdfium2.PdfDocument(content)
    return [doc[n].get_textpage().get_text_range() for n in range(len(doc))]


def test_the_study_regulation_is_a_two_page_pdf_with_paragraph_references_and_one_short_paragraph():
    resp = FB20.get(pages.PO_2023_PDF)
    assert resp.headers["content-type"] == "application/pdf" and resp.content.startswith(b"%PDF-")
    one, two = _text(resp.content)
    assert "§ 1 Geltungsbereich" in one and "§ 22 und § 23" in one and "§ 5 Bachelor-Thesis" in two
    short = one.split("§ 2 Studienziel")[1].split("§ 3")[0]
    assert len(short.split()) < 10


def test_the_regulation_copy_equals_its_original_and_names_paragraph_22():
    copy, original = FB20.get(pages.APB_COPY_PDF), SECOND.get(pages.APB_ORIGINAL_PDF)
    assert copy.content == original.content
    assert "§ 22 Wiederholung von Prüfungen" in _text(copy.content)[0]
    assert pages.PO_2023 != pages.APB


def test_the_pdfs_are_the_same_bytes_every_time():
    assert make_pdf("§ 1 Ein") == make_pdf("§ 1 Ein")
