"""The TU profile's selectors run on small pages built like each host's own markup (invented text, the structure of the
real pages): the main content is found and the page's chrome is left out. Nothing is fetched."""

from pathlib import Path

import pytest
from rag_core.settings import load_profile
from rag_core.settings.profile import Sources

from indexer.connectors.crawl_setup import active_hosts
from indexer.crawl.extract import extract

TU = Path(__file__).resolve().parents[3] / "profiles" / "tu-informatik"
RATHAUS = "digitales-rathaus.darmstadt.de"
FACULTY = "www.informatik.tu-darmstadt.de"


def selectors(host: str):
    return active_hosts(load_profile(Sources, TU / "sources.yaml"), full=True)[host].selectors


def rathaus_page(frame_id: str) -> bytes:
    return f"""<!DOCTYPE html><html lang="de"><head><title>Anmeldung | Digitales Rathaus</title></head>
    <body class="content-page"><div id="container">
    <header id="mainHead"><nav class="navbar"><a href="/">Startseite</a></nav></header>
    <div class="container page-container"><div class="columns"><div class="column">
    <div id="{frame_id}" class="frame frame-default frame-type-list"><div class="tx-wda-vrathaus">
    <h1>Anmeldung des Wohnsitzes</h1><h2>Kurzinfo</h2><p>Melden Sie Ihren Wohnsitz innerhalb von zwei Wochen an.</p>
    </div></div></div></div></div>
    <footer class="mainFooter"><p>Impressum und Kontakt</p></footer></div></body></html>""".encode()


@pytest.mark.parametrize("frame_id", ["c1", "c2"])
def test_rathaus_content_is_read_from_the_page_container(frame_id):
    page = extract(rathaus_page(frame_id), f"https://{RATHAUS}/lebensbereiche/anmeldung", selectors(RATHAUS))
    assert page.problem is None
    text = "\n".join(d.markdown for d in page.documents)
    assert "Melden Sie Ihren Wohnsitz" in text and "# Anmeldung des Wohnsitzes" in text
    assert "Startseite" not in text and "Impressum" not in text


def faculty_page(content: str) -> bytes:
    return f"""<!DOCTYPE html><html lang="de"><head><title>Seite</title></head><body><div id="app"><div id="site">
    <header id="site-header"><nav id="breadcrumb-navigation">Startseite</nav></header>
    {content}
    <footer id="site-footer">Datenschutz</footer></div></div></body></html>""".encode()


def test_faculty_hub_page_is_read_from_its_main_column():
    page = faculty_page(
        '<main id="site-main"><div class="cell-md-9"><section id="page-lead"><p>Einleitung</p></section>'
        '<div class="main-column"><section class="content-section" id="po_2023"><h3>Prüfungsordnung 2023</h3>'
        "<p>Text</p></section></div></div><div class='cell-md-3'><button>Kontakt</button></div></main>"
    )
    result = extract(page, f"https://{FACULTY}/studium_fb20/hub.de.jsp", selectors(FACULTY))
    assert [d.section_id for d in result.documents] == ["po_2023"]
    assert "Einleitung" not in result.documents[0].markdown


@pytest.mark.parametrize("wrapper", ['<div id="toc_content">{}</div>', "{}"])
def test_faculty_programme_page_without_a_main_column_is_read_from_main(wrapper):
    sections = '<section id="page-lead"><p>Einleitung</p></section>' + wrapper.format(
        '<section class="content-section" id="infobox"><h3>Auf einen Blick</h3><p>Regelstudienzeit</p></section>'
        '<section id="neuerungen"><p>Neuerungen</p></section>'
    )
    result = extract(
        faculty_page(f'<main id="site-main">{sections}</main>'),
        f"https://{FACULTY}/studium_fb20/p.de.jsp",
        selectors(FACULTY),
    )
    assert result.problem is None
    text = "\n".join(d.markdown for d in result.documents)
    assert all(word in text for word in ("Einleitung", "Regelstudienzeit", "Neuerungen"))
    assert "Startseite" not in text and "Datenschutz" not in text
