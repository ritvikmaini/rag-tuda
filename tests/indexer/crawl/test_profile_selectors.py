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
