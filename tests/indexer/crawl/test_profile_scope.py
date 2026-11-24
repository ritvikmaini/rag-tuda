"""The faculty host's scope in the TU profile: the study area of the three configured programmes is in, the other
programmes' hubs and pages and the rest of the faculty's site are out. Nothing is fetched."""

from pathlib import Path

import pytest
from rag_core.settings import load_profile
from rag_core.settings.profile import Sources

from indexer.crawl.discover import Source, in_scope

TU = Path(__file__).resolve().parents[3] / "profiles" / "tu-informatik"
HOST = "www.informatik.tu-darmstadt.de"
STUDY = "/studium_fb20/im_studium"
ORDNUNGEN = f"{STUDY}/formulare_und_dokumente/ordnungen"
PROGRAMMES = f"{STUDY}/studiengaenge_liste"

IN = [
    f"{ORDNUNGEN}/index.de.jsp",
    f"{ORDNUNGEN}/ordnungen_bsc_informatik.de.jsp",
    f"{ORDNUNGEN}/ordnungen_bsc_informatik.en.jsp",
    f"{ORDNUNGEN}/ordnungen_msc_informatik.de.jsp",
    f"{ORDNUNGEN}/ordnungen_cs.de.jsp",
    f"{PROGRAMMES}/informatik_bsc.de.jsp",
    f"{PROGRAMMES}/informatik_bsc_alte_ordnungen.de.jsp",
    f"{PROGRAMMES}/informatik_msc.de.jsp",
    f"{PROGRAMMES}/msc_computer_science.de.jsp",
    f"{STUDY}/studienbuero/faq_studienbuero/index.de.jsp",
    "/studium_fb20/faqs_1/index.de.jsp",
    "/studium_fb20/wissenssammlung_studium_details_16960.de.jsp",
    "/forschung_fb20/wissenschaftliche_karriere/promotion/index.de.jsp",
    "/media/informatik/fb20_studium/formulare_und_dokumente/ordnungen/ordnungen_cs/a.pdf",
]
OUT = [
    f"{ORDNUNGEN}/ordnungen_aim.de.jsp",
    f"{ORDNUNGEN}/ordnungen_asur.de.jsp",
    f"{ORDNUNGEN}/ordnungen_autonome_systeme.de.jsp",
    f"{ORDNUNGEN}/ordnungen_internet_und_webbasierte.de.jsp",
    f"{ORDNUNGEN}/ordnungen_it_sicherheit.de.jsp",
    f"{ORDNUNGEN}/ordnungen_mdss.de.jsp",
    f"{ORDNUNGEN}/ordnungen_visual_computing.de.jsp",
    f"{ORDNUNGEN}/itsecurity_msc.de.jsp",
    f"{ORDNUNGEN}/informatik_lehramt_an_gymnasien_3.de.jsp",
    f"{PROGRAMMES}/aim_msc.de.jsp",
    f"{PROGRAMMES}/asur_msc.de.jsp",
    f"{PROGRAMMES}/autonome_systeme_msc.de.jsp",
    f"{PROGRAMMES}/entrace_examination_mdss.de.jsp",
    f"{PROGRAMMES}/informatik_lehramt_an_gymnasien_1.de.jsp",
    f"{PROGRAMMES}/internet__und_web_basierte_systeme.de.jsp",
    f"{PROGRAMMES}/itsecurity_msc.de.jsp",
    f"{PROGRAMMES}/visual_computing_fb20.de.jsp",
    "/fb20/aktuelles_fb20/news_details_1.de.jsp",
    "/fb20/intern_1/wissenssammlung_details_16064.de.jsp",
    "/forschung_fb20/index.de.jsp",
]


@pytest.fixture(scope="module")
def source() -> Source:
    entry = load_profile(Sources, TU / "sources.yaml").hosts[HOST]
    return Source(HOST, prefixes=tuple(entry.prefixes), exclude=tuple(entry.exclude))


@pytest.mark.parametrize("path", IN)
def test_the_study_area_of_the_configured_programmes_is_in_scope(source: Source, path: str):
    assert in_scope(source, f"https://{HOST}{path}")


@pytest.mark.parametrize("path", OUT)
def test_other_programmes_and_the_rest_of_the_site_are_out_of_scope(source: Source, path: str):
    assert not in_scope(source, f"https://{HOST}{path}")
