"""The faculty host's scope in the TU profile: the study area of the three configured programmes is in, the other
programmes' hubs and pages and the rest of the faculty's site are out. The full set also takes the PDFs of the TU-wide
study pages and of the Landkreis's Ausländerwesen pages from one media folder each. Nothing is fetched."""

import dataclasses
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from rag_core.settings import load_profile
from rag_core.settings.profile import Roles, Sources

from indexer.connectors.crawl_setup import active_hosts
from indexer.crawl.discover import Source, in_scope
from indexer.crawl.extract import Extraction, Link
from indexer.crawl.metadata import Candidate, pdf_metadata, select_documents, twin_keys
from indexer.crawl.fetch import Fetcher, Skipped

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


def test_pdf_twins_pair_the_english_pdfs_with_their_german_ones():
    sources = load_profile(Sources, TU / "sources.yaml")
    keys = twin_keys(sources.pdf_twins)
    assert len(sources.pdf_twins) == 6
    for pair in sources.pdf_twins:
        assert pair.de != pair.en and keys[pair.en] == keys[pair.de] == pair.de
        assert urlsplit(pair.de).hostname in sources.hosts and urlsplit(pair.en).hostname in sources.hosts


TU_HOST = "www.tu-darmstadt.de"
LADADI = "www.ladadi.de"
# "02_studium_medien" is the study pages' folder; its neighbour "01_studieninteressierte_medien" (prospective students)
# is a different folder and stays out
TU_PDFS = "/media/daa_responsives_design/02_studium_medien/"
LADADI_PDFS = "/medien/pdfs/fb-510-auslaenderwesen-visum/"
FULL_PDFS = [
    (
        TU_HOST,
        f"{TU_PDFS}04_auslandsaufenthalte_medien/01_austauschprogramme_outgoing/checkliste_erasmus_outgoings.pdf",
    ),
    (TU_HOST, f"{TU_PDFS}06_internationale_studierende_medien/01_ihr_aufenthalt/vor_der_ankunft_checkliste.pdf"),
    (TU_HOST, f"{TU_PDFS}Wahl_FB__alle_StG.pdf"),
    (LADADI, f"{LADADI_PDFS}2025-wohnbescheinigung.pdf?cid=7q4"),
    (LADADI, f"{LADADI_PDFS}zusatzblatt-a-16d-aufenthg.pdf"),
]
NOT_FULL_PDFS = [
    (TU_HOST, "/media/dezernat_ii/studierendensekretariat/rueckmeldung.pdf"),
    (TU_HOST, "/media/daa_responsives_design/01_studieninteressierte_medien/x.pdf"),  # the folder beside TU_PDFS
    (TU_HOST, "/media/dezernat_ii/ordnungen/apb.pdf"),
    (LADADI, "/medien/pdfs/fb-300-bauen/antrag.pdf"),
    (LADADI, f"{TU_PDFS}a.pdf"),  # a folder of the other host
    (TU_HOST, f"{LADADI_PDFS}a.pdf"),
]


def _source(host: str, full: bool) -> Source | None:
    found = active_hosts(load_profile(Sources, TU / "sources.yaml"), full)
    return found[host].source if host in found else None


@pytest.mark.parametrize(("host", "path"), FULL_PDFS)
def test_the_full_set_takes_the_pdfs_of_the_two_media_folders(host: str, path: str):
    source = _source(host, full=True)
    assert source and in_scope(source, f"https://{host}{path}")
    assert not (core := _source(host, full=False)) or not in_scope(core, f"https://{host}{path}")


@pytest.mark.parametrize(("host", "path"), NOT_FULL_PDFS)
def test_no_other_media_folder_is_in_the_full_set(host: str, path: str):
    source = _source(host, full=True)
    assert source and not in_scope(source, f"https://{host}{path}")


def test_the_core_set_has_no_media_prefix_of_the_tu_wide_host_and_not_the_landkreis():
    assert _source(TU_HOST, full=False).prefixes == () and _source(LADADI, full=False) is None
    assert _source(HOST, full=False) == _source(HOST, full=True)


def test_exclusions_and_dot_segments_still_apply_in_the_media_folders():
    source = _source(TU_HOST, full=True)
    narrowed = dataclasses.replace(source, exclude=(f"{TU_PDFS}stipendien_und_preise/",))
    assert not in_scope(narrowed, f"https://{TU_HOST}{TU_PDFS}stipendien_und_preise/freudenberg/Karte.pdf")
    assert in_scope(narrowed, f"https://{TU_HOST}{TU_PDFS}Wahl_FB__alle_StG.pdf")
    assert not in_scope(source, f"https://{TU_HOST}{TU_PDFS}../../dezernat_ii/x.pdf")
    assert not in_scope(source, f"http://{TU_HOST}:8080{TU_PDFS}x.pdf")


def test_outdated_and_expired_pdfs_are_out_of_the_full_set_and_their_neighbours_stay():
    internships = f"{TU_PDFS}02_studierende_medien/04_auslandsaufenthalte_medien/03_praktikum_erasmus_outbound_medien/"
    tu, landkreis = _source(TU_HOST, full=True), _source(LADADI, full=True)
    assert not in_scope(
        tu, f"https://{TU_HOST}{internships}stellenausschreibungen_2/sdu_university_of_southern_denmark_2026.pdf"
    )
    assert not in_scope(
        landkreis, f"https://{LADADI}{LADADI_PDFS}antrag-verlaengerung-visum-stand-januar-2006.pdf?cid=38p"
    )
    assert in_scope(tu, f"https://{TU_HOST}{internships}Infoblatt_Zusatzfoerderung.pdf")
    assert in_scope(landkreis, f"https://{LADADI}{LADADI_PDFS}2025-wohnbescheinigung.pdf")


def test_robots_rules_of_the_host_still_apply_to_a_pdf_of_the_media_folder():
    def robots_txt(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200, headers={"content-type": "text/plain"}, text="User-agent: *\nDisallow: /*view=\n"
            )
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4")

    fetcher = Fetcher(
        (TU_HOST,),
        contact="dummy@example.invalid",
        client=httpx.Client(transport=httpx.MockTransport(robots_txt)),
        interval=0,
    )
    url = f"https://{TU_HOST}{TU_PDFS}Wahl_FB__alle_StG.pdf"
    assert fetcher.get(f"{url}?view=print") == Skipped(f"{url}?view=print", "robots", f"{url}?view=print")
    assert not isinstance(fetcher.get(url), Skipped)


def _typed(text: str, name: str = "x.pdf"):
    """The metadata of a PDF of the TU media folder, linked from a page that no programme owns."""
    page = Extraction(f"https://{TU_HOST}/studieren/studierende_tu/a.de.jsp", "A", (), ())
    roles = load_profile(Roles, TU / "roles.yaml")
    url = f"https://{TU_HOST}{TU_PDFS}{name}"
    return roles, Candidate(url, "h", pdf_metadata(Link(url, text, "", None), page, roles))


@pytest.mark.parametrize(
    ("text", "name", "regulation"),
    [
        ("Prüfungsordnung des Fachbereichs 3", "x.pdf", "po"),
        ("Studienordnung", "x.pdf", "po"),
        ("Modulhandbuch", "x.pdf", "mhb"),
        ("Studien- und Prüfungsplan", "x.pdf", "spp"),
    ],
)
def test_a_typed_regulation_in_the_media_folder_fails_closed(text: str, name: str, regulation: str):
    roles, candidate = _typed(text, name)
    assert candidate.metadata.regulation == regulation
    selection = select_documents([candidate], roles)
    assert not selection.kept
    assert [url for url, _ in selection.not_indexed] == [candidate.url]  # lacking programme and PO version


def test_a_regulation_word_only_in_the_file_name_also_fails_closed():
    roles, candidate = _typed("Download (PDF-Datei)", "Pruefungsordnung_FB3.pdf")
    assert candidate.metadata.regulation == "po"
    assert not select_documents([candidate], roles).kept
