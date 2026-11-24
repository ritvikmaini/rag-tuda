"""Metadata rules: the fields come from URL folders, section ids and headings, link titles and file names, the NULL rules
fail closed for the programme regulations, and a hub's copy of a university-wide regulation yields to its original."""

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rag_core.settings import load_profile
from rag_core.settings.profile import PdfTwin, Roles
from rag_testing import serve
from rag_testing.minisite import pages

from indexer.crawl.extract import Extraction, Link, PageDocument, Selectors, extract
from indexer.crawl.metadata import Candidate, page_entries, pdf_metadata, select_documents, twin_keys

TU = Path(__file__).resolve().parents[3] / "profiles" / "tu-informatik"
ROLES = load_profile(Roles, TU / "roles.yaml")
SOURCES_TWINS = [PdfTwin(de="https://a.example/media/apb.pdf", en="https://a.example/media/apb-english.pdf")]
FB20 = "https://www.informatik.tu-darmstadt.de"
ORDNUNGEN = f"{FB20}/studium_fb20/im_studium/formulare_und_dokumente/ordnungen"
MEDIA = f"{FB20}/media/informatik/fb20_studium/formulare_und_dokumente/ordnungen"
BSC_HUB = f"{ORDNUNGEN}/ordnungen_bsc_informatik.de.jsp"
MSC_HUB = f"{ORDNUNGEN}/ordnungen_msc_informatik.de.jsp"
INTERN_APB = "https://www.intern.tu-darmstadt.de/media/dezernat_ii/ordnungen/apb.pdf"


def _hub(url=BSC_HUB, *sections):
    docs = tuple(PageDocument(i, h, f"### {h}\n\ntext") for i, h in sections)
    return Extraction(url, "Ordnungen", docs, ())


PO_HUB = _hub(
    BSC_HUB, ("pruefungsordnung_2023", "Prüfungsordnung 2023"), ("pruefungsordnung_2015", "Prüfungsordnung 2015")
)


def _link(url, text="", title="", section_id=None):
    return Link(url, text, title, section_id)


def test_po_without_version_is_not_indexed_and_listed():
    undated = _link(f"{MEDIA}/ordnungen_bsc/Studiengangsordnung_Bachelor_Informatik.pdf", "Studiengangsordnung")
    dated = _link(f"{MEDIA}/ordnungen_bsc/x.pdf", "Studiengangsordnung", section_id="pruefungsordnung_2023")
    unknown_programme = _link(f"{FB20}/media/anderswo/Modulhandbuch.pdf", "Modulhandbuch")
    unlisted_year = _link(f"{MEDIA}/ordnungen_bsc/y.pdf", "Studiengangsordnung", section_id="pruefungsordnung_2010")
    hub = _hub(
        BSC_HUB, ("pruefungsordnung_2023", "Prüfungsordnung 2023"), ("pruefungsordnung_2010", "Prüfungsordnung 2010")
    )
    elsewhere = _hub("https://www.informatik.tu-darmstadt.de/other.de.jsp")
    candidates = [
        Candidate(link.url, f"h{n}", pdf_metadata(link, page, ROLES))
        for n, (link, page) in enumerate(
            ((undated, hub), (dated, hub), (unknown_programme, elsewhere), (unlisted_year, hub))
        )
    ]
    undated_meta = candidates[0].metadata
    assert undated_meta.regulation == "po" and undated_meta.programme == ("bsc-informatik",)
    assert undated_meta.po_version is None
    selection = select_documents(candidates, ROLES)
    assert [c.url for c in selection.kept] == [dated.url]
    assert selection.not_indexed == (
        (undated.url, ("po_version",)),
        (unknown_programme.url, ("programme", "po_version")),
        (unlisted_year.url, ("po_version",)),
    )


def test_apb_gets_null_degree_and_programme():
    original = pdf_metadata(
        _link(INTERN_APB, "Allgemeine Prüfungsbestimmungen (APB)"),
        _hub("https://www.intern.tu-darmstadt.de/x.de.jsp"),
        ROLES,
    )
    copy = pdf_metadata(
        _link(
            f"{MEDIA}/ordnungen_bsc/allgemeine_bestimmungen/apb.pdf",
            "APB (Kopie)",
            section_id="allg_pruefungsbest___dateiliste",
        ),
        PO_HUB,
        ROLES,
    )
    for meta in (original, copy):
        assert meta.regulation == "apb"
        assert (meta.degree, meta.programme, meta.po_version, meta.official_url) == (None, None, None, None)
    eso = pdf_metadata(
        _link("https://www.intern.tu-darmstadt.de/media/eso.pdf", "Einschreibeordnung (ESO)"),
        _hub("https://www.intern.tu-darmstadt.de/x.de.jsp"),
        ROLES,
    )
    assert eso.regulation == "eso" and eso.programme is None
    info = page_entries(
        Extraction("https://www.tu-darmstadt.de/a.de.jsp", "t", (PageDocument(None, None, "text"),), ()), ROLES
    )
    assert (info[0].metadata.regulation, info[0].metadata.degree, info[0].metadata.programme) == ("info", None, None)


def test_programme_degree_and_version_follow_the_hub_to_its_pdfs():
    msc_hub = _hub(MSC_HUB, ("pruefungsordnung_2023", "Prüfungsordnung 2023"))
    # the real MSc folder is `pruefungsordnung_9`: the version is the section's, never the folder number
    pdf = _link(
        f"{MEDIA}/ordnungen_msc/pruefungsordnung_9/Studiengangsordnung_Master.pdf",
        "Studiengangsordnung Master Informatik",
        "in Kraft seit 01.10.2022",
        "pruefungsordnung_2023",
    )
    meta = pdf_metadata(pdf, msc_hub, ROLES)
    assert (meta.regulation, meta.degree, meta.programme, meta.po_version) == (
        "po",
        ("master",),
        ("msc-informatik",),
        (2023,),
    )
    # the hub's English twin gives the same programme
    en = _hub(MSC_HUB.replace(".de.jsp", ".en.jsp"), ("pruefungsordnung_2023", "Examination regulations 2023"))
    assert pdf_metadata(pdf, en, ROLES).programme == ("msc-informatik",)
    # a date in a link title is not a version; the title can still carry one
    titled = _link(
        f"{MEDIA}/ordnungen_msc/pruefungsordnung_9/x.pdf", "Modulhandbuch", "PO 2015, in Kraft seit 01.10.2023"
    )
    assert pdf_metadata(titled, _hub(MSC_HUB), ROLES).po_version == (2015,)
    assert pdf_metadata(titled, _hub(MSC_HUB), ROLES).regulation == "mhb"
    # a programme the section cannot name falls back to the PDF's own media folder
    assert pdf_metadata(
        _link(f"{MEDIA}/ordnungen_cs/y.pdf", "Studiengangsordnung", section_id="pruefungsordnung_2023"),
        _hub("https://www.informatik.tu-darmstadt.de/z.de.jsp"),
        ROLES,
    ).programme == ("msc-computer-science",)


def test_hub_sections_without_a_po_version_are_one_document():
    hub = Extraction(
        BSC_HUB,
        "Ordnungen",
        (
            PageDocument(None, None, "# Ordnungen\n\nEinleitung"),
            PageDocument(
                "pruefungsordnung_2023", "Prüfungsordnung 2023", "### Prüfungsordnung 2023\n\nin Kraft seit 01.10.2023"
            ),
            PageDocument("pruefungsordnung_2015", "Prüfungsordnung 2015", "### Prüfungsordnung 2015\n\nalt"),
            PageDocument(
                "allg_pruefungsbest___dateiliste", "Allgemeine Prüfungsbestimmungen", "### Allgemeine\n\nKopie"
            ),
            PageDocument("kontakt", "Kontakt", "### Kontakt\n\nStudienbüro"),
        ),
        (),
    )
    entries = page_entries(hub, ROLES)
    assert [(e.section_id, e.metadata.regulation, e.metadata.po_version) for e in entries] == [
        (None, "info", None),
        ("pruefungsordnung_2023", "po", (2023,)),
        ("pruefungsordnung_2015", "po", (2015,)),
    ]
    rest = entries[0]
    assert "Einleitung" in rest.markdown and "Kopie" in rest.markdown and "Studienbüro" in rest.markdown
    assert rest.markdown.index("Einleitung") < rest.markdown.index("Kopie") < rest.markdown.index("Studienbüro")
    assert (rest.metadata.programme, rest.metadata.degree) == (("bsc-informatik",), ("bachelor",))
    assert entries[1].metadata.valid_from == date(2023, 10, 1)
    other = page_entries(
        _hub(
            "https://www.tu-darmstadt.de/p.de.jsp",
            ("pruefungsordnung_2020", "Prüfungsordnung 2020"),
            ("kontakt", "Kontakt"),
        ),
        ROLES,
    )
    assert [e.section_id for e in other] == [None]


def test_hub_apb_copy_yields_to_intern_original():
    copy_link = _link(f"{MEDIA}/ordnungen_bsc/allgemeine_bestimmungen/apb.pdf", "APB (Kopie)")
    original_link = _link(INTERN_APB, "APB (Original)", "in Kraft seit 13.01.2022")
    intern_hub = _hub("https://www.intern.tu-darmstadt.de/x.de.jsp")
    copy = Candidate(copy_link.url, "same", pdf_metadata(copy_link, PO_HUB, ROLES))
    original = Candidate(original_link.url, "same", pdf_metadata(original_link, intern_hub, ROLES))
    selection = select_documents([copy, original], ROLES)
    assert selection.kept == (original,) and selection.copies == (copy.url,)
    # another hash but the same regulation and the same entry-into-force date is still a copy
    other_hash = Candidate(
        copy_link.url, "other", pdf_metadata(_link(copy_link.url, "APB", "in Kraft seit 13.01.2022"), PO_HUB, ROLES)
    )
    assert select_documents([other_hash, original], ROLES).kept == (original,)
    # an older version of the hub is not a copy of the original
    older = Candidate(
        copy_link.url, "other", pdf_metadata(_link(copy_link.url, "APB", "in Kraft seit 01.01.2018"), PO_HUB, ROLES)
    )
    assert select_documents([older, original], ROLES).kept == (older, original)
    # without an original the copy stays; the same file on two hubs does not hide the original
    assert select_documents([copy], ROLES).kept == (copy,)
    assert select_documents([original, original], ROLES).kept == (original, original)


def test_valid_from_from_title_and_filename():
    hub = _hub(MSC_HUB, ("pruefungsordnung_2023", "Prüfungsordnung 2023"))
    base = f"{MEDIA}/ordnungen_msc/pruefungsordnung_9"
    titled = pdf_metadata(
        _link(f"{base}/a.pdf", "Studiengangsordnung", "in Kraft seit 13.01.2022", "pruefungsordnung_2023"), hub, ROLES
    )
    assert titled.valid_from == date(2022, 1, 13)
    named = pdf_metadata(
        _link(f"{base}/2022_09_30_MHB_Master_Informatik.pdf", "Modulhandbuch", "", "pruefungsordnung_2023"), hub, ROLES
    )
    assert (named.valid_from, named.regulation, named.po_version) == (date(2022, 9, 30), "mhb", (2023,))
    both = pdf_metadata(
        _link(f"{base}/2022_09_30_MHB.pdf", "Modulhandbuch", "In Kraft seit 1.10.2023", "pruefungsordnung_2023"),
        hub,
        ROLES,
    )
    assert both.valid_from == date(2023, 10, 1)
    assert (
        pdf_metadata(
            _link(f"{base}/b.pdf", "Modulhandbuch", "in Kraft seit 31.02.2023", "pruefungsordnung_2023"), hub, ROLES
        ).valid_from
        is None
    )
    assert (
        pdf_metadata(_link(f"{base}/c.pdf", "Modulhandbuch", "", "pruefungsordnung_2023"), hub, ROLES).valid_from
        is None
    )
    page = page_entries(
        Extraction(
            "https://www.tu-darmstadt.de/a.de.jsp",
            "t",
            (PageDocument(None, None, "Die Ordnung ist in Kraft seit 05.04.2021."),),
            (),
        ),
        ROLES,
    )
    assert page[0].metadata.valid_from == date(2021, 4, 5)


def test_archived_only_for_archiv_label():
    hub = _hub(BSC_HUB, ("pruefungsordnung_2015", "Prüfungsordnung 2015"))
    pdf = f"{MEDIA}/ordnungen_bsc/z.pdf"
    labelled = pdf_metadata(
        _link(pdf, "Archiv: Studiengangsordnung 2015", section_id="pruefungsordnung_2015"), hub, ROLES
    )
    assert labelled.archived is True
    in_title = pdf_metadata(
        _link(pdf, "Studiengangsordnung 2015", "Archiv: alte Fassung", "pruefungsordnung_2015"), hub, ROLES
    )
    assert in_title.archived is True
    for text in ("Studiengangsordnung 2015 (für ältere Jahrgänge)", "Archivierte Fassung", "Im Archiv gefunden"):
        assert pdf_metadata(_link(pdf, text, section_id="pruefungsordnung_2015"), hub, ROLES).archived is False
    archived_page = Extraction(BSC_HUB, "Archiv: Ältere Ordnungen", (PageDocument(None, None, "text"),), ())
    assert page_entries(archived_page, ROLES)[0].metadata.archived is True
    assert page_entries(hub, ROLES)[0].metadata.archived is False


def test_twin_key_is_section_level():
    de = page_entries(PO_HUB, ROLES)
    en_hub = Extraction(BSC_HUB.replace(".de.jsp", ".en.jsp"), "Ordnungen", PO_HUB.documents, ())
    en = page_entries(en_hub, ROLES)
    assert [e.metadata.twin_key for e in de] == [
        f"{ORDNUNGEN}/ordnungen_bsc_informatik#pruefungsordnung_2023",
        f"{ORDNUNGEN}/ordnungen_bsc_informatik#pruefungsordnung_2015",
    ]
    assert [e.metadata.twin_key for e in en] == [e.metadata.twin_key for e in de]
    assert (de[0].metadata.lang, en[0].metadata.lang) == ("de", "en")
    rest = page_entries(_hub(BSC_HUB.replace("ordnungen_bsc_informatik", "index"), ("kontakt", "Kontakt")), ROLES)
    assert rest[0].metadata.twin_key == f"{ORDNUNGEN}/index#kontakt"
    joined = page_entries(_hub(BSC_HUB.replace("ordnungen_bsc_informatik", "index"), ("a", "A"), ("b", "B")), ROLES)
    assert joined[0].metadata.twin_key == f"{ORDNUNGEN}/index"
    assert (
        page_entries(
            Extraction("https://digitales-rathaus.darmstadt.de/x", "t", (PageDocument(None, None, "a"),), ()), ROLES
        )[0].metadata.twin_key
        is None
    )
    # PDFs: twins only from the configured pair list
    keys = twin_keys(SOURCES_TWINS)
    link = _link("https://a.example/media/apb-english.pdf", "APB (English)")
    meta = pdf_metadata(link, _hub("https://a.example/x.de.jsp"), ROLES, keys)
    assert (meta.twin_key, meta.lang) == ("https://a.example/media/apb.pdf", "en")
    assert (
        pdf_metadata(
            _link("https://a.example/media/apb.pdf", "APB"), _hub("https://a.example/x.de.jsp"), ROLES, keys
        ).twin_key
        == "https://a.example/media/apb.pdf"
    )
    assert (
        pdf_metadata(
            _link("https://a.example/media/other.pdf", "APB"), _hub("https://a.example/x.de.jsp"), ROLES, keys
        ).twin_key
        is None
    )


def test_official_url_from_roles():
    hub = _hub(
        BSC_HUB, ("pruefungsordnung_2023", "Prüfungsordnung 2023"), ("pruefungsordnung_2015", "Prüfungsordnung 2015")
    )
    sb = ROLES.programmes["bsc-informatik"].po_versions[0].official_url
    assert sb and sb.endswith("SB_2023_II.pdf")
    po = pdf_metadata(
        _link(f"{MEDIA}/ordnungen_bsc/a.pdf", "Studiengangsordnung", section_id="pruefungsordnung_2023"), hub, ROLES
    )
    assert po.official_url == sb
    # no publication stated for the version: NULL
    assert (
        pdf_metadata(
            _link(f"{MEDIA}/ordnungen_bsc/b.pdf", "Studiengangsordnung", section_id="pruefungsordnung_2015"), hub, ROLES
        ).official_url
        is None
    )
    assert page_entries(hub, ROLES)[0].metadata.official_url == sb
    assert page_entries(hub, ROLES)[1].metadata.official_url is None


def test_promotion_documents_carry_their_degree_only():
    link = _link(
        f"{FB20}/media/informatik/fb20_dekanat/dokumente_promotion/Besondere_Bestimmungen_FB20.pdf",
        "Besondere Bestimmungen",
    )
    meta = pdf_metadata(link, _hub(f"{FB20}/forschung_fb20/wissenschaftliche_karriere/promotion/index.de.jsp"), ROLES)
    assert (meta.regulation, meta.degree, meta.programme, meta.po_version) == ("promotion", ("promotion",), None, None)
    assert select_documents([Candidate(link.url, "h", meta)], ROLES).not_indexed == ()


def test_the_mini_site_hub_carries_programme_and_version_to_its_pdf():
    origin = "https://faculty.test"
    site = TestClient(serve.minisite_app("fb20", "http://intern.test"), base_url=origin)
    roles = Roles.model_validate_json(ROLES.model_dump_json().replace(FB20, origin))
    hub = extract(site.get(f"{pages.BSC_HUB}.de.jsp").content, f"{origin}{pages.BSC_HUB}.de.jsp", Selectors())
    entries = page_entries(hub, roles)
    assert [(e.section_id, e.metadata.regulation, e.metadata.po_version) for e in entries[1:]] == [
        ("pruefungsordnung_2023", "po", (2023,)),
        ("pruefungsordnung_2015", "po", (2015,)),
    ]
    metas = {link.url: pdf_metadata(link, hub, roles) for link in hub.pdfs}
    po = metas[f"{origin}{pages.PO_2023_PDF}"]
    assert (po.regulation, po.degree, po.programme, po.po_version, po.valid_from) == (
        "po",
        ("bachelor",),
        ("bsc-informatik",),
        (2023,),
        date(2023, 10, 1),
    )
    assert po.official_url
    for url in (f"{origin}{pages.APB_COPY_PDF}", f"http://intern.test{pages.APB_ORIGINAL_PDF}"):
        assert (metas[url].regulation, metas[url].programme, metas[url].valid_from) == ("apb", None, date(2022, 1, 13))


@pytest.mark.parametrize("field", ["degree", "programme", "po_version", "twin_key", "official_url", "valid_from"])
def test_unknown_values_are_null_never_empty(field):
    meta = pdf_metadata(
        _link("https://www.tu-darmstadt.de/media/info.pdf", "Merkblatt"),
        _hub("https://www.tu-darmstadt.de/x.de.jsp"),
        ROLES,
    )
    assert getattr(meta, field) is None


def test_a_po_change_form_is_a_form_not_a_po_without_a_version():
    hub = _hub(BSC_HUB, ("wechsel", "Wechsel der Studiengangsordnung"))
    base = f"{MEDIA}/ordnungen_bsc"
    by_text = _link(f"{base}/Erklaerung_-_PO-Wechsel.pdf", "Erklärung zum Wechsel", section_id="wechsel")
    by_name = _link(f"{base}/Erklaerung_-_PO-Wechsel.pdf", "", section_id="wechsel")
    by_heading_only = _link(f"{base}/wechsel.pdf", "", section_id="wechsel")
    metas = [pdf_metadata(link, hub, ROLES) for link in (by_text, by_name, by_heading_only)]
    assert [m.regulation for m in metas] == ["form", "form", "info"]  # "Ordnung" in a heading is no PO
    assert all(m.programme == ("bsc-informatik",) and m.po_version is None for m in metas)
    selection = select_documents([Candidate(f"u{n}", f"h{n}", m) for n, m in enumerate(metas)], ROLES)
    assert len(selection.kept) == 3 and selection.not_indexed == ()


def test_an_unlabelled_pdf_in_a_po_year_section_is_still_a_po():
    hub = _hub(BSC_HUB, ("pruefungsordnung_2023", "Prüfungsordnung 2023"))
    meta = pdf_metadata(_link(f"{MEDIA}/ordnungen_bsc/x.pdf", "", section_id="pruefungsordnung_2023"), hub, ROLES)
    assert (meta.regulation, meta.po_version) == ("po", (2023,))


def test_a_modulhandbuch_naming_the_apb_stays_a_modulhandbuch():
    hub = _hub(BSC_HUB, ("pruefungsordnung_2023", "Prüfungsordnung 2023"))
    link = _link(
        f"{MEDIA}/ordnungen_bsc/m.pdf", "Modulhandbuch gemäß APB", "APB und Einschreibeordnung", "pruefungsordnung_2023"
    )
    meta = pdf_metadata(link, hub, ROLES)
    assert (meta.regulation, meta.programme, meta.po_version) == ("mhb", ("bsc-informatik",), (2023,))
    titled = _link(f"{MEDIA}/ordnungen_bsc/n.pdf", "Hinweise", "Modulhandbuch", "pruefungsordnung_2023")
    assert pdf_metadata(titled, hub, ROLES).regulation == "mhb"
    assert (
        pdf_metadata(
            _link("https://x.example/eso.pdf", "Einschreibeordnung (ESO)"), _hub("https://x.example/a.de.jsp"), ROLES
        ).regulation
        == "eso"
    )


def test_unknown_hashes_never_make_a_copy():
    copy_url = f"{MEDIA}/ordnungen_bsc/allgemeine_bestimmungen/apb.pdf"
    intern = _hub("https://www.intern.tu-darmstadt.de/x.de.jsp")
    copy = Candidate(copy_url, None, pdf_metadata(_link(copy_url, "APB", "in Kraft seit 01.01.2018"), PO_HUB, ROLES))
    original = Candidate(
        INTERN_APB, None, pdf_metadata(_link(INTERN_APB, "APB", "in Kraft seit 13.01.2022"), intern, ROLES)
    )
    assert select_documents([copy, original], ROLES).kept == (copy, original)
    undated = Candidate(copy_url, None, pdf_metadata(_link(copy_url, "APB"), PO_HUB, ROLES))
    undated_original = Candidate(INTERN_APB, None, pdf_metadata(_link(INTERN_APB, "APB"), intern, ROLES))
    assert select_documents([undated, undated_original], ROLES).kept == (undated, undated_original)
    # a known equal hash is a copy; so is the same date in the same language, but not in another language
    assert select_documents(
        [Candidate(copy_url, "h", undated.metadata), Candidate(INTERN_APB, "h", undated_original.metadata)], ROLES
    ).copies == (copy_url,)
    english = Candidate(
        INTERN_APB,
        None,
        pdf_metadata(
            _link("https://www.intern.tu-darmstadt.de/apb-english.pdf", "APB", "in Kraft seit 13.01.2022"),
            intern,
            ROLES,
        ),
    )
    dated_copy = Candidate(
        copy_url, None, pdf_metadata(_link(copy_url, "APB", "in Kraft seit 13.01.2022"), PO_HUB, ROLES)
    )
    assert select_documents([dated_copy, english], ROLES).copies == ()


def test_the_joined_rest_has_no_valid_from_of_its_own():
    hub = Extraction(
        BSC_HUB,
        "Ordnungen",
        (
            PageDocument("a", "A", "### A\n\nin Kraft seit 13.01.2022"),
            PageDocument("b", "B", "### B\n\nText"),
            PageDocument("pruefungsordnung_2023", "Prüfungsordnung 2023", "in Kraft seit 01.10.2023"),
        ),
        (),
    )
    entries = page_entries(hub, ROLES)
    assert [e.section_id for e in entries] == [None, "pruefungsordnung_2023"]
    assert entries[0].metadata.valid_from is None and entries[1].metadata.valid_from == date(2023, 10, 1)
    single = page_entries(_hub(BSC_HUB, ("a", "A")), ROLES)  # one remaining section keeps its own
    assert single[0].section_id == "a"


def test_language_of_a_pdf_name():
    hub = _hub("https://a.example/x.de.jsp")
    names = (("apb.en.pdf", "en"), ("apb-english.pdf", "en"), ("apb.de.pdf", "de"), ("apb.pdf", "de"))
    for name, lang in (*names, ("Promotionsordnung-en.pdf", "en"), ("ESO_englisch.pdf", "en"), ("studien.pdf", "de")):
        assert pdf_metadata(_link(f"https://a.example/media/{name}", "APB"), hub, ROLES).lang == lang


def test_implementing_provisions_of_an_unconfigured_po_year_are_listed_not_indexed():
    hub = _hub(BSC_HUB, ("pruefungsord__209_dateiliste", None))

    def provisions(year, text="Ausführungsbestimmungen (AB) {} zu den APB der TU Darmstadt"):
        url = f"{MEDIA}/ordnungen_bsc/pruefungsordnung_x/Ausfuehrungsbestimmungen_AB_{year}_zu_den_APB.pdf"
        link = _link(url, text.format(year), section_id="pruefungsord__209_dateiliste")
        return Candidate(url, None, pdf_metadata(link, hub, ROLES))

    old, configured = provisions(2011), provisions(2015)
    current = Candidate(
        INTERN_APB,
        None,
        pdf_metadata(_link(INTERN_APB, "Allgemeine Prüfungsbestimmungen (APB), 7. Novelle"), hub, ROLES),
    )
    selection = select_documents([old, configured, current], ROLES)
    assert selection.not_indexed == ((old.url, ("po_version",)),)
    assert [c.url for c in selection.kept] == [configured.url, INTERN_APB]
    assert old.metadata.regulation == "apb" and old.metadata.programme is None  # it would apply to everyone
