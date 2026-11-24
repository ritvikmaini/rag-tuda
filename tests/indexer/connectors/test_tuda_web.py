"""The TU web connector against the fixture mini-site served in-process under the profile's real host names (https): the hosts
of the source set, the documents and PDFs with their metadata, the fail-closed selection, the PDF sync through the fetcher's
complete files, the crawl cache and resume, and the contact in the user agent. No request leaves the process."""

import json
import os
import re
import shutil
from collections import Counter
from datetime import date, datetime, timedelta
from importlib.metadata import entry_points
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pypdfium2 as pdfium
import pytest
import yaml
from fastapi.testclient import TestClient
from rag_core.domain.sources import CorruptStateError, PageItem, PdfItem
from rag_core.settings import load_profile
from rag_core.settings.profile import Profile, Sources
from rag_testing import serve
from rag_testing.minisite import pages

from indexer import build, cli, gates
from indexer.chunking.units import chunk_documents
from indexer.connectors.crawl_setup import active_hosts
from indexer.connectors.tuda_web import TudaWeb
from indexer.crawl.fetch import user_agent
from indexer.extract.documents import pdf_documents
from indexer.index import generations as gen
from indexer.index import writer
from indexer.settings import load_settings
from indexer.sync.registry import REGISTRY, local_path
from indexer.sync.safeguard import LIST_STATE

FB20 = "www.informatik.tu-darmstadt.de"
INTERN = "www.intern.tu-darmstadt.de"
LADADI = "www.ladadi.de"
HUB = f"https://{FB20}{pages.BSC_HUB}.de.jsp"
HUB_EN = f"https://{FB20}{pages.BSC_HUB}.en.jsp"
STUDIENBUERO = f"https://{FB20}{pages.STUDIENBUERO}/index.de.jsp"
PO_PDF = f"https://{FB20}{pages.PO_2023_PDF}"
APB_COPY = f"https://{FB20}{pages.APB_COPY_PDF}"
APB_ORIGINAL = f"https://{INTERN}{pages.APB_ORIGINAL_PDF}"
INTERN_INDEX = f"https://{INTERN}{pages.INTERN_INDEX}"
BUNDLE = f"https://{INTERN}/media/dezernat_ii/satzungsbeilagen/SB_2023_II.pdf"
LADADI_PAGE = f"https://{LADADI}/umwelt/auslaenderwesen/"
TU = Path(__file__).resolve().parents[3] / "profiles" / "tu-informatik"
TODAY = date(2026, 11, 20)
MAIN = {"main": ".main-column", "section": "section.content-section", "drop": None}
FACULTY = TestClient(serve.minisite_app("fb20", f"https://{INTERN}"), base_url=f"https://{FB20}")
UNIVERSITY = TestClient(serve.minisite_app("intern", f"https://{INTERN}"), base_url=f"https://{INTERN}")
LADADI_HTML = (
    '<html><head><title>Ausländerwesen</title></head><body><div class="main-column"><h1>Ausländerwesen</h1>'
    '<section class="content-section" id="landkreis"><h3>Landkreis</h3><p>Zuständig ist die Ausländerbehörde des '
    "Landkreises.</p></section></div></body></html>"
)


def sources_yaml(*, extra_seeds=(), extra_prefixes=()) -> dict:
    """The profile's three kinds of host in miniature, with the paths of the mini-site: the faculty (sitemap, link-following,
    English twins), the university's host (its APB under /media/dezernat_ii/ordnungen/) and a full-set host."""
    ordnungen = pages.ORDNUNGEN
    media = "/media/informatik/fb20_studium/formulare_und_dokumente/ordnungen/ordnungen_bsc/"
    return {
        "hosts": {
            FB20: {
                "set": "core",
                "seeds": [HUB, STUDIENBUERO, *extra_seeds],
                "sitemap": [f"https://{FB20}/sitemap.de.xml"],
                "prefixes": [f"{ordnungen}/ordnungen_bsc_informatik", f"{pages.STUDIENBUERO}/", media, *extra_prefixes],
                "exclude": [],
                "depth": 1,
                "twins": True,
                "selectors": MAIN,
            },
            INTERN: {
                "set": "core",
                "seeds": [INTERN_INDEX],
                "sitemap": [],
                "prefixes": [
                    "/verwaltung/dez_ii/hochschulrecht/gesetze_und_ordnungen/",
                    "/media/dezernat_ii/ordnungen/",
                ],
                "exclude": [],
                "depth": 1,
                "twins": False,
                "selectors": MAIN,
            },
            LADADI: {
                "set": "full",
                "seeds": [LADADI_PAGE],
                "sitemap": [],
                "prefixes": ["/umwelt/"],
                "exclude": [],
                "depth": 0,
                "twins": False,
                "selectors": MAIN,
            },
        },
        "pdf_twins": [],
    }


@pytest.fixture
def tu(profile_dir: Path) -> Path:
    (profile_dir / "sources.yaml").write_text(yaml.safe_dump(sources_yaml()), encoding="utf-8")
    return profile_dir


def site(served, overrides=None, down=()):
    """The mini-site's hosts plus a full-set host behind one in-process client. `overrides` maps a URL to a
    (media type, body), a function of the request or None; a host in `down` answers 503."""
    overrides = overrides or {}

    def handler(request: httpx.Request):
        url = str(request.url)
        if request.url.host in down:
            return httpx.Response(503)
        if url in overrides:
            made = overrides[url]
            if callable(made):
                return made(request)
            return httpx.Response(200, headers={"content-type": made[0]}, content=made[1]) if made else None
        if request.url.host == LADADI:
            return (
                httpx.Response(200, headers={"content-type": "text/html"}, text=LADADI_HTML)
                if url == LADADI_PAGE
                else None
            )
        app = {FB20: FACULTY, INTERN: UNIVERSITY}.get(request.url.host)
        if app is None:
            return None
        reply = app.get(request.url.raw_path.decode())
        return httpx.Response(
            reply.status_code, headers={"content-type": reply.headers["content-type"]}, content=reply.content
        )

    return served(handler)


def connector(net, today=TODAY) -> TudaWeb:
    return TudaWeb(client=net.client, interval=0, today=lambda: today)


def collect(net, make_context, **context):
    return connector(net).collect(make_context(**context))


def documents(result) -> dict[str, PageItem]:
    return {i.url: i for i in result.items if isinstance(i, PageItem)}


def pdfs(result) -> dict[str, PdfItem]:
    return {i.url: i for i in result.items if isinstance(i, PdfItem)}


def test_core_set_by_default(tu, make_context, served):
    net = site(served)
    result = collect(net, make_context)
    assert result.report["sources"] == "core" and not [u for u in net.urls() if LADADI in u]
    assert LADADI_PAGE + "#landkreis" not in documents(result)
    full = site(served)
    both = collect(full, make_context, options={"sources": "full", "resume": False})
    assert both.report["sources"] == "full" and documents(both)[LADADI_PAGE + "#landkreis"].metadata["title"] == (
        "Ausländerwesen – Landkreis"
    )
    assert set(documents(result)) <= set(documents(both))  # the full set is the core set and more


def test_the_full_set_swaps_in_a_core_hosts_full_block_and_adds_the_full_hosts():
    sources = load_profile(Sources, TU / "sources.yaml")
    core, full = active_hosts(sources, False), active_hosts(sources, True)
    assert "www.ladadi.de" not in core and "www.ladadi.de" in full
    assert set(core) < set(full)
    host = "www.tu-darmstadt.de"
    assert core[host].source.prefixes == () and core[host].source.depth == 0 and core[host].source.sitemaps == ()
    assert full[host].source.prefixes == ("/studieren/studierende_tu/", "/studieren/studieren_von_a_bis_z/")
    assert full[host].source.sitemaps == ("https://www.tu-darmstadt.de/sitemap.de.xml",)
    assert full[host].source.seeds == core[host].source.seeds  # the listed pages stay
    assert core["www.informatik.tu-darmstadt.de"] == full["www.informatik.tu-darmstadt.de"]


def test_a_core_build_makes_documents_of_hub_sections_and_pdfs_with_their_metadata(tu, make_context, served):
    net = site(served)
    result = collect(net, make_context, settings={"min_documents": 30})
    assert result.expected_minimum == 30
    docs = documents(result)
    po, old = docs[f"{HUB}#pruefungsordnung_2023"].metadata, docs[f"{HUB}#pruefungsordnung_2015"].metadata
    assert (po["regulation"], po["degree"], po["programme"], po["po_version"], po["lang"]) == (
        "po",
        ["bachelor"],
        ["bsc-informatik"],
        [2023],
        "de",
    )
    assert po["twin_key"].endswith("ordnungen_bsc_informatik#pruefungsordnung_2023") and po["archived"] is False
    assert old["po_version"] == [2015] and po["doc_type"] == "Prüfungsordnung"
    english = docs[f"https://{FB20}{pages.STUDIENBUERO}/index.en.jsp#kontakt"].metadata
    assert (
        english["lang"] == "en"
        and english["twin_key"] == docs[f"https://{FB20}{pages.STUDIENBUERO}/index.de.jsp#kontakt"].metadata["twin_key"]
    )
    assert (
        "regulation" in english and "programme" not in english and "degree" not in english
    )  # applies to all: absent, not []
    assert HUB_EN not in docs and not [u for u in docs if "pruefungen/index.en" in u]  # fallback twins are dropped
    # the PDFs: the programme's PO, and the university's APB from the other host the hub links (its original, not the copy)
    assert set(pdfs(result)) == {PO_PDF, APB_ORIGINAL}
    po_pdf = pdfs(result)[PO_PDF].metadata
    assert po_pdf["regulation"] == "po" and po_pdf["programme"] == ["bsc-informatik"] and po_pdf["po_version"] == [2023]
    assert po_pdf["valid_from"] == "2023-10-01" and po_pdf["url"] == HUB and po_pdf["origin"] == "website"
    assert po_pdf["download_title"] == "Studiengangsordnung Bachelor Informatik 2023"
    apb = pdfs(result)[APB_ORIGINAL].metadata
    assert apb["regulation"] == "apb" and "programme" not in apb and "po_version" not in apb
    assert result.report["copies"] == [APB_COPY]
    skipped = {s["url"]: s["reason"] for s in result.report["skipped"]}
    assert skipped[f"https://{FB20}{pages.NOAI_PAGE}"] == "meta robots"
    assert skipped[f"https://{FB20}{pages.LOGIN_PAGE}"] == "robots"
    for item in pdfs(result).values():  # the files are in the PDF folder under the registry's names
        assert Path(item.path) == local_path(item.url, make_context().paths["pdf_dir"]) and Path(item.path).is_file()


def extraction_cache(items, extracted_dir: Path) -> None:
    """What the PDF extraction leaves: the text of each page under the file's hash (docling is not run in a test)."""
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        pdf = pdfium.PdfDocument(item.path)
        text = "".join(f"[Page {n}]\n{page.get_textpage().get_text_range()}\n\n" for n, page in enumerate(pdf, 1))
        record = {"text": text, "file_hash": item.file_hash, "extraction_method": "test"}
        (extracted_dir / f"{item.file_hash}.json").write_text(json.dumps(record), encoding="utf-8")


def test_a_po_pdf_reaches_the_index_rows_with_its_scope(tu, make_context, served, tmp_path):
    result = collect(site(served), make_context)
    po = pdfs(result)[PO_PDF]
    extraction_cache([po], tmp_path / "x")
    docs, missing = pdf_documents([po], set(), extracted_dir=tmp_path / "x")
    assert missing == 0 and docs
    profile = load_profile(Profile, TU / "profile.yaml")
    parents = [
        {**p.metadata, "source": p.source}
        for p in chunk_documents([{**d, "source": "tuda_web"} for d in docs], mode="structure", header="tier1")
    ]
    assert parents and all(p["programme"] == ["bsc-informatik"] and p["po_version"] == [2023] for p in parents)
    sections, chunks = writer.rows(
        chunk_documents([{**d, "source": "tuda_web"} for d in docs], mode="structure"), profile
    )
    from indexer.index import ddl

    for row in (
        *(dict(zip(ddl.SECTION_COLUMNS, s)) for s in sections),
        *(dict(zip(ddl.CHUNK_COLUMNS, c.values("v"))) for c in chunks),
    ):
        assert (row["regulation"], row["programme"], row["po_version"], row["degree"]) == (
            "po",
            ["bsc-informatik"],
            [2023],
            ["bachelor"],
        )
        assert row["valid_from"] == date(2023, 10, 1) and row["lang"] == "de" and row["audience"] == "public"


def truncated(body: bytes):
    """A PDF whose connection breaks off: the server promises all of it and sends a part."""

    def reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "application/pdf", "content-length": str(len(body))}, content=body[:200]
        )

    return reply


def registry(ctx) -> dict:
    return ctx.state.load(REGISTRY) or {}


def test_interrupted_pdf_is_not_registered(tu, make_context, served):
    """A download that breaks off is neither registered nor kept, however far it got; the page that links it is still built."""
    ctx = make_context()
    net = site(served, {PO_PDF: truncated(pages.PO_2023)})
    result = connector(net).collect(ctx)
    assert set(pdfs(result)) == {APB_ORIGINAL} and f"{HUB}#pruefungsordnung_2023" in documents(result)
    assert PO_PDF not in registry(ctx) and APB_ORIGINAL in registry(ctx)
    assert not local_path(PO_PDF, ctx.paths["pdf_dir"]).exists()
    assert not list(Path(ctx.paths["crawl_dir"]).rglob("*.part")) and not list(
        Path(ctx.paths["crawl_dir"]).rglob("*.pdf.tmp")
    )
    assert {"url": PO_PDF, "reason": "unreachable"} in result.report["skipped"]


def test_an_interrupted_download_keeps_the_pdf_the_registry_already_holds(tu, make_context, served):
    ctx = make_context()
    first = connector(site(served)).collect(ctx)
    kept = Path(pdfs(first)[PO_PDF].path)
    before = (kept.read_bytes(), registry(ctx)[PO_PDF])
    later = connector(site(served, {PO_PDF: truncated(pages.PO_2023)}), TODAY + timedelta(days=1)).collect(ctx)
    assert (kept.read_bytes(), registry(ctx)[PO_PDF]) == before
    assert PO_PDF in pdfs(later) and later.report["pdf_sync"]["retained"] == [
        PO_PDF
    ]  # kept for a few builds, not dropped


def test_a_pdf_kept_after_a_failed_download_is_dated_by_its_file_not_by_the_crawl(tu, make_context, served):
    ctx = make_context()
    first = connector(site(served)).collect(ctx)
    day = date(2026, 1, 5)
    os.utime(pdfs(first)[PO_PDF].path, (t := datetime(day.year, day.month, day.day, 12).timestamp(), t))
    later = connector(site(served, {PO_PDF: truncated(pages.PO_2023)}), TODAY + timedelta(days=1)).collect(ctx)
    listed = {m["url"]: m for m in later.report["manifest"]}
    assert PO_PDF in later.report["pdf_sync"]["retained"]
    assert (
        listed[PO_PDF]["fetched"] == day.isoformat()
        and listed[APB_ORIGINAL]["fetched"] == (TODAY + timedelta(days=1)).isoformat()
    )


def without_pdf_links(page: str) -> str:
    return re.sub(r'<a [^>]*\.pdf"[^>]*>.*?</a>', "", page, flags=re.S)


def test_pdfs_that_vanish_from_intact_pages_are_kept_for_a_few_builds_and_named(tu, make_context, served):
    ctx = make_context()
    connector(site(served)).collect(ctx)
    pages_without = {
        url: (
            "text/html",
            without_pdf_links(
                FACULTY.get(urlsplit(url).path).text if FB20 in url else UNIVERSITY.get(urlsplit(url).path).text
            ),
        )
        for url in (HUB, INTERN_INDEX)
    }
    later = connector(site(served, pages_without), TODAY + timedelta(days=1)).collect(ctx)
    assert f"{HUB}#pruefungsordnung_2023" in documents(later)  # the pages are intact
    assert set(pdfs(later)) == {PO_PDF, APB_ORIGINAL}  # and their PDFs are kept, as for a partial list
    assert sorted(later.report["pdf_sync"]["retained"]) == sorted([PO_PDF, APB_ORIGINAL])
    lines = cli.summary_lines("tu_web", later.report)
    assert f"warning: tu_web: PDF kept though this crawl did not list it: {PO_PDF}" in lines
    assert registry(ctx).keys() >= {PO_PDF, APB_ORIGINAL}


def test_a_pdf_kept_from_an_earlier_build_is_in_the_manifest_with_its_stored_metadata(tu, make_context, served):
    ctx = make_context()
    first = connector(site(served)).collect(ctx)
    pages_without = {url: ("text/html", without_pdf_links(FACULTY.get(urlsplit(url).path).text)) for url in (HUB,)}
    later = connector(site(served, pages_without), TODAY + timedelta(days=1)).collect(ctx)
    assert PO_PDF in later.report["pdf_sync"]["retained"]
    listed = {m["url"]: m for m in later.report["manifest"]}
    before = {m["url"]: m for m in first.report["manifest"]}
    assert listed[PO_PDF]["content_hash"] == before[PO_PDF]["content_hash"] == pdfs(later)[PO_PDF].file_hash
    assert listed[PO_PDF]["metadata"] == before[PO_PDF]["metadata"] and listed[PO_PDF]["fetched"]
    assert f"{HUB}#pruefungsordnung_2023" in listed  # the pages stay


def test_a_kept_pdf_without_its_list_record_is_not_indexed_for_lacking_its_metadata(tu, make_context, served):
    ctx = make_context()
    connector(site(served)).collect(ctx)
    ctx.state.path(LIST_STATE).unlink()  # the refusal for a corrupt list state tells the user to delete it
    pages_without = {url: ("text/html", without_pdf_links(FACULTY.get(urlsplit(url).path).text)) for url in (HUB,)}
    later = connector(site(served, pages_without), TODAY + timedelta(days=1)).collect(ctx)
    assert PO_PDF not in pdfs(later) and PO_PDF in registry(ctx)  # the file stays, it is not indexed
    assert {"url": PO_PDF, "missing": ["regulation"]} in later.report["not_indexed"]


def test_a_pdf_whose_redirect_ends_outside_its_scope_is_not_indexed_and_is_named(tu, make_context, served):
    elsewhere = f"https://{FB20}/media/informatik/other/x.pdf"
    moved = lambda request: httpx.Response(302, headers={"location": elsewhere})  # noqa: E731
    net = site(served, {PO_PDF: moved, elsewhere: ("application/pdf", pages.PO_2023)})
    ctx = make_context()
    result = connector(net).collect(ctx)
    assert PO_PDF not in pdfs(result) and PO_PDF not in registry(ctx) and elsewhere not in pdfs(result)
    assert {"url": PO_PDF, "reason": "redirected out of scope"} in result.report["skipped"]
    assert f"note: tu_web: not fetched (redirected out of scope): {PO_PDF}" in cli.summary_lines(
        "tu_web", result.report
    )


def test_a_pdf_this_crawl_excluded_on_purpose_is_not_kept_while_a_vanished_one_is(tu, make_context, served):
    ctx = make_context()
    connector(site(served)).collect(ctx)
    elsewhere = f"https://{FB20}/media/informatik/other/x.pdf"
    moved = lambda request: httpx.Response(302, headers={"location": elsewhere})  # noqa: E731
    net = site(served, {PO_PDF: moved, elsewhere: ("application/pdf", pages.PO_2023)})
    later = connector(net, TODAY + timedelta(days=1)).collect(ctx)
    assert PO_PDF not in pdfs(later) and PO_PDF not in later.report["pdf_sync"]["retained"]  # the APB is unaffected
    assert APB_ORIGINAL in pdfs(later)
    vanished = {
        url: ("text/html", without_pdf_links((FACULTY if FB20 in url else UNIVERSITY).get(urlsplit(url).path).text))
        for url in (HUB, INTERN_INDEX)
    }
    gone = connector(site(served, vanished), TODAY + timedelta(days=2)).collect(ctx)
    assert sorted(gone.report["pdf_sync"]["retained"]) == sorted([PO_PDF, APB_ORIGINAL])


MSC_HUB = f"https://{FB20}{pages.ORDNUNGEN}/ordnungen_msc_informatik.de.jsp"
MSC_PDF = (
    f"https://{FB20}/media/informatik/fb20_studium/formulare_und_dokumente/ordnungen/ordnungen_msc/"
    "pruefungsordnung_9/Ordnung_des_Studiengangs_M_Sc_Informatik_PO_2023.pdf"
)


def test_one_file_linked_from_two_programme_hubs_gets_both_programmes(tu, make_context, served):
    msc_hub = (
        '<html><head><title>Ordnungen M.Sc.</title></head><body><div class="main-column">'
        '<section class="content-section" id="pruefungsordnung_2023"><h3>Prüfungsordnung 2023</h3>'
        f'<a href="{MSC_PDF}">Ordnung des Studiengangs M.Sc. Informatik PO 2023</a></section></div></body></html>'
    )
    (tu / "sources.yaml").write_text(
        yaml.safe_dump(
            sources_yaml(
                extra_seeds=(MSC_HUB,),
                extra_prefixes=(
                    f"{pages.ORDNUNGEN}/ordnungen_msc_informatik",
                    "/media/informatik/fb20_studium/formulare_und_dokumente/ordnungen/ordnungen_msc/",
                ),
            )
        ),
        encoding="utf-8",
    )
    net = site(served, {MSC_HUB: ("text/html", msc_hub), MSC_PDF: ("application/pdf", pages.PO_2023)})
    result = collect(net, make_context)
    both = pdfs(result)[PO_PDF].metadata  # the first link; the file is the same
    assert pdfs(result)[MSC_PDF].file_hash == pdfs(result)[PO_PDF].file_hash
    assert (both["programme"], both["degree"], both["po_version"]) == (
        ["bsc-informatik", "msc-informatik"],
        ["bachelor", "master"],
        [2023],
    )
    assert both["regulation"] == "po"
    listed = {m["url"]: m for m in result.report["manifest"]}
    assert (listed[PO_PDF]["metadata"]["programme"], listed[PO_PDF]["metadata"]["degree"]) == (
        ["bsc-informatik", "msc-informatik"],
        ["bachelor", "master"],
    )  # the manifest shows the merged scope of the indexed document, not the first link's
    assert listed[MSC_PDF]["metadata"]["programme"] == ["msc-informatik"]
    assert listed[PO_PDF]["content_hash"] == pdfs(result)[PO_PDF].file_hash


OLD_AB = (
    f"https://{FB20}/media/informatik/fb20_studium/formulare_und_dokumente/ordnungen/ordnungen_bsc/"
    "pruefungsordnung_2007/Ausfuehrungsbestimmungen_AB_2007_zu_den_APB.pdf"
)


def test_implementing_provisions_of_an_old_po_year_are_listed_and_not_indexed(tu, make_context, served):
    hub = (
        '<html><head><title>Ordnungen B.Sc.</title></head><body><div class="main-column">'
        '<section class="content-section" id="pruefungsordnung_2023"><h3>Prüfungsordnung 2023</h3>'
        f'<a href="{PO_PDF}">Studiengangsordnung Bachelor Informatik 2023</a></section>'
        '<section class="content-section" id="pruefungsordnung_2007_dateiliste">'
        f'<a href="{OLD_AB}">Ausführungsbestimmungen (AB) 2007 zu den APB der TU Darmstadt</a></section>'
        "</div></body></html>"
    )
    result = collect(site(served, {HUB: ("text/html", hub), OLD_AB: ("application/pdf", pages.PO_2023)}), make_context)
    assert {"url": OLD_AB, "missing": ["po_version"]} in result.report["not_indexed"]
    assert OLD_AB not in pdfs(result) and PO_PDF in pdfs(result)


def test_every_url_is_asked_for_once_per_build(tu, make_context, served):
    net = site(served)
    collect(net, make_context, options={"sources": "full", "resume": False})
    assert {u: n for u, n in Counter(net.urls()).items() if n > 1} == {}


def test_a_pdf_outside_the_listing_hosts_prefixes_is_not_fetched(tu, make_context, served):
    index = (
        '<html><body><div class="main-column"><section class="content-section" id="apb"><h3>APB</h3>'
        f'<a href="{pages.APB_ORIGINAL_PDF}">APB</a> <a href="{BUNDLE}">Satzungsbeilage 2023 II</a>'
        "</section></div></body></html>"
    )
    net = site(served, {INTERN_INDEX: ("text/html", index)})
    result = collect(net, make_context)
    assert BUNDLE not in net.urls() and BUNDLE not in pdfs(result) and APB_ORIGINAL in pdfs(result)


def test_a_page_that_ends_outside_its_scope_after_a_redirect_is_not_indexed(tu, make_context, served):
    moved = lambda request: httpx.Response(302, headers={"location": f"https://{LADADI}/anderswo/"})  # noqa: E731
    net = site(served, {LADADI_PAGE: moved, f"https://{LADADI}/anderswo/": ("text/html", LADADI_HTML)})
    result = collect(net, make_context, options={"sources": "full", "resume": False})
    assert not [u for u in documents(result) if LADADI in u]
    assert {"url": LADADI_PAGE, "reason": "redirected out of scope"} in result.report["skipped"]


def test_the_user_agent_carries_the_contact_only_when_there_is_one(tu, make_context, served):
    net = site(served)
    collect(net, make_context)
    assert {r.headers["user-agent"] for r in net.log} == {user_agent("dummy@example.invalid")}
    bare = site(served)
    collect(
        bare,
        make_context,
        options={"contact": None},
        paths={**make_context().paths, "crawl_dir": make_context().paths["crawl_dir"] + "-bare"},
    )
    agents = {r.headers["user-agent"] for r in bare.log}
    assert agents == {user_agent(None)} and "student research prototype" in agents.pop() and "@" not in user_agent(None)


def test_resume_continues_the_interrupted_crawl_and_a_fresh_crawl_begins_again(tu, make_context, served):
    ctx = make_context()

    def stop(request):
        raise RuntimeError("the machine went to sleep")

    with pytest.raises(RuntimeError):
        connector(site(served, {PO_PDF: stop})).collect(ctx)
    assert ctx.state.load("crawl_run.json") == {"started": TODAY.isoformat(), "sources": "core", "complete": False}

    next_day = TODAY + timedelta(days=1)
    resumed = site(served)
    done = connector(resumed, next_day).collect(make_context(options={"resume": True}))
    assert PO_PDF in pdfs(done) and ctx.state.load("crawl_run.json")["complete"] is True
    asked = set(resumed.urls())
    assert PO_PDF in asked and HUB not in asked  # the page the first crawl fetched is not fetched again

    fresh = site(served)
    connector(fresh, next_day + timedelta(days=1)).collect(
        make_context(options={"resume": True})
    )  # the last crawl finished
    assert HUB in fresh.urls() and PO_PDF in fresh.urls()


def test_a_list_state_of_an_older_format_refuses_the_build(tu, make_context, served):
    ctx = make_context()
    connector(site(served)).collect(ctx)
    state = json.loads(ctx.state.path(LIST_STATE).read_text(encoding="utf-8"))
    state["known"][PO_PDF]["retired_field"] = 1
    ctx.state.path(LIST_STATE).write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(CorruptStateError, match="list state"):
        connector(site(served), TODAY + timedelta(days=1)).collect(ctx)


def test_frozen_inputs_build_from_the_cache_without_a_request(tu, make_context, served):
    ctx = make_context()
    first = connector(site(served)).collect(ctx)
    frozen = site(served)
    again = connector(frozen).collect(make_context(reuse=True))
    assert frozen.log == [] and set(documents(again)) == set(documents(first)) and set(pdfs(again)) == set(pdfs(first))


def test_the_connector_is_found_by_its_entry_point():
    found = {ep.name: ep for ep in entry_points(group="rag_core.connectors")}
    assert found["tuda_web"].load() is TudaWeb and TudaWeb.name == "tuda_web"


class Lock:
    def __init__(self, target, holder, connect_fn):
        pass

    def acquire(self):
        return self

    def release(self):
        pass


class Conn:
    def close(self):
        pass


PRUEFUNGEN = f"https://{FB20}{pages.STUDIENBUERO}/pruefungen/index.de.jsp"
EXTRA_HUB = f"https://{FB20}{pages.BSC_HUB}/weitere.de.jsp"
EMPTY_PAGE = f"https://{FB20}{pages.STUDIENBUERO}/leer.de.jsp"
UNDATED_MHB = f"https://{FB20}/media/informatik/fb20_studium/formulare_und_dokumente/ordnungen/ordnungen_bsc/Modulhandbuch_BSc.pdf"
EXTRA_PAGES = {
    EXTRA_HUB: (
        "text/html",
        '<html><head><title>Weitere Unterlagen</title></head><body><div class="main-column">'
        '<section class="content-section" id="sonstiges"><h3>Sonstiges</h3>'
        f'<p>Das Modulhandbuch ohne Angabe einer Ordnung.</p><a href="{UNDATED_MHB}">Modulhandbuch</a>'
        "</section></div></body></html>",
    ),
    UNDATED_MHB: ("application/pdf", pages.PO_2023),
    EMPTY_PAGE: (
        "text/html",
        "<html><head><title>Leer</title></head><body><p>Umgebaut: kein Hauptbereich mehr.</p></body></html>",
    ),
}


@pytest.fixture
def builder(tmp_path, monkeypatch, served):
    """builder(site, day) -> (exit code, printed lines) of `indexer build` over the mini-site on that day, on a copy of the TU
    profile whose sources are the mini-site's. write_and_publish, the lock and the PDF extraction are replaced (they have
    their own tests); the crawl, the sync, the selection, chunking and the gates run as they are."""
    folder = tmp_path / "tu"
    shutil.copytree(TU, folder, ignore=shutil.ignore_patterns("smoke"))
    (folder / "sources.yaml").write_text(
        yaml.safe_dump(sources_yaml(extra_seeds=(EXTRA_HUB, EMPTY_PAGE))), encoding="utf-8"
    )
    spec = yaml.safe_load((folder / "profile.yaml").read_text(encoding="utf-8"))
    spec["ingest"]["sources"]["tu_web"]["min_documents"] = 1
    (folder / "profile.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    environ = {
        "RAG_PROFILE": str(folder / "profile.yaml"),
        "RAG__DATABASE__NAME": "vector_db",
        "RAG__DATABASE__USER": "rag_indexer",
        "RAG__DATABASE__PASSWORD": "pw",
        **{
            f"RAG__PATHS__{k.upper()}": str(tmp_path / v)
            for k, v in {
                "crawl_dir": "crawl",
                "state_dir": "state",
                "extracted_dir": "extracted",
                "pdf_dir": "pdfs",
                "skipped_pdfs_log": "skipped.json",
            }.items()
        },
    }
    settings = load_settings(environ, secrets_dir=str(tmp_path / "secrets"))
    published: list[int] = []

    def extract_pdfs(*, inventory, extracted_dir, **_):
        extraction_cache([type("Item", (), e)() for e in inventory], Path(extracted_dir))
        return {"processed": len(inventory), "cached": 0, "failed": 0, "skipped_page_limit": 0}

    async def write_and_publish(settings, target, connect, embedding, parents, **_):
        published.append(len(parents))
        return {
            "generation": len(published),
            "index_counts": {"children": 1, "parents": len(parents)},
            "embeddings": {"cached": 0, "computed": 1},
            "changes": {"added": 1, "changed": 0, "removed": 0},
            "swapped": True,
            "previous_generation": len(published) - 1,
        }

    monkeypatch.setattr(gen, "BuildLock", Lock)
    monkeypatch.setattr(build, "extract_pdfs", extract_pdfs)
    monkeypatch.setattr(build, "write_and_publish", write_and_publish)
    wiring = cli.wire(settings, lambda application_name: Conn())

    class Embedding:
        def info(self):
            return None

    def run(overrides=None, day=TODAY, down=()):
        lines: list[str] = []
        net = site(served, overrides, down)
        code = cli.build_index(
            wiring, Embedding(), {"tu_web": connector(net, day)}, lines.append, {"sources": "core", "resume": False}
        )
        return code, lines

    run.published, run.settings = published, settings
    return run


def test_build_summary_lists_fail_closed_and_empty_pages(builder):
    code, lines = builder(EXTRA_PAGES)
    assert code == 0
    assert f"warning: tu_web: empty page {EMPTY_PAGE} (main content selector '.main-column' matched nothing)" in lines
    assert f"warning: tu_web: not indexed, lacks po_version: {UNDATED_MHB}" in lines  # a Modulhandbuch of no PO version
    assert f"note: tu_web: copy skipped, its original is indexed: {APB_COPY}" in lines
    assert any(line.startswith("generation 1 published") for line in lines)
    assert builder.published and not any("Modulhandbuch_BSc" in line for line in lines if line.startswith("source"))


def test_a_published_build_writes_its_snapshot_manifest(builder, tmp_path):
    assert builder(EXTRA_PAGES)[0] == 0
    manifest = json.loads((tmp_path / "snapshots" / "1.json").read_text(encoding="utf-8"))
    documents_ = {d["url"]: d for d in manifest["documents"]}
    assert manifest["generation"] == "1" and list(documents_) == sorted(documents_)
    assert documents_[PO_PDF]["fetched"] == TODAY.isoformat() and len(documents_[PO_PDF]["content_hash"]) == 64
    assert documents_[PO_PDF]["metadata"]["po_version"] == [2023]
    assert f"{HUB}#pruefungsordnung_2023" in documents_ and APB_COPY not in documents_ and UNDATED_MHB not in documents_


def test_a_retained_pdf_in_the_live_manifest_does_not_stop_the_next_build(builder, tmp_path):
    assert builder(EXTRA_PAGES)[0] == 0
    no_links = {url: ("text/html", without_pdf_links(FACULTY.get(urlsplit(url).path).text)) for url in (HUB,)}
    code, lines = builder({**EXTRA_PAGES, **no_links}, TODAY + timedelta(days=1))
    assert code == 0 and not any(line.startswith("refused:") for line in lines)
    kept = json.loads((tmp_path / "snapshots" / "2.json").read_text(encoding="utf-8"))["documents"]
    assert PO_PDF in {d["url"] for d in kept}  # generation 2 lists the PDF it kept
    code, lines = builder({**EXTRA_PAGES, **no_links}, TODAY + timedelta(days=2))
    assert code == 0 and not any(
        line.startswith("refused:") for line in lines
    )  # and it is no page: the gate lets it be


def test_shrunk_build_keeps_live_generation(builder, tmp_path):
    assert builder(EXTRA_PAGES)[0] == 0
    live = list(builder.published)
    stats = (tmp_path / "state" / gates.STATS_NAME).read_text(encoding="utf-8")
    registry_before = (tmp_path / "state" / REGISTRY).read_text(encoding="utf-8")
    # the next day the faculty's site was rebuilt: its pages have no main content any more
    umbau = ("text/html", "<html><head><title>Umbau</title></head><body><p>Neue Seite.</p></body></html>")
    redesigned = {**EXTRA_PAGES, **{url: umbau for url in (HUB, STUDIENBUERO, PRUEFUNGEN, EXTRA_HUB)}}
    code, lines = builder(redesigned, TODAY + timedelta(days=1))
    refused = [line for line in lines if line.startswith("refused:")]
    assert code == 1 and len(refused) == 1 and "corpus shrank" in refused[0] and "swap refused" in refused[0]
    assert builder.published == live  # nothing was written: the live generation stays
    assert sorted(p.name for p in (tmp_path / "snapshots").iterdir()) == ["1.json"]
    assert (tmp_path / "state" / gates.STATS_NAME).read_text(encoding="utf-8") == stats  # and the size reference
    # the summary names the pages behind the shrink, before the refusal
    named = f"warning: tu_web: empty page {HUB} (main content selector '.main-column' matched nothing)"
    assert named in lines and lines.index(named) < lines.index(refused[0])
    # the PO PDF the hub no longer links is kept for a few builds, not dropped from the registry
    assert (
        (tmp_path / "state" / REGISTRY).read_text(encoding="utf-8").count(PO_PDF) == registry_before.count(PO_PDF) == 1
    )


UMBAU = ("text/html", "<html><head><title>Umbau</title></head><body><p>Neue Seite.</p></body></html>")


def test_a_hub_whose_selector_finds_nothing_keeps_the_live_generation(builder, tmp_path):
    assert builder(EXTRA_PAGES)[0] == 0
    live = list(builder.published)
    stats = (tmp_path / "state" / gates.STATS_NAME).read_text(encoding="utf-8")
    code, lines = builder({**EXTRA_PAGES, EXTRA_HUB: UMBAU}, TODAY + timedelta(days=1))
    refused = [line for line in lines if line.startswith("refused:")]
    assert code == 1 and len(refused) == 1 and refused[0].startswith(f"refused: {EXTRA_HUB}: main content not found")
    assert "generation 1 (live)" in refused[0] and "swap refused" in refused[0] and "corpus shrank" not in refused[0]
    assert builder.published == live and (tmp_path / "state" / gates.STATS_NAME).read_text(encoding="utf-8") == stats
    named = f"warning: tu_web: empty page {EXTRA_HUB} (main content selector '.main-column' matched nothing)"
    assert named in lines and lines.index(named) < lines.index(refused[0])


def test_an_empty_page_that_is_not_in_the_live_manifest_is_reported_not_refused(builder, tmp_path):
    assert builder(EXTRA_PAGES)[0] == 0
    empty = f"warning: tu_web: empty page {EMPTY_PAGE} (main content selector '.main-column' matched nothing)"
    # EMPTY_PAGE is empty again and was never in a manifest
    code, lines = builder(EXTRA_PAGES, TODAY + timedelta(days=1))
    assert code == 0 and empty in lines
    (tmp_path / "snapshots" / "2.json").unlink()  # the live generation has no manifest
    code, lines = builder({**EXTRA_PAGES, EXTRA_HUB: UMBAU}, TODAY + timedelta(days=2))
    assert code == 0 and not any(line.startswith("refused:") for line in lines)
    assert f"warning: tu_web: empty page {EXTRA_HUB} (main content selector '.main-column' matched nothing)" in lines


def test_stats_without_a_generation_do_not_gate_the_pages(builder, tmp_path):
    assert builder(EXTRA_PAGES)[0] == 0
    stats_file = tmp_path / "state" / gates.STATS_NAME
    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    stats_file.write_text(json.dumps({k: v for k, v in stats.items() if k != "generation"}), encoding="utf-8")
    code, lines = builder({**EXTRA_PAGES, EXTRA_HUB: UMBAU}, TODAY + timedelta(days=1))
    assert code == 0 and not any(line.startswith("refused:") for line in lines)


def test_a_broken_live_manifest_fails_the_build_and_keeps_the_live_generation(builder, tmp_path):
    assert builder(EXTRA_PAGES)[0] == 0
    live = list(builder.published)
    (tmp_path / "snapshots" / "1.json").write_text("{broken", encoding="utf-8")
    code, lines = builder(EXTRA_PAGES, TODAY + timedelta(days=1))
    assert code == 1 and any(line.startswith("failed: CorruptStateError") and "1.json" in line for line in lines)
    assert builder.published == live


def test_a_shrunk_build_right_after_a_publish_or_rollback_is_refused(builder, tmp_path):
    assert builder(EXTRA_PAGES)[0] == 0
    live = list(builder.published)
    stats_file = tmp_path / "state" / gates.STATS_NAME
    written = json.loads(stats_file.read_text(encoding="utf-8"))
    # what `indexer publish` and a rollback record: counts and profile, no source set
    stats_file.write_text(json.dumps({k: written[k] for k in ("parents", "children", "profile")}), encoding="utf-8")
    umbau = ("text/html", "<html><head><title>Umbau</title></head><body><p>Neue Seite.</p></body></html>")
    redesigned = {**EXTRA_PAGES, **{url: umbau for url in (HUB, STUDIENBUERO, PRUEFUNGEN, EXTRA_HUB)}}
    code, lines = builder(redesigned, TODAY + timedelta(days=1))
    refused = [line for line in lines if line.startswith("refused:")]
    assert code == 1 and len(refused) == 1 and "corpus shrank" in refused[0]
    assert builder.published == live and not any(line.startswith("note: no size gate") for line in lines)


def test_a_host_that_is_down_keeps_the_live_generation(builder, tmp_path):
    assert builder(EXTRA_PAGES)[0] == 0
    live = list(builder.published)
    stats = (tmp_path / "state" / gates.STATS_NAME).read_text(encoding="utf-8")
    code, lines = builder(EXTRA_PAGES, TODAY + timedelta(days=1), down={INTERN})
    warning = f"warning: tu_web: host down {INTERN}: robots.txt unreadable (http 503)"
    refused = [line for line in lines if line.startswith("refused:")]
    assert code == 1 and len(refused) == 1 and INTERN in refused[0] and "swap refused" in refused[0]
    assert warning in lines and lines.index(warning) < lines.index(refused[0])
    assert builder.published == live
    assert sorted(p.name for p in (tmp_path / "snapshots").iterdir()) == ["1.json"]
    assert (tmp_path / "state" / gates.STATS_NAME).read_text(encoding="utf-8") == stats


def test_a_full_set_host_that_is_down_does_not_stop_a_core_build(builder):
    code, lines = builder(EXTRA_PAGES, down={LADADI})
    assert code == 0 and not any("host down" in line for line in lines)
