"""The TU profile's crawl settings run through discovery itself: the PDFs its hub and index pages link are found at the
configured depth, and a PDF outside the prefixes is not. The hosts are served in-process; no request leaves the process."""

from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from rag_core.settings import load_profile
from rag_core.settings.profile import Sources
from rag_testing import serve
from rag_testing.minisite import pages

from indexer.crawl.discover import Source, discover
from indexer.crawl.fetch import Fetcher

TU = Path(__file__).resolve().parents[3] / "profiles" / "tu-informatik"
FB20 = "www.informatik.tu-darmstadt.de"
INTERN = "www.intern.tu-darmstadt.de"
APB = "/media/dezernat_ii/ordnungen/apb.pdf"
OTHER_PDF = "/media/dezernat_ii/satzungsbeilagen/SB_2023_II.pdf"  # the bundle of acts: linked, not listed
INDEX = "/verwaltung/dez_ii/hochschulrecht/gesetze_und_ordnungen/index.de.jsp"
INTERN_PAGES = {
    "/robots.txt": ("text/plain", "User-agent: *\nDisallow: /*.login\n"),
    INDEX: (
        "text/html",
        f'<html><body><main><a href="{APB}">APB</a> <a href="{OTHER_PDF}">Satzungsbeilage</a>'
        '<a href="/verwaltung/dez_vii/info.de.jsp">other page</a></main></body></html>',
    ),
}


def serve_hosts(request: httpx.Request) -> httpx.Response:
    if request.url.host == INTERN:
        content_type, body = INTERN_PAGES.get(request.url.path, ("text/plain", ""))
        return httpx.Response(
            200 if request.url.path in INTERN_PAGES else 404, headers={"content-type": content_type}, content=body
        )
    reply = FACULTY.get(request.url.raw_path.decode())
    return httpx.Response(
        reply.status_code, headers={"content-type": reply.headers["content-type"]}, content=reply.content
    )


FACULTY = TestClient(serve.minisite_app("fb20", f"https://{INTERN}"), base_url=f"https://{FB20}")


def found(host: str, depth: int | None = None) -> list[str]:
    """What discovery lists for a host of the profile's sources.yaml (its seeds, sitemaps, prefixes and depth)."""
    entry = load_profile(Sources, TU / "sources.yaml").hosts[host]
    source = Source(
        host,
        prefixes=tuple(entry.prefixes),
        exclude=tuple(entry.exclude),
        sitemaps=tuple(entry.sitemap),
        seeds=tuple(entry.seeds),
        depth=entry.depth if depth is None else depth,
    )
    client = httpx.Client(transport=httpx.MockTransport(serve_hosts))
    return discover(source, Fetcher((host,), contact="dummy@example.invalid", client=client, interval=0))


def test_the_intern_index_leads_to_the_apb_pdf_and_only_to_pdfs_under_the_prefixes():
    urls = found(INTERN)
    assert f"https://{INTERN}{APB}" in urls
    assert f"https://{INTERN}{OTHER_PDF}" not in urls and f"https://{INTERN}/verwaltung/dez_vii/info.de.jsp" not in urls
    # the same source without link-following lists the seed only, so the depth is what reaches the PDF
    assert f"https://{INTERN}{APB}" not in found(INTERN, 0)


def test_the_faculty_hub_leads_to_its_po_pdf_and_not_to_the_universitys_original():
    urls = found(FB20)
    assert f"https://{FB20}{pages.PO_2023_PDF}" in urls and f"https://{FB20}{pages.APB_COPY_PDF}" in urls
    assert not [u for u in urls if INTERN in u]  # the hub links the original on the other host: not this host's page
    assert f"https://{FB20}{pages.PO_2023_PDF}" not in found(FB20, 0)
