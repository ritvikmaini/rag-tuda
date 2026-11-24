"""Discovery against the fixture mini-site served in-process: sitemaps filtered by prefix and exclusion, link-following to a
depth, seed-only hosts, the host rules that stay with the fetcher, and English twins. No request leaves the process."""

import httpx
from fastapi.testclient import TestClient
from rag_testing import serve
from rag_testing.minisite import pages

from indexer.crawl.discover import GERMAN_ONLY_NOTICE, Source, discover, in_scope, normalise
from indexer.crawl.fetch import Fetcher

FACULTY = "faculty.test"
INTERN = "intern.test"
SITEMAP = f"http://{FACULTY}/sitemap.de.xml"
ORDNUNGEN_INDEX = f"http://{FACULTY}{pages.ORDNUNGEN}/index.de.jsp"
HUB = f"http://{FACULTY}{pages.BSC_HUB}.de.jsp"
EXCLUDED_HUB = f"http://{FACULTY}{pages.EXCLUDED_HUB}.de.jsp"
STUDIENBUERO_DE = f"http://{FACULTY}{pages.STUDIENBUERO}/index.de.jsp"
STUDIENBUERO_EN = f"http://{FACULTY}{pages.STUDIENBUERO}/index.en.jsp"
PRUEFUNGEN_DE = f"http://{FACULTY}{pages.STUDIENBUERO}/pruefungen/index.de.jsp"
PRUEFUNGEN_PRINT = f"{PRUEFUNGEN_DE}?view=print"
NOAI = f"http://{FACULTY}{pages.NOAI_PAGE}"
LOGIN = f"http://{FACULTY}{pages.LOGIN_PAGE}"
INTERN_INDEX = f"http://{INTERN}{pages.INTERN_INDEX}"
STUDIUM = "/studium_fb20/"


class Net:
    """The mini-site's two hosts behind an httpx client; `extra` maps a URL to a (media type, body) served in their place and
    `log` records every request that reaches the transport."""

    def __init__(self, extra=None, redirects=None):
        self.log: list[str] = []
        self.extra = extra or {}
        self.redirects = redirects or {}  # url -> the Location to answer with
        self.apps = {
            FACULTY: TestClient(serve.minisite_app("fb20", f"http://{INTERN}"), base_url=f"http://{FACULTY}"),
            INTERN: TestClient(serve.minisite_app("intern", f"http://{INTERN}"), base_url=f"http://{INTERN}"),
        }
        self.client = httpx.Client(transport=httpx.MockTransport(self._serve))

    def _serve(self, request: httpx.Request) -> httpx.Response:
        self.log.append(str(request.url))
        if str(request.url) in self.redirects:
            return httpx.Response(302, headers={"location": self.redirects[str(request.url)]})
        if str(request.url) in self.extra:
            content_type, body = self.extra[str(request.url)]
            return httpx.Response(200, headers={"content-type": content_type}, content=body)
        app = self.apps.get(request.url.host)
        if app is None:
            raise httpx.ConnectError("no such host", request=request)
        reply = app.get(request.url.raw_path.decode())
        return httpx.Response(
            reply.status_code, headers={"content-type": reply.headers["content-type"]}, content=reply.content
        )


def src(host, **kw):
    """A source on the mini-site's plain-http hosts."""
    return Source(host, scheme="http", **kw)


def fetcher(net, hosts=(FACULTY, INTERN)):
    return Fetcher(hosts, contact="dummy@example.invalid", client=net.client, interval=0)


def run(source, net=None, **kw):
    net = net or Net()
    return discover(source, fetcher(net, **kw)), net


def test_sitemap_filtered_by_prefix_and_exclusions():
    source = src(FACULTY, prefixes=(STUDIUM,), exclude=(pages.EXCLUDED_HUB,), sitemaps=(SITEMAP,))
    found, net = run(source)
    assert found == [ORDNUNGEN_INDEX, HUB, STUDIENBUERO_DE, PRUEFUNGEN_DE, NOAI]
    assert EXCLUDED_HUB not in net.log  # a sitemap entry is listed, never fetched
    narrow, _ = run(src(FACULTY, prefixes=(pages.STUDIENBUERO + "/",), sitemaps=(SITEMAP,)))
    assert narrow == [STUDIENBUERO_DE, PRUEFUNGEN_DE, NOAI]


def test_sitemap_index_is_followed_and_foreign_hosts_are_left_out():
    child = f"http://{FACULTY}/sitemap-1.xml"
    ns = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
    extra = {
        SITEMAP: ("application/xml", f"<sitemapindex {ns}><sitemap><loc>{child}</loc></sitemap></sitemapindex>"),
        child: (
            "application/xml",
            f"<urlset {ns}><url><loc> {HUB}#top </loc></url><url><loc>http://{INTERN}/x.de.jsp</loc></url>"
            f"<url><loc>{HUB}</loc></url></urlset>",
        ),
    }
    found, _ = run(src(FACULTY, prefixes=("/",), sitemaps=(SITEMAP,)), Net(extra))
    assert found == [HUB]  # the fragment is dropped, the duplicate and the other host's entry are not listed


def test_sitemap_does_not_resolve_external_entities(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    bomb = (
        f'<?xml version="1.0"?><!DOCTYPE urlset [<!ENTITY x SYSTEM "{secret.as_uri()}">]>'
        f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>http://{FACULTY}/a/&x;.de.jsp</loc></url></urlset>'
    )
    found, _ = run(src(FACULTY, prefixes=("/",), sitemaps=(SITEMAP,)), Net({SITEMAP: ("application/xml", bomb)}))
    assert not any("TOPSECRET" in url for url in found)


def test_unreadable_sitemap_yields_nothing():
    found, _ = run(src(FACULTY, prefixes=("/",), sitemaps=(SITEMAP,)), Net({SITEMAP: ("application/xml", "<urlset")}))
    assert found == []


def test_link_following_depth():
    def at(depth):
        source = src(FACULTY, prefixes=(STUDIUM,), exclude=(pages.EXCLUDED_HUB,), seeds=(ORDNUNGEN_INDEX,), depth=depth)
        return run(source)

    found, net = at(0)
    assert found == [ORDNUNGEN_INDEX] and net.log == []
    found, _ = at(1)
    assert set(found) == {
        ORDNUNGEN_INDEX,
        HUB,
        STUDIENBUERO_DE,
        STUDIENBUERO_EN,
    }  # the excluded hub is linked but left out
    found, net = at(3)
    # the print view, the login page and the noai page were linked at depth 2; fetching them at depth 3 dropped them
    assert set(found) == {ORDNUNGEN_INDEX, HUB, STUDIENBUERO_DE, STUDIENBUERO_EN, PRUEFUNGEN_DE}
    assert NOAI in net.log  # fetched, found noai, dropped
    assert not {LOGIN, PRUEFUNGEN_PRINT, EXCLUDED_HUB} & set(net.log)  # robots and exclusions keep them from the wire


def test_link_following_stays_on_the_host_and_skips_pdfs():
    source = src(FACULTY, prefixes=("/",), seeds=(HUB,), depth=2)
    found, net = run(source)
    assert all(
        url.startswith(f"http://{FACULTY}/") for url in found
    )  # the APB original on the other host is not followed
    assert f"http://{FACULTY}{pages.PO_2023_PDF}" in found
    assert not any(url.endswith(".pdf") for url in net.log)  # listed, never downloaded while following links


def test_seed_only_host():
    source = src(INTERN, prefixes=(), seeds=(INTERN_INDEX + "#apb",))
    found, net = run(source)
    assert found == [INTERN_INDEX]  # fragment dropped, nothing else listed
    assert net.log == []  # depth 0: not even fetched
    assert normalise(f"http://{INTERN}/a?b=1&c=2#x") == f"http://{INTERN}/a?b=1&c=2"


def test_host_allow_list_excludes_hda():
    sitemap = "https://h-da.de/sitemap.xml"
    seeds = ("https://h-da.de/studium/", "https://www.darmstadt.de/")
    listing = (
        f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{HUB}</loc></url>'
        "<url><loc>https://h-da.de/studium/</loc></url><url><loc>https://www.darmstadt.de/a.de.jsp</loc></url></urlset>"
    )
    net = Net({SITEMAP: ("application/xml", listing)})
    assert run(src(FACULTY, prefixes=("/",), sitemaps=(SITEMAP,)), net)[0] == [HUB]  # other hosts are not this source's
    # even on the allow-list, the never-fetched hosts are not reached
    net = Net()
    hosts = (FACULTY, "h-da.de", "www.darmstadt.de")
    assert (
        run(src("h-da.de", prefixes=("/",), sitemaps=(sitemap,), seeds=seeds[:1], depth=1), net, hosts=hosts)[0] == []
    )
    assert run(src("www.darmstadt.de", prefixes=("/",), seeds=seeds[1:], depth=1), net, hosts=hosts)[0] == []
    assert net.log == []


def test_en_twin_dropped_on_german_body():
    seeds = (HUB, STUDIENBUERO_DE, PRUEFUNGEN_DE, NOAI)
    found, net = run(src(FACULTY, prefixes=(STUDIUM,), seeds=seeds, twins=True))
    # the hub twin carries the notice, the Pruefungen twin has a German body without it, the noai page has no twin at all
    assert found == [*seeds[:1], STUDIENBUERO_DE, PRUEFUNGEN_DE, NOAI, STUDIENBUERO_EN]
    assert STUDIENBUERO_EN in net.log and HUB.replace(".de.", ".en.") in net.log


ENGLISH_CHROME = '<nav class="main-nav">Contact us. Please read the notes about the use of this page and the data that we keep.</nav>'
LINKS_ONLY = (
    '<ul><li><a href="/media/ordnungen_cs/studien-_und_pruefungsplan_der_ordnung_fuer_das_studium.en.pdf">Study and '
    'examination plan</a></li><li><a href="/media/ordnungen_cs/modulhandbuch_und_ordnung_der_studiengaenge.en.pdf">'
    "Module handbook for the programme</a></li></ul>"
)


def twin_page(main: str, outside: str = "") -> tuple[str, str]:
    """An English twin of the study office page: English page chrome around `.main-column` (and `outside` beside it)."""
    html = f'<html><body>{ENGLISH_CHROME}{outside}<div class="main-column">{main}</div></body></html>'
    return "text/html", html


def twins_found(twin: tuple[str, str]) -> list[str]:
    found, _ = run(
        src(FACULTY, prefixes=(STUDIUM,), seeds=(STUDIENBUERO_DE,), twins=True), Net(extra={STUDIENBUERO_EN: twin})
    )
    return found


def test_twin_language_is_read_from_the_main_content():
    german = (
        "<p>Das Studienbüro berät zu Prüfungen, Anmeldungen und Abschlussarbeiten. Die Sprechstunde ist dienstags.</p>"
    )
    assert twins_found(twin_page(german)) == [STUDIENBUERO_DE]  # German text between English page chrome
    assert twins_found(twin_page("")) == [STUDIENBUERO_DE]  # nothing to index
    english = "<p>The study office advises on examinations, registrations and theses. Office hours are on Tuesdays.</p>"
    assert twins_found(twin_page(english)) == [STUDIENBUERO_DE, STUDIENBUERO_EN]


def test_twin_that_is_a_list_of_links_is_judged_by_the_link_texts():
    # the words of the German file names in the link targets do not make the page German
    assert twins_found(twin_page(LINKS_ONLY)) == [STUDIENBUERO_DE, STUDIENBUERO_EN]


def test_german_only_notice_outside_the_main_content_drops_the_twin():
    notice = f'<p class="notice">{GERMAN_ONLY_NOTICE}.</p>'
    english = "<p>The study office advises on examinations, registrations and theses. Office hours are on Tuesdays.</p>"
    assert twins_found(twin_page(english, outside=notice)) == [STUDIENBUERO_DE]


def test_linked_en_pages_go_through_the_twin_check():
    seeds = (STUDIENBUERO_DE,)
    found, _ = run(src(FACULTY, prefixes=(STUDIUM,), seeds=seeds, depth=2, twins=True))
    assert STUDIENBUERO_EN in found
    assert f"http://{FACULTY}{pages.STUDIENBUERO}/pruefungen/index.en.jsp" not in found


def test_normalise_resolves_the_path_once():
    base = f"http://{FACULTY}"
    assert normalise(f"{base}/studium_fb20/../intern/x") == f"{base}/intern/x"
    assert normalise(f"{base}/studium_fb20/%2e%2E/intern/x") == f"{base}/intern/x"
    assert normalise(f"{base}/a//b/./c/") == f"{base}/a/b/c/"
    assert normalise(f"{base}/a/b/..") == f"{base}/a/"
    assert normalise(f"{base}/../../x") == f"{base}/x"
    assert normalise(f"{base}/%7Euser/a%2Fb/%e4") == f"{base}/~user/a%2Fb/%E4"  # only unreserved characters are decoded
    assert normalise(base) == f"{base}/"


def test_dot_segments_cannot_lead_out_of_a_prefix():
    base = f"http://{FACULTY}"
    entries = (
        f"{base}/studium_fb20/../intern/a.de.jsp",
        f"{base}/studium_fb20/%2e%2e/intern/b.de.jsp",
        f"{base}/studium_fb20/x/..//../intern/c.de.jsp",
        f"{base}/studium_fb20/{pages.EXCLUDED_HUB.rsplit('/', 1)[1]}/../../x.de.jsp",
        f"{base}{pages.EXCLUDED_HUB.replace('/ordnungen_visual', '//./ordnungen_visual')}.de.jsp",
        f"{base}/studium_fb20/in/../ok.de.jsp",
    )
    listing = "".join(f"<url><loc>{u}</loc></url>" for u in entries)
    ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
    extra = {SITEMAP: ("application/xml", f'<urlset xmlns="{ns}">{listing}</urlset>')}
    source = src(FACULTY, prefixes=(STUDIUM,), exclude=(pages.EXCLUDED_HUB,), sitemaps=(SITEMAP,))
    assert run(source, Net(extra))[0] == [f"{base}/studium_fb20/ok.de.jsp"]


def test_in_scope_checks_the_normalised_url_against_origin_prefixes_and_exclusions():
    source = src(FACULTY, prefixes=(STUDIUM,), exclude=(pages.EXCLUDED_HUB,))
    base = f"http://{FACULTY}"
    assert in_scope(source, f"{base}/studium_fb20/in/../ok.de.jsp#top")
    assert not in_scope(source, f"{base}/studium_fb20/%2e%2e/intern/a.de.jsp")
    assert not in_scope(source, f"{base}{pages.EXCLUDED_HUB}.de.jsp")
    assert not in_scope(source, f"http://{INTERN}/studium_fb20/ok.de.jsp")


def test_a_link_with_dot_segments_is_checked_and_requested_as_normalised():
    ns = '<a href="/studium_fb20/%2e%2e/intern/x.de.jsp">x</a><a href="/studium_fb20/a/../b.de.jsp">b</a>'
    seed = f"http://{FACULTY}/studium_fb20/start.de.jsp"
    net = Net({seed: ("text/html", f"<html><body>{ns}</body></html>")})
    found, net = run(src(FACULTY, prefixes=(STUDIUM,), seeds=(seed,), depth=1), net)
    assert set(found) == {seed, f"http://{FACULTY}/studium_fb20/b.de.jsp"}
    assert not [u for u in net.log if "intern" in u or ".." in u or "%2e" in u.lower()]


def test_scheme_host_and_port_are_the_sources():
    base = f"{FACULTY}/studium_fb20/a.de.jsp"
    entries = (
        f"http://{base}",
        f"https://{base}",
        f"http://{FACULTY}:8080/studium_fb20/b.de.jsp",
        f"http://{FACULTY}:80/studium_fb20/c.de.jsp",
    )
    listing = "".join(f"<url><loc>{u}</loc></url>" for u in entries)
    ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
    net = Net({SITEMAP: ("application/xml", f'<urlset xmlns="{ns}">{listing}</urlset>')})
    found, _ = run(src(FACULTY, prefixes=(STUDIUM,), sitemaps=(SITEMAP,)), net)
    assert found == [f"http://{base}", f"http://{FACULTY}:80/studium_fb20/c.de.jsp"]  # the default port is the scheme's
    assert run(Source(FACULTY, prefixes=(STUDIUM,), sitemaps=(SITEMAP,), scheme="http", port=8080), net)[0] == [
        f"http://{FACULTY}:8080/studium_fb20/b.de.jsp"
    ]


def test_a_redirect_out_of_scope_drops_the_page_and_its_links():
    outside = f"http://{FACULTY}/intern/page.de.jsp"
    seed = f"http://{FACULTY}/studium_fb20/moved.de.jsp"
    extra = {outside: ("text/html", f'<html><body><a href="{HUB}">hub</a></body></html>')}
    net = Net(extra, redirects={seed: outside, f"{HUB}": outside})
    source = src(FACULTY, prefixes=(STUDIUM,), seeds=(seed, STUDIENBUERO_DE), depth=1)
    found, net = run(source, net)
    assert seed not in found and outside not in found
    assert HUB not in found  # its link was not followed
    assert STUDIENBUERO_DE in found


def test_a_seed_may_redirect_to_another_seed():
    target = INTERN_INDEX
    found, _ = run(
        src(INTERN, prefixes=(), seeds=(f"http://{INTERN}/start", target), depth=1),
        Net(redirects={f"http://{INTERN}/start": target}),
    )
    assert found == [f"http://{INTERN}/start", target]


def test_a_twin_that_redirects_out_of_scope_is_dropped():
    twin = STUDIENBUERO_DE.replace(".de.", ".en.")
    net = Net(
        redirects={twin: f"http://{FACULTY}/intern/en.html"},
        extra={
            f"http://{FACULTY}/intern/en.html": (
                "text/html",
                "<html><body>The study office advises on the thesis and the exams.</body></html>",
            )
        },
    )
    found, _ = run(src(FACULTY, prefixes=(STUDIUM,), seeds=(STUDIENBUERO_DE,), twins=True), net)
    assert found == [STUDIENBUERO_DE]
