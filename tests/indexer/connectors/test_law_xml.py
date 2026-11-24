"""The law connector: the XML of a law becomes Markdown with one section per norm of the configured range, and the connector
fetches the zip through the crawl fetcher and hands the law over as one page document. The XML is written here and served
in-process; nothing is downloaded."""

import io
import zipfile
from datetime import date
from importlib.metadata import entry_points

import httpx
import pytest
from rag_core.domain.sources import PageItem

from indexer.connectors.law_xml import LawXml, law_markdown

HOST = "https://www.gesetze-im-internet.de"
TODAY = date(2026, 11, 18)


def norm(enbez: str, title: str, body: str = "", footnote: str = "") -> str:
    return (
        f'<norm><metadaten><jurabk>AufenthG</jurabk><enbez>{enbez}</enbez><titel format="parat">{title}</titel></metadaten>'
        f'<textdaten><text format="XML"><Content>{body}</Content></text>{footnote}</textdaten></norm>'
    )


def law_xml(abbreviation: str, norms: str, stand: str = "Zuletzt geändert durch Art. 1 G v. 1.1.2026 I Nr. 1") -> bytes:
    head = (
        f'<norm><metadaten><jurabk>{abbreviation}</jurabk><standangabe checked="ja"><standtyp>Stand</standtyp>'
        f"<standkommentar>{stand}</standkommentar></standangabe></metadaten></norm>"
        "<norm><metadaten><jurabk>X</jurabk><enbez>Eingangsformel</enbez></metadaten></norm>"
    )
    return f'<?xml version="1.0" encoding="UTF-8"?><dokumente doknr="X">{head}{norms}</dokumente>'.encode()


STUDY = (
    "<P>(1) Einem Ausländer wird zum Zweck des Vollzeitstudiums eine Aufenthaltserlaubnis erteilt, wenn er zugelassen "
    "worden ist.</P><P>(2) Die Aufenthaltserlaubnis wird erteilt<BR/>für mindestens</P>"
    '<P><DL Type="arabic" Font="normal"><DT>1.</DT><DD Font="normal"><LA Size="normal">ein Jahr bei der Einreise,'
    '</LA></DD><DT>2.</DT><DD Font="normal"><LA Size="normal">zwei Jahre bei der Verlängerung.</LA></DD></DL></P>'
)
AUFENTHG = law_xml(
    "AufenthG",
    "<norm><metadaten><jurabk>AufenthG</jurabk><gliederungseinheit><gliederungskennzahl>002</gliederungskennzahl>"
    "<gliederungsbez>Kapitel 2</gliederungsbez><gliederungstitel>Einreise und Aufenthalt</gliederungstitel>"
    "</gliederungseinheit></metadaten></norm>"
    + norm("§ 16", "Grundsatz", "<P>Der Aufenthalt zum Studium wird in diesem Abschnitt geregelt.</P>")
    + norm(
        "§ 16b",
        "Studium",
        STUDY,
        footnote="<fussnoten><Content><P>Fußnote der Redaktion, die nicht in den Index gehört.</P></Content></fussnoten>",
    )
    + norm("§ 16c", "Mobilität", "<P>Eine andere Vorschrift.</P>"),
)
AUFENTHV = law_xml(
    "AufenthV", norm("§ 44", "Vorher", "<P>Davor.</P>") + norm("§ 45", "Nachweise", "<P>Nachweise zum Studium.</P>")
)


def zipped(xml: bytes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("BJNR.xml", xml)
    return out.getvalue()


def serve(served, zips: dict[str, bytes]):
    def handler(request: httpx.Request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, headers={"content-type": "text/plain"}, text="User-agent: *\nDisallow:\n")
        body = zips.get(str(request.url))
        return httpx.Response(200, headers={"content-type": "application/zip"}, content=body) if body else None

    return served(handler)


ZIPS = {
    f"{HOST}/aufenthg_2004/xml.zip": zipped(AUFENTHG),
    f"{HOST}/aufenthv/xml.zip": zipped(AUFENTHV),
}


def connector(net) -> LawXml:
    return LawXml(client=net.client, interval=0, today=lambda: TODAY)


def test_law_xml_builds_16b_section():
    stand, markdown = law_markdown(AUFENTHG, "§ 16b", "§ 16b")
    assert stand == "Zuletzt geändert durch Art. 1 G v. 1.1.2026 I Nr. 1"
    assert markdown.startswith("# Kapitel 2 Einreise und Aufenthalt\n\n## § 16b Studium\n\n(1) Einem Ausländer")
    assert "(2) Die Aufenthaltserlaubnis wird erteilt für mindestens" in markdown
    assert "- 1. ein Jahr bei der Einreise," in markdown and "- 2. zwei Jahre bei der Verlängerung." in markdown
    assert "Fußnote" not in markdown  # editorial footnotes are dropped
    assert "§ 16c" not in markdown and "§ 16 " not in markdown and "Eingangsformel" not in markdown


def test_a_range_without_ends_takes_every_norm_and_a_missing_one_is_an_error():
    _, everything = law_markdown(AUFENTHG)
    assert [line for line in everything.splitlines() if line.startswith("## ")] == [
        "## § 16 Grundsatz",
        "## § 16b Studium",
        "## § 16c Mobilität",
    ]
    with pytest.raises(ValueError, match="not in the XML"):
        law_markdown(AUFENTHG, "§ 99", "§ 99")
    with pytest.raises(ValueError, match="no Stand line"):
        law_markdown(b"<dokumente><norm><metadaten/></norm></dokumente>")


def test_an_unknown_tag_fails_loudly_instead_of_losing_text():
    changed = law_xml("X", norm("§ 1", "Neu", "<P>Text <NEU>mit neuem Tag</NEU></P>"))
    with pytest.raises(ValueError, match="unknown tag <NEU>"):
        law_markdown(changed)


def test_the_connector_hands_each_law_over_as_one_page_document(make_context, served):
    net = serve(served, ZIPS)
    result = connector(net).collect(make_context(settings={"min_documents": 2}))
    first, second = result.items
    assert isinstance(first, PageItem) and isinstance(second, PageItem)
    assert first.url == f"{HOST}/aufenthg_2004/__16b.html"
    assert first.text.startswith("# Aufenthaltsgesetz (AufenthG), § 16b Studium\n\nStand: Zuletzt geändert")
    assert "## § 16b Studium" in first.text and "§ 16c" not in first.text
    assert first.text.rstrip().endswith(
        "Quelle: gesetze-im-internet.de, abgerufen am 18.11.2026; amtliches Werk, § 5 UrhG"
    )
    assert "## § 45 Nachweise" in second.text and "§ 44" not in second.text
    assert dict(first.metadata) | {"title": "-"} == {
        "title": "-",
        "url": first.url,
        "source_type": "page",
        "doc_type": "Gesetz",
        "origin": "gesetz",
        "section": "",
        "doc_year": None,
        "doc_date": None,
        "page_number": -1,
        "doc_key": first.url,
        "regulation": "residence",
        "lang": "de",
    }
    assert second.metadata["doc_type"] == "Verordnung"
    assert (result.expected_minimum, result.report["laws"], result.report["skipped"]) == (2, 2, [])
    assert [(m["url"], m["metadata"]) for m in result.report["manifest"]] == [
        (item.url, {"regulation": "residence", "lang": "de"}) for item in result.items
    ]  # the snapshot manifest lists each law page with the date it was fetched
    assert all(len(m["content_hash"]) == 64 and m["fetched"] for m in result.report["manifest"])


def test_a_law_that_is_down_or_unreadable_is_listed_not_indexed(make_context, served):
    zips = {f"{HOST}/aufenthv/xml.zip": b"not a zip"}
    result = connector(serve(served, zips)).collect(make_context())
    assert result.items == []
    assert [(s["url"], s["reason"].split(":")[0]) for s in result.report["skipped"]] == [
        (f"{HOST}/aufenthg_2004/xml.zip", "http 404"),
        (f"{HOST}/aufenthv/xml.zip", "unreadable"),
    ]


def test_the_zip_goes_through_the_fetchers_rules_and_the_cache(make_context, served):
    net = serve(served, ZIPS)
    ctx = make_context()
    connector(net).collect(ctx)
    assert net.urls().count(f"{HOST}/robots.txt") == 1  # robots.txt is read once for the host
    again = serve(served, ZIPS)
    connector(again).collect(ctx)  # fetched today already: answered from the cache
    assert [u for u in again.urls() if u.endswith(".zip")] == []
    frozen = serve(served, {})
    result = connector(frozen).collect(make_context(reuse=True))
    assert len(result.items) == 2 and frozen.log == []  # frozen inputs: the cache alone, no request at all


def test_the_connector_is_found_by_its_entry_point():
    found = {ep.name: ep for ep in entry_points(group="rag_core.connectors")}
    assert found["law_xml"].load() is LawXml
