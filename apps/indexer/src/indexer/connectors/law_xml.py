"""The law connector: statutes from the official XML of gesetze-im-internet.de, one document per configured law.

sources.yaml lists the laws (`laws`: title, kind, the zip with the XML, the page citations link to and the first and last
norm to take). The zip is fetched through the crawl fetcher, so the host allow-list, robots.txt, the rate limit and the cache
apply. `law_markdown` turns the XML into Markdown with one "## § N Titel" section per norm; editorial footnotes are dropped,
and an unknown XML tag raises, so a changed layout fails loudly instead of losing text."""

import io
import re
import zipfile
from dataclasses import asdict
from datetime import date
from typing import Any

import lxml.etree as etree

from rag_core.domain.sources import PageItem, SourceResult
from rag_core.ports.sources import SourceContext
from rag_core.settings.profile import Law

from indexer.connectors.crawl_setup import load_sources, make_fetcher
from indexer.crawl.fetch import Response, Skipped
from indexer.crawl.manifest import ManifestDoc

XML = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, remove_comments=True, remove_pis=True)
MAX_XML_BYTES = 64 * 1024 * 1024

SUP = {"2": "²", "3": "³"}


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def inline(el):
    """Inline text of an element: SUB/SUP rendered, line breaks as \0."""
    s = el.text or ""
    for ch in el:
        if ch.tag == "BR":
            s += "\0"
        elif ch.tag == "SUB":
            s += "_" + inline(ch)
        elif ch.tag == "SUP":
            t = inline(ch)
            s += SUP.get(t, "^" + t)
        else:
            s += inline(ch)
        s += ch.tail or ""
    return s


def plain(el):
    """Normalised text; a hyphen at a line break is joined (Heiz-<BR/>wert)."""
    s = re.sub(r"(\w)-\s*\0\s*([a-zäöüß])", r"\1\2", inline(el))
    return norm(s.replace("\0", " "))


def walk(el, buf, out):
    """Append inline text to buf, finished blocks (str paragraph or list of lines) to out."""

    def flush():
        t = norm("".join(buf))
        buf.clear()
        if t:
            out.append(t)

    if el.text:
        buf.append(el.text)
    for ch in el:
        tag = ch.tag
        if tag == "BR":
            buf.append(" ")
        elif tag in ("noindex", "LA", "DD", "P", "Content"):
            walk(ch, buf, out)
        elif tag in ("B", "I"):
            mark = "**" if tag == "B" else "*"
            inner = plain(ch)
            if inner:
                buf.append(f"{mark}{inner}{mark}")
        elif tag == "SUB":
            buf.append("_" + plain(ch))
        elif tag == "SUP":
            t = plain(ch)
            buf.append(SUP.get(t, "^" + t))
        elif tag == "IMG":
            buf.append(f" [Formel im Original als Bild ({ch.get('SRC')}), hier nicht wiedergegeben] ")
        elif tag == "DL":
            flush()
            out.append(dl_lines(ch, 0))
        elif tag == "table":
            flush()
            out.append(table(ch))
        elif tag == "pre":
            flush()
            out.append(["```"] + pre_lines(ch) + ["```"])
        else:
            raise ValueError(f"unknown tag <{tag}>")
        if ch.tail:
            buf.append(ch.tail)
    if el.tag in ("P", "Content", "text"):
        flush()


def pre_lines(pre):
    s = pre.text or ""
    for ch in pre:
        s += "\n" if ch.tag == "BR" else plain(ch)
        s += ch.tail or ""
    return [ln.rstrip() for ln in s.split("\n")]


def dl_lines(dl, depth):
    """DL -> nested Markdown list; the label (1., a)) stays as text."""
    lines, label = [], ""
    ind = "  " * depth
    for ch in dl:
        if ch.tag == "DT":
            label = plain(ch)
        elif ch.tag == "DD":
            buf, out = [], []
            walk(ch, buf, out)
            if buf:
                out.append(norm("".join(buf)))
            head = out[0] if out and isinstance(out[0], str) else ""
            rest = out[1:] if head else out
            if head or label:
                lines.append(f"{ind}- {label} {head}".replace("-  ", "- ").rstrip())
            for blk in rest:
                if isinstance(blk, str):
                    lines.append(f"{ind}  {blk}")
                elif blk and blk[0].startswith("```"):
                    lines.extend(f"{ind}  {ln}" for ln in blk)
                else:  # nested list built at depth 0: indent it (or keep level if no bullet above)
                    shift = "  " * (depth + 1) if (head or label) else ind
                    lines.extend(shift + ln for ln in blk)
            label = ""
        else:
            raise ValueError(f"unknown tag <{ch.tag}> in DL")
    return lines


def table(t):
    rows = [[plain(e) for e in r.iter("entry")] for r in t.iter("row")]
    cells = [c for r in rows for c in r if c]
    if len(cells) == 1:  # one-cell layout table = a displayed formula
        return cells[0]
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    if t.find(".//thead") is None:  # Markdown needs a header row
        rows.insert(0, [""] * n)

    def esc(r):
        return "| " + " | ".join(c.replace("|", "\\|") for c in r) + " |"

    return [esc(rows[0]), "|" + " --- |" * n] + [esc(r) for r in rows[1:]]


def section_body(textel):
    out = []
    for p in textel.find("Content"):
        buf = []
        walk(p, buf, out)
    parts = []
    for b in out:
        parts.append(b if isinstance(b, str) else "\n".join(b))
    return "\n\n".join(parts)


def law_markdown(xml: bytes, start: str | None = None, end: str | None = None) -> tuple[str, str]:
    """(the Stand line, the Markdown) of the norms from `start` to `end` (the XML's `enbez`, e.g. "§ 16b"; None takes the
    first or the last). ValueError when there is no Stand line, a norm has an unknown tag or the range is not in the XML."""
    root = etree.fromstring(xml, XML)
    stand = ""
    for n in root:  # the first norm carries the Stand line
        for sa in n.findall("metadaten/standangabe"):
            if sa.findtext("standtyp") == "Stand":
                stand = norm(sa.findtext("standkommentar") or "")
        if stand:
            break
    if not stand:
        raise ValueError("no Stand line found")
    md: list[str] = []
    stack: list[str] = []
    pending, on, seen_end = False, start is None, False
    for n in root:
        m = n.find("metadaten")
        if m is None:
            continue
        g = m.find("gliederungseinheit")
        enbez = m.findtext("enbez")
        if g is not None:
            depth = len(g.findtext("gliederungskennzahl") or "") // 3
            del stack[depth - 1 :]
            stack.append(norm((g.findtext("gliederungsbez") or "") + " " + (g.findtext("gliederungstitel") or "")))
            pending = True
            continue
        if not enbez or not enbez.startswith(("§", "Anhang", "Anlage", "Art")):
            continue  # Inhaltsübersicht, Eingangsformel etc.
        if start and enbez == start:
            on = True
        if not on:
            continue
        if pending and stack:
            md += ["# " + " > ".join(stack), ""]
            pending = False
        titel = norm(m.findtext("titel") or "")
        md += [f"## {enbez} {titel}".rstrip(), ""]
        t = n.find("textdaten/text")
        body = section_body(t) if t is not None else ""
        if body:
            md += [body, ""]
        if end and enbez == end:
            seen_end = True
            break
    if start and not on or end and not seen_end:
        raise ValueError(f"the norms {start or 'first'} to {end or 'last'} are not in the XML")
    return stand, "\n".join(md)


def xml_of_zip(archive: bytes) -> bytes:
    """The one XML file of a law's zip; ValueError for anything else or a file that unpacks to more than MAX_XML_BYTES."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = [i for i in zf.infolist() if i.filename.lower().endswith(".xml")]
        if len(names) != 1 or names[0].file_size > MAX_XML_BYTES:
            raise ValueError("the zip does not hold exactly one XML file of a reasonable size")
        return zf.read(names[0])


def source_note(fetched: str) -> str:
    return f"Quelle: gesetze-im-internet.de, abgerufen am {date.fromisoformat(fetched).strftime('%d.%m.%Y')}; amtliches Werk, § 5 UrhG"


def document(law: Law, reply: Response) -> PageItem:
    """The law as one page document: its title, the Stand line, the norms and the source note."""
    stand, markdown = law_markdown(xml_of_zip(reply.body), law.start, law.end)
    text = f"# {law.title}\n\nStand: {stand}\n\n{markdown}\n---\n\n{source_note(reply.fetched)}\n"
    return PageItem(
        law.page_url,
        text,
        {
            "title": law.title,
            "url": law.page_url,
            "source_type": "page",
            "doc_type": law.kind,
            "origin": "gesetz",
            "section": "",
            "doc_year": None,
            "doc_date": None,
            "page_number": -1,  # not a PDF page, as for a crawled web page
            "doc_key": law.page_url,
            "regulation": "residence",
            "lang": "de",
        },
    )


class LawXml:
    """The profile entry's `min_documents` is the source gate; a law that cannot be fetched or read is listed under
    `skipped` in the report. `fetcher_options` are Fetcher's own (a test injects a client and a clock)."""

    name = "law_xml"
    state_files: tuple[str, ...] = ()

    def __init__(self, **fetcher_options: Any) -> None:
        self.fetcher_options = fetcher_options

    def collect(self, ctx: SourceContext) -> SourceResult:
        sources = load_sources(ctx)
        today = self.fetcher_options.get("today", date.today)()
        fetcher = make_fetcher(ctx, sources.hosts, since=today, **self.fetcher_options)
        items: list[PageItem] = []
        skipped: list[dict[str, str]] = []
        manifest: list[dict[str, Any]] = []
        for law in sources.laws[: ctx.limit]:
            reply = fetcher.get(law.xml_url)
            if isinstance(reply, Skipped):
                skipped.append({"url": law.xml_url, "reason": reply.reason})
                continue
            try:
                item = document(law, reply)
                items.append(item)
                meta = {k: item.metadata[k] for k in ("regulation", "lang")}
                manifest.append(asdict(ManifestDoc(item.url, reply.content_hash, reply.fetched, meta)))
            except (ValueError, zipfile.BadZipFile, etree.XMLSyntaxError) as exc:
                skipped.append({"url": law.xml_url, "reason": f"unreadable: {exc}"})
        return SourceResult(
            items,
            {"laws": len(items), "skipped": skipped, "manifest": manifest},
            expected_minimum=int(ctx.settings.get("min_documents", 0)),
        )
