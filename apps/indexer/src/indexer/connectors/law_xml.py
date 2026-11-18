"""gesetze-im-internet.de XML -> Markdown, one '## § N Titel' section per norm.

Usage: python3 -I laws_to_md.py IN.xml OUT.md --title T --source-url U
           [--doc-type Gesetz|Verordnung] [--from "§ 535"] [--to "§ 580a"]
Stdlib only. Editorial footnotes (textdaten/fussnoten) are dropped. Unknown XML
tags raise, so a changed layout fails loudly instead of losing text.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET

SUP = {"2": "²", "3": "³"}
SOURCE_NOTE = "Quelle: gesetze-im-internet.de, abgerufen am 29.10.2026; amtliches Werk, § 5 UrhG"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("out")
    ap.add_argument("--title", required=True)
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--doc-type", default="Gesetz", choices=["Gesetz", "Verordnung"])
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    a = ap.parse_args()

    root = ET.parse(a.xml).getroot()
    stand = ""
    for n in root:  # the first norm carries the Stand line
        for sa in n.findall("metadaten/standangabe"):
            if sa.findtext("standtyp") == "Stand":
                stand = norm(sa.findtext("standkommentar"))
        if stand:
            break
    if not stand:
        sys.exit("no Stand line found")

    q = json.dumps
    md = [
        "---",
        f"title: {q(a.title, ensure_ascii=False)}",
        f"doc_type: {a.doc_type}",
        "origin: gesetz",
        f"stand: {q(stand, ensure_ascii=False)}",
        f"url: {q(a.source_url)}",
        "---",
        "",
    ]
    stack, pending, on, seen_end = [], False, a.start is None, False
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
        if a.start and enbez == a.start:
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
        if a.end and enbez == a.end:
            seen_end = True
            break
    if a.start and not on or a.end and not seen_end:
        sys.exit("--from/--to not found in XML")
    md += ["---", "", SOURCE_NOTE, ""]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
