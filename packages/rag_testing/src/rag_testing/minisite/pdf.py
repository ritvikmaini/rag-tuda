"""A minimal PDF writer: Helvetica text on A4 pages, no timestamps, so the same input gives the same bytes. The fixture PDFs
are built when the module loads; no binary file is committed."""

import textwrap

_WIDTH = 86  # characters per line at 11 pt on A4 with the margins below


def _literal(line: str) -> bytes:
    """A PDF string literal in WinAnsi (cp1252): `(`, `)` and `\\` escaped, every non-ASCII byte as an octal escape."""
    out = bytearray(b"(")
    for byte in line.encode("cp1252"):
        if byte in b"()\\":
            out += b"\\" + bytes([byte])
        elif 32 <= byte < 127:
            out.append(byte)
        else:
            out += b"\\%03o" % byte
    return bytes(out + b")")


def _lines(page: str) -> list[str]:
    """Paragraphs separated by blank lines, each wrapped to the page width, with one empty line between them."""
    lines: list[str] = []
    for paragraph in page.strip().split("\n\n"):
        lines += textwrap.wrap(" ".join(paragraph.split()), _WIDTH) + [""]
    return lines[:-1]


def make_pdf(*pages: str) -> bytes:
    """One PDF page per argument; the text of a page is paragraphs separated by blank lines."""
    count = len(pages)
    kids = b" ".join(b"%d 0 R" % (4 + 2 * n) for n in range(count))
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, count),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for n, page in enumerate(pages):
        text = (
            b"BT /F1 11 Tf 15 TL 56 780 Td\n" + b"".join(_literal(line) + b" Tj T*\n" for line in _lines(page)) + b"ET"
        )
        bodies.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> "
            b"/Contents %d 0 R >>" % (5 + 2 * n)
        )
        bodies.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(text), text))
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(bodies, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(bodies) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(bodies) + 1, xref)
    return bytes(out)
