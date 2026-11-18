"""The example profile's source: every Markdown file in the overlay's
connectors.<source>.dir is one document, titled by its first line ("# <title>"), its url the profile entry's base_url plus the
file name. No year and no date: the assistant's ranking would read the clock (the harness fixtures' rule)."""

from rag_core.domain.sources import PageItem, SourceResult
from rag_core.ports.sources import SourceContext


def page(path, base_url):
    text = path.read_text(encoding="utf-8")
    first = text.split("\n", 1)[0]
    if not first.startswith("# ") or not first[2:].strip():
        raise ValueError(f"{path.name}: the first line must be '# <title>'")
    url = f"{base_url}{path.stem}/"
    meta = {
        "title": first[2:].strip(),
        "url": url,
        "source_type": "page",
        "doc_type": "Gesetz",
        "origin": "website",
        "section": "gesetze",
        "doc_year": None,
        "doc_date": None,
        "page_number": -1,  # not a PDF page, as for a crawled web page
        "doc_key": url,
    }
    return PageItem(url, text, meta)


class Fixture:
    """The profile entry's keys: base_url (required), min_documents (the source gate)."""

    name = "example.fixture"
    state_files: tuple[str, ...] = ()

    def collect(self, ctx: SourceContext) -> SourceResult:
        if ctx.data_dir is None:
            raise RuntimeError("example.fixture: the overlay names no directory (connectors.<source>.dir)")
        base_url = ctx.settings.get("base_url")
        if not base_url:
            raise RuntimeError("example.fixture: the profile entry has no base_url")
        files = sorted(ctx.data_dir.glob("*.md"))[: ctx.limit or None]
        return SourceResult(
            [page(path, base_url) for path in files],
            {"files": len(files)},
            expected_minimum=int(ctx.settings.get("min_documents", 0)),
        )
