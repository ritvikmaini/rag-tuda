"""Metadata of crawled documents, derived by fixed rules from the URL folder, the section id and heading, the link title
and text and the file name (never by a model).

`page_entries` turns an extracted page into its documents: a section of a programme's hub page that carries its own PO
version ("Prüfungsordnung 2023") is a document of its own, every other section is joined into one document of the page.
`pdf_metadata` classifies a linked PDF; its programme, degree and PO version come from the hub page it is linked from (its
URL folder, the section's id and heading, the link's text and title), not from the PDF's own path, which may not name them
(the PO folder number is not the PO year). `select_documents` applies the fail-closed rule and drops hub copies of a
university-wide regulation whose original is in the same set.

Fields: `regulation` (po, mhb, spp, apb, eso, promotion, residence, info, form); `degree`, `programme` and `po_version`
are tuples or None; None means "applies to all" and is never an empty tuple. The PO, Modulhandbuch and Studien- und
Prüfungsplan documents need a programme and a PO version, and a document without both is listed, not indexed."""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from urllib.parse import unquote, urlsplit

from rag_core.settings.profile import PdfTwin, Roles

from indexer.crawl.extract import Extraction, Link, PageDocument

__all__ = [
    "Candidate",
    "Metadata",
    "PageEntry",
    "Selection",
    "page_entries",
    "pdf_metadata",
    "select_documents",
    "twin_keys",
]

PROGRAMME_BOUND = ("po", "mhb", "spp")  # these need a programme and a PO version
UNIVERSAL = ("apb", "eso", "promotion", "residence")  # these apply beyond one programme
HUB_COPIED = ("apb", "eso")  # university-wide regulations a faculty hub may carry a copy of
# checked in order against the folded text, the most specific kind first (a Modulhandbuch may name the APB); the first
# match is the regulation. A form is checked before the PO: "Erklärung zum PO-Wechsel" is a form, not a PO.
REGULATIONS = (
    ("mhb", re.compile(r"modulhandbuch|\bmhb\b")),
    ("spp", re.compile(r"studien und pruefungsplan|studienplan|\bspp\b")),
    ("apb", re.compile(r"\bapb\b|allg\w* pruefungsbest")),
    ("eso", re.compile(r"\beso\b|einschreibeordnung")),
    ("form", re.compile(r"erklaerung|antrag|formular")),
    ("po", re.compile(r"ordnung")),
)
# a section heading names a regulation only by these kinds (its "Ordnung" alone is no PO); a PO year in it names the PO
HEADING_REGULATIONS = ("mhb", "spp", "apb", "eso")
RESIDENCE = ("aufenthalt_international", "digitales-rathaus.darmstadt.de", "ladadi.de")
PAGE_EXTENSION = re.compile(r"\.(?:de|en)\.jsp$")
DATE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}")
YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
IN_KRAFT = re.compile(r"in\s+kraft\s+seit\s+(\d{1,2})\.(\d{1,2})\.(\d{4})", re.IGNORECASE)
DATE_PREFIX = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_")
ARCHIV = re.compile(r"^\s*archiv\s*:", re.IGNORECASE)
FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


@dataclass(frozen=True)
class Metadata:
    regulation: str
    degree: tuple[str, ...] | None
    programme: tuple[str, ...] | None
    po_version: tuple[int, ...] | None
    archived: bool
    valid_from: date | None
    lang: str
    twin_key: str | None
    official_url: str | None
    # the PO year an "Ausführungsbestimmungen" document names, when no version has it
    unconfigured_year: int | None = None

    @property
    def missing(self) -> tuple[str, ...]:
        """The required fields a programme regulation lacks; empty for every other document. Implementing provisions of
        a PO year that is not a configured version count as lacking the PO version (they would apply to everyone)."""
        if self.unconfigured_year:
            return ("po_version",)
        if self.regulation not in PROGRAMME_BOUND:
            return ()
        return tuple(f for f in ("programme", "po_version") if getattr(self, f) is None)


@dataclass(frozen=True)
class PageEntry:
    """A document of a page: the section it is (None for the joined rest of the page), its Markdown and its metadata."""

    section_id: str | None
    heading: str | None
    markdown: str
    metadata: Metadata


@dataclass(frozen=True)
class Candidate:
    """A document offered to the index: its URL, content hash and metadata."""

    url: str
    sha256: str | None
    metadata: Metadata


@dataclass(frozen=True)
class Selection:
    """`kept` are indexed; `not_indexed` lists (url, missing fields) for the build summary; `copies` are hub copies skipped."""

    kept: tuple[Candidate, ...]
    not_indexed: tuple[tuple[str, tuple[str, ...]], ...]
    copies: tuple[str, ...]


def _fold(*texts: str | None) -> str:
    """Lowercase ASCII words of the texts: umlauts spelled out, every other run of non-alphanumerics one space."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", " ".join(t for t in texts if t).lower().translate(FOLD)).split())


def _programme(url: str, roles: Roles) -> str | None:
    """The programme whose hub page (any language) or PDF folder the URL is under."""
    base = PAGE_EXTENSION.sub("", urlsplit(url)._replace(query="", fragment="").geturl())
    for name, programme in roles.programmes.items():
        rest = base.removeprefix(programme.url_prefix)
        if rest != base and (not rest or rest.startswith("/")):
            return name
        if base.startswith(programme.pdf_prefix):
            return name
    return None


def _years(text: str | None) -> set[int]:
    return {int(y) for y in YEAR.findall(DATE.sub(" ", text or ""))}


def _section_year(section_id: str | None, heading: str | None) -> int | None:
    """The PO year a section names ("pruefungsordnung_2023"), when its id or heading says Ordnung and one year."""
    text = " ".join(t for t in (section_id, heading) if t)
    years = _years(text)
    return next(iter(years)) if "ordnung" in _fold(text) and len(years) == 1 else None


def _version(programme: str | None, roles: Roles, *sources: str | None) -> tuple[int, ...] | None:
    """The PO year of the first source that names exactly one year, when the programme has that version."""
    if not programme:
        return None
    for source in sources:
        years = _years(source)
        if len(years) == 1:
            year = next(iter(years))
            return (year,) if any(v.year == year for v in roles.programmes[programme].po_versions) else None
    return None


def _unconfigured_year(regulation: str, programme: str | None, roles: Roles, *texts: str | None) -> int | None:
    """The one year the texts of an "Ausführungsbestimmungen" document on a programme's hub name (dates and a file name's
    date prefix are no PO year), when the programme has no PO version of that year; None for any other document."""
    if regulation != "apb" or not programme or "ausfuehrungsbestimmung" not in _fold(*texts):
        return None
    years = set().union(*(_years(t) for t in texts))
    configured = {v.year for v in roles.programmes[programme].po_versions}
    return next(iter(years)) if len(years) == 1 and not years & configured else None


def _official_url(programme: str | None, version: tuple[int, ...] | None, roles: Roles) -> str | None:
    if not programme or not version:
        return None
    return next((v.official_url for v in roles.programmes[programme].po_versions if v.year == version[0]), None)


def _valid_from(*texts: str | None, name: str = "") -> date | None:
    for text in texts:
        for m in IN_KRAFT.finditer(text or ""):
            try:
                return date(int(m[3]), int(m[2]), int(m[1]))
            except ValueError:
                continue
    m = DATE_PREFIX.match(name)
    try:
        return date(int(m[1]), int(m[2]), int(m[3])) if m else None
    except ValueError:
        return None


def _twin_key(url: str, section_id: str | None) -> str | None:
    base = PAGE_EXTENSION.sub("", urlsplit(url)._replace(query="", fragment="").geturl())
    if base == urlsplit(url)._replace(query="", fragment="").geturl():
        return None  # not a ".de.jsp"/".en.jsp" page: no twin
    return f"{base}#{section_id}" if section_id else base


def _lang(url: str, stem: str = "") -> str:
    """English for a ".en.jsp"/".en.pdf" file name, a file name ending in "-en" or naming "english" or "englisch", else
    German."""
    folded = _fold(stem)
    english = (
        re.search(r"\.en\.\w+$", urlsplit(url).path) or "english" in folded or folded.endswith((" en", "englisch"))
    )
    return "en" if english else "de"


def _scope(
    regulation: str, programme: str | None, version: tuple[int, ...] | None, roles: Roles
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None, tuple[int, ...] | None]:
    """(degree, programme, po_version): a university-wide regulation has none (a promotion document has its degree)."""
    if regulation == "promotion":
        return ("promotion",), None, None
    if regulation in UNIVERSAL or not programme:
        return None, None, None
    return (roles.programmes[programme].degree,), (programme,), version if regulation in PROGRAMME_BOUND else None


def _regulation(url: str, *texts: str | None, only: Iterable[str] | None = None) -> str | None:
    """The regulation of the first text that names one (a promotion folder in the URL or a text counts first), or None.
    `only` limits the kinds that may match."""
    if "promotion" in _fold(url) or any("promotion" in _fold(t) for t in texts):
        return "promotion"
    allowed = tuple(only) if only is not None else None
    for text in texts:
        folded = _fold(text)
        for name, pattern in REGULATIONS:
            if (allowed is None or name in allowed) and pattern.search(folded):
                return name
    return None


def _residence_or_info(url: str) -> str:
    return "residence" if any(marker in url for marker in RESIDENCE) else "info"


def page_entries(page: Extraction, roles: Roles) -> list[PageEntry]:
    """The documents of a page. On a programme's hub page a section naming a PO year stays a document of its own; the rest
    is one document (a single remaining section keeps its id and heading)."""
    programme = _programme(page.url, roles)
    rest: list[PageDocument] = []
    own: list[tuple[PageDocument, int]] = []
    for doc in page.documents:
        year = _section_year(doc.section_id, doc.heading) if programme else None
        if year:
            own.append((doc, year))
        else:
            rest.append(doc)
    joined = len(rest) > 1
    if joined:
        rest = [PageDocument(None, page.title or None, "\n\n".join(d.markdown for d in rest))]
    entries = []
    for doc, year in [*((d, None) for d in rest), *own]:
        regulation = "po" if year else _regulation(page.url) or _residence_or_info(page.url)
        degree, prog, po_version = _scope(regulation, programme, _version(programme, roles, str(year)), roles)
        entries.append(
            PageEntry(
                doc.section_id,
                doc.heading,
                doc.markdown,
                Metadata(
                    regulation,
                    degree,
                    prog,
                    po_version,
                    any(ARCHIV.match(t or "") for t in (doc.heading, page.title)),
                    None if joined and not year else _valid_from(doc.markdown),
                    _lang(page.url),
                    _twin_key(page.url, doc.section_id),
                    _official_url(programme, po_version, roles),
                ),
            )
        )
    return entries


def pdf_metadata(link: Link, hub: Extraction, roles: Roles, twins: Mapping[str, str] | None = None) -> Metadata:
    """The metadata of the PDF `link` found on the page `hub` (the page's section is looked up by the link's section id)."""
    section = next((d for d in hub.documents if link.section_id and d.section_id == link.section_id), None)
    heading = section.heading if section else None
    name = unquote(urlsplit(link.url).path.rsplit("/", 1)[-1])
    stem = name.rsplit(".", 1)[0]
    regulation = (
        _regulation(link.url, link.text, link.title, stem)
        or ("po" if _section_year(link.section_id, heading) else None)
        or _regulation("", link.section_id, heading, only=HEADING_REGULATIONS)
        or _residence_or_info(link.url)
    )
    programme = _programme(hub.url, roles) or _programme(link.url, roles)
    version = _version(programme, roles, link.section_id, heading, link.text, link.title, DATE_PREFIX.sub("", stem))
    degree, prog, po_version = _scope(regulation, programme, version, roles)
    return Metadata(
        regulation,
        degree,
        prog,
        po_version,
        any(ARCHIV.match(t or "") for t in (link.text, link.title, heading)),
        _valid_from(link.title, link.text, name=name),
        _lang(link.url, stem),
        (twins or {}).get(link.url),
        _official_url(programme, po_version, roles),
        _unconfigured_year(regulation, programme, roles, link.text, link.title, DATE_PREFIX.sub("", stem)),
    )


def twin_keys(pairs: Iterable[PdfTwin]) -> dict[str, str]:
    """Both URLs of every configured pair mapped to the German one."""
    return {url: pair.de for pair in pairs for url in (pair.de, pair.en)}


def select_documents(candidates: Iterable[Candidate], roles: Roles) -> Selection:
    """Fail closed: a programme regulation lacking its programme or PO version is listed, not kept. A hub's copy of a
    university-wide regulation (a PDF in a programme's folder) is skipped when an original is in the set: the same known
    hash, or the same regulation and language with the same date in force. A missing hash or date never matches."""
    kept, listed = [], []
    for c in candidates:
        (listed if c.metadata.missing else kept).append(c)

    def hub_copy(c: Candidate) -> bool:
        return c.metadata.regulation in HUB_COPIED and any(
            c.url.startswith(p.pdf_prefix) for p in roles.programmes.values()
        )

    originals = [c for c in kept if c.metadata.regulation in HUB_COPIED and not hub_copy(c)]
    copies = [
        c
        for c in kept
        if hub_copy(c)
        and any(
            o.metadata.regulation == c.metadata.regulation
            and o.metadata.lang == c.metadata.lang
            and (
                bool(c.sha256 and o.sha256 == c.sha256)
                or (o.metadata.valid_from and o.metadata.valid_from == c.metadata.valid_from)
            )
            for o in originals
        )
    ]
    return Selection(
        tuple(c for c in kept if c not in copies),
        tuple((c.url, c.metadata.missing) for c in listed),
        tuple(c.url for c in copies),
    )
