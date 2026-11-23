"""The profile's sections: what decides the index, then the assistant's sections and keys. Every model forbids unknown
keys, except a source entry, whose connector-specific keys the connector validates."""

import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_core.settings import ProfileModel

Audience = Literal["public"]  # the index table allows no other


class Collection(ProfileModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    audiences: list[Audience] = Field(min_length=1)


class Embedding(ProfileModel):
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dimensions: int = Field(gt=0)
    query_prefix: str
    passage_prefix: str


class Rerank(ProfileModel):
    model: str
    revision: str
    threshold: float = Field(ge=0.0, le=1.0)  # 0.50 with parent_text, 0.15 without
    parent_text: bool
    min_docs: int = Field(ge=0)
    # An out-of-scope question whose best passage (top-1 cross_encoder_score) reaches this is answered from the documents;
    # null = off, the question is declined without a search
    scope_fallback_min: float | None = Field(default=None, ge=0.0, le=1.0)


class Switches(ProfileModel):
    """The retrieval stages that can be turned off one by one (an ablation); all on by default."""

    filters: bool = True  # the hard filter of a question (degree, programme, PO version, regulation, date); off = none
    acronyms: bool = (
        True  # the keyword search widens a named acronym to its group (acronyms.yaml); off = the words as asked
    )
    fusion_cut: bool = True  # each query's legs are fused and the card's pre_rerank_k hits go to the reranker, scored
    # against its rerank_queries first queries; off = every hit of every leg, scored against every query
    # the parent texts go to the reranker whole (the model server's length setting decides); off = each cut to 512 tokens
    # (512 * the card's chars_per_token characters)
    whole_parents: bool = True
    # after the reranker, a section a kept parent names ("§ 22 APB") is fetched, scored against the question and added (at
    # most 3, above the threshold); off = only the sections the search found
    reference_hop: bool = True
    # a question the analysis marks as needing more documents gets one extra search with follow-up queries, on a card that
    # gates it on and a profile that ships the follow-up prompt; off = the first search only
    iterative_round: bool = True


class Retrieval(ProfileModel):
    embedding: Embedding
    fts_language: str
    semantic_k: int = Field(gt=0)
    keyword_k: int = Field(gt=0)
    fts_extra_stopwords: tuple[str, ...]
    semantic_max_distance: float | None
    # w of fused = w * semantic + (1 - w) * lexical, each leg min-max normalised per query
    fusion_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank: Rerank
    switches: Switches = Switches()


class Chunking(ProfileModel):
    mode: Literal["recursive", "structure"]
    parent_chars: int = Field(gt=0)
    parent_overlap: int = Field(ge=0)
    child_chars: int = Field(gt=0)
    child_overlap: int = Field(ge=0)
    children: Literal["recursive", "sentences"]
    header: Literal["title", "tier1"]


class CollectionIngest(ProfileModel):
    min_build_ratio: float = Field(ge=0, le=1)


class Extraction(ProfileModel):
    processing: Literal["fast", "full"]
    max_ocr_pages: int = Field(ge=0)
    ocr_pages_per_run: int = Field(ge=0)
    clean_markup: bool
    min_text_length: int = Field(ge=0)


class ScopeVocabulary(ProfileModel):
    """ingest.scope: the words that make a document's own scope statement a limit for the source header.
    Stems are lowercase; the matcher folds the text."""

    groups: dict[
        str, tuple[str, ...]
    ]  # label -> stems naming a person group (the addressee, a statement, the page-1 heading)
    areas: tuple[str, ...]  # stems naming an employer or area: in the addressee or a statement only, never from a title
    compilations: tuple[str, ...]  # title stems of documents that bundle acts: no statement is read in them

    @model_validator(mode="after")
    def _lowercase_stems(self) -> "ScopeVocabulary":
        stems = [s for group in self.groups.values() for s in group] + list(self.areas) + list(self.compilations)
        if any(not group for group in self.groups.values()) or any(
            not s or s != s.lower() or s != s.strip() for s in stems
        ):
            raise ValueError("ingest.scope: every group has stems, and every stem is lowercase, trimmed and non-empty")
        return self


class Source(BaseModel):
    """One entry of ingest.sources: the generic keys; the connector's own keys stay in model_extra for the connector."""

    model_config = ConfigDict(extra="allow", frozen=True)
    connector: str
    audience: Audience
    min_documents: int = Field(default=0, ge=0)
    min_doc_ratio: float | None = Field(default=None, ge=0, le=1)


class Ingest(ProfileModel):
    chunking: Chunking
    collections: dict[str, CollectionIngest]
    extraction: Extraction
    scope: ScopeVocabulary
    sources: dict[str, Source]  # in build order


# --- the assistant's sections ----------------------------------------------------------------------------------------
# Every key is required (an answer-affecting value has no code default); a nullable key is written as null.


class Identity(ProfileModel):
    bot_name: str
    language: str
    locale: str
    date_format: str  # strftime format; %A is the weekday's name from day_names (rag_core.domain.dates)
    day_names: tuple[str, str, str, str, str, str, str]  # Monday first

    @field_validator("day_names")
    @classmethod
    def _plain_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        if any("%" in name for name in names):
            raise ValueError("a day name may not contain %")
        return names


class Ranking(ProfileModel):
    recency_types: list[str] = []  # document types whose newer sources win; none by default
    recency_bonus_per_year: float = Field(ge=0.0)
    recency_max_bonus: float = Field(ge=0.0)
    in_range_bonus: float
    in_range_by_year: bool
    question_years: bool
    no_recency_in_range: bool
    relevance_floor: float  # a reranked score below it abstains the answer; the min_docs guarantee keeps none below it
    source_min_score: float
    max_per_document: int = Field(ge=0)  # 0 = no cap

    @model_validator(mode="after")
    def _coupled(self) -> "Ranking":
        if self.recency_max_bonus < self.recency_bonus_per_year:
            raise ValueError("recency_max_bonus must be at least recency_bonus_per_year")
        return self


class History(ProfileModel):
    max_age_hours: float = Field(ge=0.0)  # 0 = no limit
    position: Literal["after_system", "first"]  # "first" puts the turns before the system message


class Guard(ProfileModel):
    max_question_chars: int = Field(gt=0)  # = the page's QUESTION_LIMIT
    safety_margin: int = Field(ge=0)
    source_window_chars: int = Field(ge=0)


class Generation(ProfileModel):
    """The analysis call's values; the answer's sampling, window and answer tokens are the model card's."""

    analysis_temperature: float
    analysis_max_tokens: int = Field(gt=0)


class Formats(ProfileModel):
    sources_event: Literal["before", "seen", "cited"]


class StatusGroup(ProfileModel):
    """One status group (a row of roles.yaml's group table); the words are lowercase."""

    label: str  # the group's name in the answer texts
    label_en: str | None = None  # its name in English answer texts; null = the label
    side: str  # the groups of one side are contrasted with those of the other sides, never a subgroup with its parent
    part_of: str | None = None  # the parent group of a subgroup (a key of the table)
    ask: tuple[str, ...] = ()  # question words: at a word start or the end of a compound ("Verwaltungsbeamtin")
    ask_self: tuple[str, ...] = ()  # question words that count only after "als", "ich bin", ... (ambiguous words)


def _check_part_of(groups: dict[str, StatusGroup]) -> None:
    for key, group in groups.items():
        parent = groups.get(group.part_of) if group.part_of else None
        if group.part_of and (parent is None or parent.side != group.side or parent.part_of):
            raise ValueError(f"status group {key}: part_of must name a top group of the same side")


class StatusGroups(ProfileModel):
    """Which lines about status groups and PO versions the answer prompt gets; the group table is roles.yaml's `groups`, the
    version table of an answer is the model card's `gated.version_table`. The questions' hard filter is the retrieval's."""

    compare: bool = False  # a question comparing groups or PO versions gets a prompt line asking for each one's rule
    # a question whose best sources are for different groups, programmes or PO versions gets a prompt line to tell them apart
    reminder: Literal["off", "differ"] = "off"
    # a German answer cited only from one side's sources gets a closing note on whom the rule is for
    note: Literal["off", "classic"] = "off"
    # only the best-ranked context sources count for the reminder: a far-down source of the other side does not
    # make a process question group-specific
    reminder_sources: int = 5

    @field_validator("reminder", "note", mode="before")
    @classmethod
    def _unquoted_off(cls, value: object) -> object:
        return "off" if value is False else value  # YAML reads a bare off as false


class Profile(ProfileModel):
    collections: list[Collection] = Field(min_length=1)
    retrieval: Retrieval
    ingest: Ingest
    identity: Identity
    ranking: Ranking
    history: History
    guard: Guard
    generation: Generation
    formats: Formats
    status_groups: StatusGroups = StatusGroups()


# --- the files beside profile.yaml: sources.yaml, roles.yaml and acronyms.yaml ----------------------------------------
# Each is a model of its own, read with load_profile(); none is a section of Profile.

# www.darmstadt.de (its robots.txt shuts out AI agents) and h-da.de: no crawl host, whatever else is listed
NEVER_CRAWLED = ("darmstadt.de", "www.darmstadt.de", "h-da.de")
TERM = re.compile(r"(WiSe) (\d{4})/(\d{2})|(SoSe) (\d{4})")
PROGRAMME_ID = re.compile(r"[a-z][a-z0-9]*(-[a-z0-9]+)*")


def _term_order(term: str) -> tuple[int, int]:
    """(year, 0 for a summer term, 1 for a winter term) of "SoSe 2023" or "WiSe 2023/24"; ValueError for any other text."""
    match = TERM.fullmatch(term)
    if not match:
        raise ValueError(f"{term!r}: a term is written 'WiSe 2023/24' or 'SoSe 2024'")
    if match.group(1):
        year = int(match.group(2))
        if int(match.group(3)) != (year + 1) % 100:
            raise ValueError(f"{term!r}: a winter term spans two consecutive years")
        return year, 1
    return int(match.group(5)), 0


class Selectors(ProfileModel):
    """Where a host's content is on its pages: a small CSS subset (tag, .class, #id, descendant, child, comma)."""

    main: str
    section: str  # an outermost match becomes one document of its own
    drop: str | None  # the chrome removed from the main content; null = the crawler's default list


class FullBuild(ProfileModel):
    """What a full build uses on a core host instead of the host's own sitemap, prefixes, exclusions and depth."""

    sitemap: list[str]
    prefixes: list[str]
    exclude: list[str]
    depth: int = Field(ge=0, le=3)


class CrawlHost(ProfileModel):
    """One host of the crawl. A core host is built by every build; a full host only by a full build. A host that both
    sets use is a core host with a `full` block. Seeds are listed pages; a host without prefixes lists its seeds only."""

    set: Literal["core", "full"]
    seeds: list[str]
    sitemap: list[str]
    prefixes: list[str]  # URL path prefixes a page must be under
    exclude: list[str]  # URL path prefixes left out, whatever else matches
    depth: int = Field(ge=0, le=3)  # links followed from the seeds, inside the prefixes
    twins: bool  # the English page of each ".de.jsp" page is fetched too
    selectors: Selectors
    full: FullBuild | None = None


class PdfTwin(ProfileModel):
    """A German PDF and its English counterpart: both get the German URL as their twin key."""

    de: str
    en: str


class Law(ProfileModel):
    """A statute taken from the official XML of its publisher instead of crawled pages: the norms from `start` to `end`
    (both written as the XML's `enbez`, e.g. "§ 16b"; null = from the first or to the last norm) become one document."""

    title: str
    kind: Literal["Gesetz", "Verordnung"]
    xml_url: str  # the zip with the law's XML, on a listed host
    page_url: str  # the page citations link to, on a listed host
    start: str | None
    end: str | None


class Sources(ProfileModel):
    """sources.yaml: the crawl's host allow-list (every other host is refused by the crawler), the statutes read from XML
    and the PDF twin pairs."""

    hosts: dict[str, CrawlHost] = Field(min_length=1)
    laws: list[Law] = []  # none by default
    pdf_twins: list[PdfTwin] = []  # none by default: PDFs have no twin key unless a pair is listed

    @model_validator(mode="after")
    def _hosts_are_wellformed(self) -> "Sources":
        for host, entry in self.hosts.items():
            if host != host.lower() or not re.fullmatch(r"[a-z0-9.-]+", host):
                raise ValueError(f"{host}: a host is written in lowercase without scheme or path")
            if host in NEVER_CRAWLED or host.endswith(".h-da.de"):
                raise ValueError(f"{host}: this host is never crawled")
            urls = [*entry.seeds, *entry.sitemap, *(entry.full.sitemap if entry.full else [])]
            if any(not url.startswith(f"https://{host}/") for url in urls):
                raise ValueError(f"{host}: seeds and sitemaps are https URLs on the host")
            paths = [
                *entry.prefixes,
                *entry.exclude,
                *((*entry.full.prefixes, *entry.full.exclude) if entry.full else ()),
            ]
            if any(not path.startswith("/") for path in paths):
                raise ValueError(f"{host}: prefixes and exclusions are URL paths starting with /")
            if entry.full and entry.set != "core":
                raise ValueError(f"{host}: only a core host has a full block")
            if (entry.depth and not entry.prefixes) or (entry.full and entry.full.depth and not entry.full.prefixes):
                raise ValueError(f"{host}: links are followed inside prefixes, so a depth needs prefixes")
        for law in self.laws:
            if any(
                urlsplit(url).scheme != "https" or urlsplit(url).hostname not in self.hosts
                for url in (law.xml_url, law.page_url)
            ):
                raise ValueError(f"{law.title}: a law's URLs are https URLs on listed hosts")
        for pair in self.pdf_twins:
            if any(
                urlsplit(url).scheme != "https" or urlsplit(url).hostname not in self.hosts
                for url in (pair.de, pair.en)
            ):
                raise ValueError(f"{pair.de}: a PDF twin pair is https URLs on listed hosts")
        return self


class PoVersion(ProfileModel):
    """One version of a programme's Prüfungsordnung; the terms are the first and last start term it governs (null =
    not stated). A term is written 'WiSe 2023/24' or 'SoSe 2024'."""

    year: int = Field(ge=1990, le=2099)
    start_from: str | None
    start_to: str | None
    official_url: str | None  # the official publication of the version

    @model_validator(mode="after")
    def _terms_in_order(self) -> "PoVersion":
        terms = [_term_order(t) for t in (self.start_from, self.start_to) if t is not None]
        if len(terms) == 2 and terms[0] > terms[1]:
            raise ValueError(f"PO {self.year}: start_from is after start_to")
        return self


class Wording(ProfileModel):
    text_de: str = Field(min_length=1)
    text_en: str = Field(min_length=1)  # an information translation: the German text binds


class Programme(ProfileModel):
    label_de: str
    label_en: str
    degree: Literal["bachelor", "master"]
    url_prefix: str  # the URL of the programme's hub page; its page sections are under it
    pdf_prefix: str  # the URL of the folder holding the programme's PDFs
    po_versions: list[PoVersion] = Field(min_length=1)
    po_change: Wording | None  # how a student moves to a newer version; null = not stated for the programme

    @field_validator("po_versions")
    @classmethod
    def _one_row_per_year(cls, versions: list[PoVersion]) -> list[PoVersion]:
        years = [v.year for v in versions]
        if len(set(years)) != len(years):
            raise ValueError("a PO version appears once per programme")
        return versions


class Office(ProfileModel):
    label_de: str
    label_en: str
    url: str | None  # null = no page known


class Roles(ProfileModel):
    """roles.yaml: the programmes with their PO versions, the status-group table and the referral offices by topic."""

    po_rule: Wording  # which version governs a student when the question names none
    programmes: dict[str, Programme] = Field(min_length=1)
    groups: dict[str, StatusGroup]
    offices: dict[str, Office]

    @model_validator(mode="after")
    def _consistent(self) -> "Roles":
        if bad := [p for p in self.programmes if not PROGRAMME_ID.fullmatch(p)]:
            raise ValueError(f"programme {bad[0]}: an id is lowercase words joined by hyphens")
        for programme in self.programmes.values():
            if not (programme.url_prefix.startswith("https://") and programme.pdf_prefix.startswith("https://")):
                raise ValueError(f"{programme.label_de}: url_prefix and pdf_prefix are https URLs")
        _check_part_of(self.groups)
        return self


class AcronymGroup(ProfileModel):
    """Terms that stand for one thing: a matched term widens to the whole group."""

    terms: tuple[str, ...] = Field(min_length=2)

    @field_validator("terms")
    @classmethod
    def _plain_terms(cls, terms: tuple[str, ...]) -> tuple[str, ...]:
        if any(not t or t != t.strip() for t in terms):
            raise ValueError("a term is non-empty and trimmed")
        return terms


class Acronyms(ProfileModel):
    """acronyms.yaml: groups of terms, an acronym with its long form and variants; no term is in two groups."""

    groups: tuple[AcronymGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _terms_are_unique(self) -> "Acronyms":
        seen: set[str] = set()
        for group in self.groups:
            for term in group.terms:
                if term.casefold() in seen:
                    raise ValueError(f"{term!r} is in two groups, or twice in one")
                seen.add(term.casefold())
        return self
