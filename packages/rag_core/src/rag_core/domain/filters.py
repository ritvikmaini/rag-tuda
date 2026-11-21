"""The hard filter of a question (the values the retrieval narrows by) and where its values come from.

A value comes from the first source that gives one, in this order: explicit text read by deterministic readers ("PO 2015",
"im Master", "laut Modulhandbuch"), the page chips, the term-to-PO parser (cohort.py), the analysis enums, the previous turn's
filter. A source's value that contradicts one already taken (a programme of another degree, a PO year the programme lacks) is
dropped. Text values hold for the question that carries them; the chips are sent with every question and so stay. A field a
chip supplied is never carried to a turn whose chips no longer set it (the user cleared the chip); a value from text or the
term parser is. Regulation and date are carried from the previous turn only when the new question names no value at all.

A degree the question names only inside a compound ("Mastermodule") that contradicts the chips' degree picks no side: the
question is widened (no degree, programme or chip PO year), and the chips stay for the next question.

Roles are plain data: the dump of roles.yaml (`programmes` with `degree` and `po_versions`, `groups`)."""

import datetime as dt
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from rag_core.domain.cohort import enrolment_terms, po_for_term
from rag_core.domain.dates import parse_term, term_start, terms_written, written_dates

REGULATIONS = ("po", "mhb", "spp", "apb", "eso", "promotion", "residence", "info", "form")  # what the index holds


@dataclass(frozen=True)
class Filter:
    """What the retrieval narrows by; None = the field is not filtered. NULL rows (documents that apply to all) pass every
    filtered field. `date` keeps rows with valid_from <= date. Rows marked archived are left out unless include_archived."""

    degree: tuple[str, ...] | None = None
    programme: tuple[str, ...] | None = None
    po_version: tuple[int, ...] | None = None
    regulation: tuple[str, ...] | None = None
    date: dt.date | None = None
    include_archived: bool = False
    # where po_version came from: "text", "chip", "term" (derived from a start term: the PO "in der Regel" in force), "analysis",
    # "carried" (the previous turn's; a term-derived one stays "term"); None without a PO
    po_source: str | None = None
    # the fields (degree, programme, po_version) whose value a chip supplied: not carried once the chip is cleared
    from_chips: frozenset[str] = frozenset()

    def as_json(self) -> dict[str, Any]:
        """The filter as plain JSON (lists, the date as ISO text): what the turn store keeps and the status frame names."""
        return {
            "degree": None if self.degree is None else list(self.degree),
            "programme": None if self.programme is None else list(self.programme),
            "po_version": None if self.po_version is None else list(self.po_version),
            "regulation": None if self.regulation is None else list(self.regulation),
            "date": None if self.date is None else self.date.isoformat(),
            "include_archived": self.include_archived,
            "po_source": self.po_source,
            "from_chips": sorted(self.from_chips),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Filter":
        """The filter `as_json` wrote; a field the data lacks is not filtered. Raises ValueError or TypeError on a value of
        the wrong kind."""

        def values(name: str) -> tuple[Any, ...] | None:
            value = data.get(name)
            if value is not None and not isinstance(value, list):
                raise TypeError(f"{name}: a list or null, not {type(value).__name__}")
            return None if value is None else tuple(cast("list[Any]", value))

        chipped = data.get("from_chips")  # absent in a filter stored before the field existed
        if chipped is not None and not isinstance(chipped, list):
            raise TypeError(f"from_chips: a list or null, not {type(chipped).__name__}")
        day = data.get("date")
        return cls(
            degree=values("degree"),
            programme=values("programme"),
            po_version=values("po_version"),
            regulation=values("regulation"),
            date=None if day is None else dt.date.fromisoformat(day),
            include_archived=data.get("include_archived") is True,
            po_source=data.get("po_source"),
            from_chips=frozenset(cast("list[str]", chipped or ())),
        )


@dataclass(frozen=True)
class Chips:
    """The page's selectors, one value each; None = not chosen."""

    degree: str | None = None
    programme: str | None = None
    po: int | None = None


# explicit text ---------------------------------------------------------------------------------------------------------

_YEAR = r"(?<!\d)(?:19|20)\d{2}(?!\d)"
# "PO 2015", "PO (2015)", "Prüfungsordnung von 2015", "Prüfungsordnung aus dem Jahr 2015", "PO 2015 und PO 2023",
# "PO 2015 vs 2023", "examination regulations 2023"
PO_YEARS = re.compile(
    r"\b(?:PO|Prüfungsordnung(?:en)?|(?:examination|exam) regulations?)[\s-]*(?:(?:von|vom|aus|of|from)\s+)?"
    r"(?:(?:dem\s+)?Jahre?\s+|the\s+year\s+)?\(?\s*("
    + _YEAR
    + r"(?:\s*(?:,|/|&|und|oder|bzw\.?|vs\.?|versus|and|or)\s*(?:PO\s*)?"
    + _YEAR
    + r")*)",
    re.IGNORECASE,
)
PO_SHORT_YEAR = re.compile(r"\bPO[\s-]?'?(\d{2})(?!\d)", re.IGNORECASE)  # "PO15", "PO 15": the year 2015
DEGREE_WORDS = {
    "bachelor": re.compile(r"\bbachelor\w*|\bb\.?\s?sc\b|\bbsc\b", re.IGNORECASE),
    "master": re.compile(r"\bmaster\w*|\bm\.?\s?sc\b|\bmsc\b", re.IGNORECASE),
    "promotion": re.compile(
        r"\bpromotion\w*|\bpromovier\w*|\bpromovend\w*|\bdoktorand\w*|\bdoktorarbeit\w*|\bdoctoral|\bdoctorate|\bph\.?\s?d\b",
        re.IGNORECASE,
    ),
}
# a degree word that continues into a longer word ("Mastermodule", "Bachelorarbeit"), apart from the study forms
# ("Masterstudent", "Bachelorstudierende") and the plural: such a word names a topic, not the asker's degree
DEGREE_COMPOUND = re.compile(r"\b(?:bachelor|master)(?!s?\b)(?!stud)\w+", re.IGNORECASE)
# a document type counts when a preposition names it: "im Modulhandbuch", "laut APB" (a bare "PO-Wechsel" or "Modulhandbuch"
# names a topic, not where the answer stands)
_CUE = r"\b(?:im|in\s+(?:der|dem)|laut|gemäß|gemaess|lt\.|in\s+the|according\s+to)\s+(?:der\s+|dem\s+|den\s+)?"
REGULATION_WORDS = {
    "mhb": r"modulhandbuch|mhb",
    "spp": r"studien-?\s?und\s?prüfungsplan|studienplan|spp",
    "apb": r"apb|allgemeinen?\s+prüfungsbestimmungen",
    "eso": r"eso|einschreib\w*ordnung",
    "promotion": r"promotionsordnung",
    "po": r"po|prüfungsordnung|studiengangsordnung|examination regulations?",
}
REGULATION_TEXT = {
    name: re.compile(_CUE + "(?:" + words + r")(?![\w-])", re.IGNORECASE) for name, words in REGULATION_WORDS.items()
}
# "die alte PO", "ältere Fassung", "previous version", "Archiv", "history": the question asks for an old version
HISTORY = re.compile(
    r"\b(?:alt|älter|früher|bisherig|ehemalig|vorherig|vorig)\w*\s+(?:PO|Prüfungsordnung\w*|Version\w*|Fassung\w*|Regelung\w*"
    r"|Ordnung\w*|APB|Bestimmung\w*)|"
    r"\b(?:old|older|previous|former|earlier|archived)\s+(?:PO|versions?|regulations?|examination regulations?|rules?)|"
    r"\barchiv\w*|\bhistori\w*|\bhistory\b",
    re.IGNORECASE,
)


def _fold(text: str) -> str:
    """Lower case, "B. Sc." written "bsc", hyphens as spaces, one space between words."""
    return " ".join(re.sub(r"\b([bm])\.?\s?sc\b\.?", r"\1sc", text.casefold()).replace("-", " ").split())


def _programme_aliases(programme: Mapping[str, Any]) -> set[str]:
    """The spellings of a programme's label in a question: both labels, and "Bachelor"/"Master" for "B.Sc."/"M.Sc."."""
    labels = {_fold(programme["label_de"]), _fold(programme["label_en"])}
    return labels | {re.sub(r"^bsc\b", "bachelor", re.sub(r"^msc\b", "master", label)) for label in labels}


def degrees(roles: Mapping[str, Any]) -> set[str]:
    """The degrees a chip or a value may name: those of the programmes, and promotion when roles has that group."""
    found = {programme["degree"] for programme in roles.get("programmes", {}).values()}
    return found | ({"promotion"} if "promotion" in roles.get("groups", {}) else set())


def _years(roles: Mapping[str, Any], programmes: Iterable[str] | None = None) -> set[int]:
    """The PO years of the programmes (all of them without the argument)."""
    chosen = roles.get("programmes", {})
    return {v["year"] for name in (chosen if programmes is None else programmes) for v in chosen[name]["po_versions"]}


def _read(text: str, roles: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    """What the question's own words name: degree, programme, po_version, regulation (the fields it names)."""
    found: dict[str, tuple[Any, ...]] = {}
    if names := tuple(name for name, words in DEGREE_WORDS.items() if words.search(text)):
        found["degree"] = names
    folded = _fold(text)
    if names := tuple(
        name
        for name, programme in roles.get("programmes", {}).items()
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", folded) for alias in _programme_aliases(programme))
    ):
        found["programme"] = names
    years = [int(y) for m in PO_YEARS.finditer(text) for y in re.findall(_YEAR, m[1])]
    if years := years + [2000 + int(m[1]) for m in PO_SHORT_YEAR.finditer(text)]:
        found["po_version"] = tuple(years)
    if names := tuple(name for name, pattern in REGULATION_TEXT.items() if pattern.search(text)):
        found["regulation"] = names
    return found


# the other sources -----------------------------------------------------------------------------------------------------


def _listed(analysis: object, name: str) -> tuple[Any, ...] | None:
    """The analysis' array field `name`; None when it has no such field or it is null."""
    value: Any = getattr(analysis, name, None)
    return tuple(cast("Iterable[Any]", value)) if isinstance(value, (list, tuple)) else None


def _keep(values: Iterable[Any] | None, allowed: Iterable[Any]) -> tuple[Any, ...] | None:
    """The values among the allowed ones, once each, in order; None when none is left."""
    allowed = set(allowed)
    return tuple(dict.fromkeys(v for v in values or () if v in allowed)) or None


def _fits(field: str, value: tuple[Any, ...], current: Mapping[str, Any], roles: Mapping[str, Any]) -> bool:
    """False when value for field contradicts a taken degree, programme or PO year."""
    programmes = roles.get("programmes", {})
    if field == "degree":
        return all(programmes[p]["degree"] in value for p in current["programme"] or ())
    if field == "programme":
        return all(current["degree"] is None or programmes[p]["degree"] in current["degree"] for p in value) and (
            current["po_version"] is None or set(current["po_version"]) <= _years(roles, value)
        )
    if field == "po_version":  # a year the taken programmes, else the programmes of the taken degree, have a PO for
        chosen = current["programme"] or [
            p for p, v in programmes.items() if current["degree"] is None or v["degree"] in current["degree"]
        ]
        return set(value) <= _years(roles, chosen)
    return True


def _int(value: object) -> int | None:
    """value as an int when it is one or a string of digits."""
    return int(value) if isinstance(value, int) or (isinstance(value, str) and value.isdigit()) else None


def _cohort_term(text: str, analysis: object) -> str | None:
    """The start term the analysis gives, else the first term the text gives as the start of the studies."""
    term = getattr(analysis, "enrolment_term", None)
    if isinstance(term, str) and parse_term(term):
        return term
    return next((text[start:end] for start, end in enrolment_terms(text)), None)


def _text_date(text: str, today: dt.date) -> dt.date | None:
    """The first date the text writes out, else the start of the first term it names that is not the start of the studies."""
    if dates := written_dates(text, today):
        return dates[0][2]
    skipped = enrolment_terms(text)
    return next((term_start(t) for start, end, t in terms_written(text) if (start, end) not in skipped), None)


def resolve_filter(
    text: str,
    chips: Chips | None,
    analysis: object | None,
    previous: Filter | None,
    roles: Mapping[str, Any],
    *,
    today: dt.date | None = None,
) -> Filter:
    """The filter for the question `text` (the user's own words, so the readers run when the analysis fell back, too).
    `analysis` may lack the enum fields; `previous` is the last turn's applied filter."""
    chips, previous = chips or Chips(), previous or Filter()
    allowed = {
        "degree": degrees(roles),
        "programme": set(roles.get("programmes", {})),
        "po_version": _years(roles),
        "regulation": set(REGULATIONS),
    }
    current: dict[str, Any] = dict.fromkeys(allowed)
    own = _read(text, roles)
    chip_degree = chips.degree or (roles.get("programmes", {}).get(chips.programme or "", {}).get("degree"))
    topical = {n for n in ("bachelor", "master") if DEGREE_WORDS[n].search(text)} - {
        n for n in ("bachelor", "master") if DEGREE_WORDS[n].search(DEGREE_COMPOUND.sub(" ", text))
    }
    widen = bool(chip_degree and topical and set(own.get("degree", ())) <= topical and chip_degree not in topical)
    if widen:
        chips, own = Chips(), {k: v for k, v in own.items() if k != "degree"}
    po_source: str | None = None
    from_chips: set[str] = set()

    def take(layer: Mapping[str, Iterable[Any] | None], source: str) -> None:
        nonlocal po_source
        for field in allowed:  # degree, programme, po_version, regulation: the order the fits are checked in
            if source != "text" and field in own:
                continue  # the question named this field: a value of it that was rejected is not replaced by a weaker source
            value = _keep(layer.get(field), allowed[field])
            if current[field] is not None or not value or not _fits(field, value, current, roles):
                continue
            if field == "po_version":
                # a PO year belongs to the degree and programme its layer names with it: dropped when they were not taken
                named = (_keep(layer.get(f), allowed[f]) for f in ("degree", "programme"))
                if any(
                    v is not None and set(v) != set(current[f] or ()) for f, v in zip(("degree", "programme"), named)
                ):
                    continue
                po_source = source
            current[field] = value
            if source == "chip":
                from_chips.add(field)

    # the chips are sent with every question: a chip's value the chips no longer set is not carried
    carry = {
        field: None if field in previous.from_chips else getattr(previous, field)
        for field in ("degree", "programme", "po_version")
    }
    take(own, "text")
    take({"degree": [chips.degree], "programme": [chips.programme], "po_version": [chips.po]}, "chip")
    from_analysis = {f: _listed(analysis, f) for f in allowed}
    if term := _cohort_term(text, analysis):
        known = current["programme"] or _keep(from_analysis["programme"], allowed["programme"]) or carry["programme"]
        years = [po_for_term(p, term, roles) for p in known or ()]
        if years and None not in years:
            take({"po_version": years}, "term")
    take(from_analysis | {"po_version": [_int(v) for v in from_analysis["po_version"] or ()]}, "analysis")
    date = _text_date(text, today or dt.date.today())
    history = bool(HISTORY.search(text))
    named = date is not None or any(v is not None for v in current.values())
    carried = "term" if previous.po_source == "term" else "carried"
    take(carry, carried)
    if widen:  # a PO year stays only when the question wrote it
        current["degree"] = current["programme"] = None
        if "po_version" not in own:
            current["po_version"], po_source = None, None
    if not named:
        take({"regulation": previous.regulation}, carried)
        date = previous.date
    # a carried history request counts; a carried True that came from a PO year does not (the PO is carried or dropped itself)
    archived = (
        history or current["po_version"] is not None or (previous.include_archived and previous.po_version is None)
    )
    return Filter(
        current["degree"],
        current["programme"],
        current["po_version"],
        current["regulation"],
        date,
        archived,
        po_source,
        frozenset(from_chips),
    )


def check_chips(chips: Chips, roles: Mapping[str, Any]) -> str | None:
    """Why the chips are refused, or None when each value is one the roles know and the three agree."""
    programmes = roles.get("programmes", {})
    if chips.degree is not None and chips.degree not in degrees(roles):
        return f"degree {chips.degree!r}: one of {', '.join(sorted(degrees(roles))) or 'none configured'}"
    if chips.programme is not None and chips.programme not in programmes:
        return f"programme {chips.programme!r}: one of {', '.join(programmes) or 'none configured'}"
    if chips.degree and chips.programme and programmes[chips.programme]["degree"] != chips.degree:
        return f"programme {chips.programme!r} is not a {chips.degree} programme"
    pool = _years(roles, [chips.programme] if chips.programme else None)
    if chips.po is not None and chips.po not in pool:
        return f"po {chips.po}: one of {', '.join(map(str, sorted(pool))) or 'none configured'}"
    return None
