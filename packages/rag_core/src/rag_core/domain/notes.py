"""The lines code appends to a finished answer, from the metadata of the sources it cites: the PO versions it rests on
("Grundlage: PO 2023 und PO 2015 des B.Sc. Informatik"), the disclaimer on a residence source and the note on an English
information translation of a binding German text; an urgent reply that leaves out a phone number of the profile's contact line
gets the line. The wording is the profile's, in the answer's language; a text the profile
leaves empty drops its line. Standard library only."""

import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from rag_core.domain.retrieval import BINDING_GERMAN
from rag_core.domain.status_groups import join_labels, source_programmes, source_versions

Metadata = Mapping[str, Any] | None
NUMBER = re.compile(r"\d+(?: \d+)*")  # a phone number as the profile writes it: digit groups with single spaces


class Texts(Protocol):
    """The wording of the appended lines (the profile's `answer` messages)."""

    @property
    def po_line(self) -> str: ...  # {entries}: one entry per programme
    @property
    def po_line_term(self) -> str: ...  # the same for a PO that follows from a start term (a default: "in der Regel")
    @property
    def po_one(self) -> str: ...  # {version}: the entry for a programme's one version
    @property
    def po_programme(self) -> str: ...  # {entry}, {programme}: an entry bound to its programme's label
    @property
    def non_binding(self) -> str: ...
    @property
    def visa_disclaimer(self) -> str: ...
    @property
    def urgent_contacts(self) -> str: ...  # the line an urgent reply gets when it leaves out one of its numbers


def _years(programmes: Mapping[str, Mapping[str, Any]], name: str | None) -> set[int]:
    """The PO years the roles list for a programme; none for no programme or one the roles do not know."""
    found: Mapping[str, Any] = programmes[name] if name is not None and name in programmes else {}
    return {int(v["year"]) for v in found.get("po_versions") or ()}


def po_line(
    cited: Sequence[Metadata], programmes: Mapping[str, Mapping[str, Any]], texts: Texts, *, english: bool, term: bool
) -> str:
    """The versions of the cited sources, one entry per programme, versions newest first; "" when no cited source is bound to
    a version. A source that names several programmes counts, for each, only the versions the programme has (as far as
    `programmes`, the roles' table, lists them)."""
    text = texts.po_line_term if term else texts.po_line
    years: dict[str | None, set[int]] = {}
    for meta in cited:
        named = set(source_versions(meta))
        for name in source_programmes(meta) or (None,):
            valid = _years(programmes, name)
            if named:
                years.setdefault(name, set()).update(named & valid or named)
    if not years or not text:
        return ""
    entries: list[str] = []
    for name, found in years.items():
        ordered = sorted(found, reverse=True)
        entry = (
            texts.po_one.format(version=ordered[0])
            if len(ordered) == 1
            else join_labels([f"PO {y}" for y in ordered], "and" if english else "und")
        )
        if name and texts.po_programme:
            label = (programmes.get(name) or {}).get("label_en" if english else "label_de") or name
            entry = texts.po_programme.format(entry=entry, programme=label)
        entries.append(entry)
    return text.format(entries="; ".join(entries))


def _flat(text: str) -> str:
    """The text with every run of whitespace as one space."""
    return " ".join(text.split())


def appended_lines(
    cited: Sequence[Metadata],
    programmes: Mapping[str, Mapping[str, Any]],
    texts: Texts,
    *,
    english: bool,
    term: bool,
    answer: str = "",
) -> list[str]:
    """The lines for the cited sources' metadata, in this order: the PO versions; the note when a cited source is an English
    section of a binding regulation (an information translation); the disclaimer when one is a residence source. A line the
    answer already contains (whitespace aside) is not added again."""
    lines = [po_line(cited, programmes, texts, english=english, term=term)]
    if texts.non_binding and any(m and m.get("lang") == "en" and m.get("regulation") in BINDING_GERMAN for m in cited):
        lines.append(texts.non_binding)
    if texts.visa_disclaimer and any(m and m.get("regulation") == "residence" for m in cited):
        lines.append(texts.visa_disclaimer)
    have = _flat(answer)
    return [line for line in lines if line and _flat(line) not in have]


def urgent_line(answer: str, contacts: str) -> str:
    """The contact line (the profile's, in the answer's language) when the answer lacks any phone number the line names, else
    "". The numbers are the line's runs of digits and single spaces ("0800 111 0 111"); the answer holds one when its digits
    appear in order, as a whole number, with spaces or hyphens between the groups."""
    for number in NUMBER.findall(contacts):
        written = r"[\s-]*".join(number.split())
        if not re.search(rf"(?<!\d){written}(?!\d)", answer):
            return contacts
    return ""


def _line_pattern(template: str) -> re.Pattern[str] | None:
    """A pattern for a line made from a template: its {entries} read as any text; None for an empty template."""
    if not template:
        return None
    head, _, tail = template.partition("{entries}")
    return re.compile(re.escape(_flat(head)) + (".+" if "{entries}" in template else "") + re.escape(_flat(tail)))


def strip_lines(answer: str, texts: Sequence[Texts]) -> str:
    """The answer without the lines appended to it: the paragraphs at its end that read as one of the texts' lines (any
    language given). The model sees earlier answers this way, so it does not imitate the lines."""
    patterns = [
        p
        for t in texts
        for p in map(_line_pattern, (t.po_line, t.po_line_term, t.non_binding, t.visa_disclaimer, t.urgent_contacts))
        if p is not None
    ]
    paragraphs = answer.split("\n\n")
    while len(paragraphs) > 1 and any(p.fullmatch(_flat(paragraphs[-1])) for p in patterns):
        paragraphs.pop()
    return "\n\n".join(paragraphs)
