"""The cohort parser: the Prüfungsordnung in force for a student's start term, from the PO versions and start terms of the
roles (plain data: the dump of roles.yaml). A term the roles state no boundary for gives None, never a guess."""

import re
from collections.abc import Mapping
from typing import Any

from rag_core.domain.dates import Term, parse_term, terms_written

# the words that make a term in a question the start of the asker's studies: within this many characters of the term
ENROLMENT_CUE = re.compile(
    r"\bseit\b|\bsince\b|eingeschrieben|immatrikuliert|einschreibung|studienbeginn|angefangen|begonnen|gestartet|"
    r"\benrolled\b|\benrolment\b|\bstarted\b|\bbegan\b|\bbegun\b",
    re.IGNORECASE,
)
CUE_WINDOW = 50


def po_for_term(programme: str, term: str, roles: Mapping[str, Any]) -> int | None:
    """The PO year in force for students of the programme who start in term ("WiSe 2023/24"): of the versions whose first start
    term is stated and not after the term and whose last start term is open or not before it, the one that starts latest. None
    for an unknown programme or term and when no version states a start for it."""
    start, found = parse_term(term), roles.get("programmes", {}).get(programme)
    if start is None or found is None:
        return None
    best: tuple[Term, int] | None = None
    for version in found["po_versions"]:
        first = parse_term(version["start_from"] or "")
        last = parse_term(version["start_to"] or "")
        if (
            first is not None
            and first <= start
            and (last is None or start <= last)
            and (best is None or first > best[0])
        ):
            best = (first, version["year"])
    return best[1] if best else None


def enrolment_terms(text: str) -> list[tuple[int, int]]:
    """(start, end) in text of each term the text gives as the start of the studies."""
    return [
        (start, end)
        for start, end, _ in terms_written(text)
        if ENROLMENT_CUE.search(text[max(0, start - CUE_WINDOW) : end + CUE_WINDOW])
    ]
