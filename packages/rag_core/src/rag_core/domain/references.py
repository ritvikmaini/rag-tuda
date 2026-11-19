"""The explicit references in a section's text that name a paragraph of a regulation: "§ 22 APB", "gemäß § 5 Abs. 2 der
Ausführungsbestimmungen", "§ 16b AufenthG", "§ 5 dieser Ordnung". Pure; stdlib only.

A reference reads "§ n" (an optional letter), any number of "Abs. 2" / "Satz 1" / "Nr. 3" qualifiers, an optional article and
a regulation word. The word gives the regulation of the index: APB, the Ausführungsbestimmungen, Prüfungsordnung and
Studienordnung (a programme's own rules are its PO), the Modulhandbuch, the Einschreibeordnung, and the residence laws
AufenthG and AufenthV. A programme regulation (po, mhb, spp) is the citing section's own programme and PO version; "dieser
Ordnung" is the citing section's own regulation, programme and version. A list ("§ 5 und § 6 APB") names every paragraph,
a year after a programme regulation ("§ 10 PO 2015") names its PO version, and a bare "§ 5" without a regulation, , one whose
regulation word starts the next line, or a line of nothing else (a heading, a table-of-contents entry), is not a reference."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

PROGRAMME_REGULATIONS = frozenset({"po", "mhb", "spp"})  # bound to a programme and a PO version

# regulation -> its spellings (acronyms case-sensitive, words not)
_SPELLINGS = {
    "apb": r"(?-i:APB)|allgemeinen?[ \t]+prüfungsbestimmungen",
    "po": r"(?-i:PO|AB)|ausführungsbestimmungen|prüfungsordnung|studienordnung",
    "mhb": r"(?-i:MHB)|modulhandbuch",
    "eso": r"(?-i:ESO)|einschreib(?:e|ungs)?ordnung",
    "residence": r"(?-i:AufenthG|AufenthV)",
    "own": r"dieser[ \t]+(?:prüfungs|studien|einschreibe)?ordnung",
}
# "§ 5", "§ 16b Abs. 2 Satz 1"; a list of them ("§ 5 und § 6", "§ 5, § 6") shares the regulation word that follows. The
# reference sits on one line: the paragraph, its word and the PO year are separated by spaces and tabs only.
_PARAGRAPH = r"§[ \t]*\d+[a-z]?(?:[ \t]+(?:Abs\.?|Absatz|Satz|S\.|Nr\.?|Nummer|Ziffer|lit\.?)[ \t]*\d+[a-z]?)*"
_REFERENCE = re.compile(
    rf"(?P<paragraphs>{_PARAGRAPH}(?:[ \t]*(?:,|und|oder|sowie)[ \t]*{_PARAGRAPH})*)"
    r"[ \t]+(?:(?:der|des|dem|zur|zum|von|im|in|nach)[ \t]+)?"
    r"(?:" + "|".join(f"(?P<{name}>{spelling})" for name, spelling in _SPELLINGS.items()) + r")\b"
    r"(?:[ \t]+(?P<year>(?:19|20)\d\d)\b)?",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"§[ \t]*(\d+[a-z]?)", re.IGNORECASE)


@dataclass(frozen=True)
class Reference:
    """A paragraph of a regulation: `paragraph` as "§ 22"; `programme` and `po_version` only for a programme regulation of a
    citing section that has them."""

    regulation: str
    paragraph: str
    programme: tuple[str, ...] | None
    po_version: tuple[int, ...] | None


def _values(value: Any, kind: Callable[[Any], Any]) -> tuple[Any, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    return tuple(kind(v) for v in cast("Sequence[Any]", value))


def _is_whole_line(text: str, m: re.Match[str]) -> bool:
    """The match is all there is on its line: a heading or a table-of-contents entry ("§ 2 Prüfungsordnung"), not a citation."""
    start = text.rfind("\n", 0, m.start()) + 1
    end = text.find("\n", m.end())
    return not text[start : m.start()].strip() and not text[m.end() : len(text) if end < 0 else end].strip()


def find_references(text: str, own: Mapping[str, Any]) -> list[Reference]:
    """The references in `text`, each once in reading order. `own` is the citing section's metadata (regulation, programme,
    po_version): it resolves "dieser Ordnung" and gives a programme regulation its programme and PO version."""
    found: dict[Reference, None] = {}
    text = text or ""
    for m in _REFERENCE.finditer(text):
        if _is_whole_line(text, m):
            continue
        name = next(n for n in _SPELLINGS if m.group(n))
        regulation = own.get("regulation") if name == "own" else name
        if not isinstance(regulation, str) or not regulation:
            continue
        bound = regulation in PROGRAMME_REGULATIONS
        programme = _values(own.get("programme"), str) if bound else None
        po_version = _values(own.get("po_version"), int) if bound else None
        if bound and name != "own" and m.group("year"):  # "§ 10 PO 2015" names its version
            po_version = (int(m.group("year")),)
        for number in _NUMBER.findall(m.group("paragraphs")):
            found[Reference(regulation, f"§ {number}", programme, po_version)] = None
    return list(found)
