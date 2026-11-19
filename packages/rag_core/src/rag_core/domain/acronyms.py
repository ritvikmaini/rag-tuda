"""The acronym groups a question names. A group is terms that stand for one thing (an acronym, its long form, the students'
variants); the keyword search widens a named term to its whole group. Matching is by whole word, ignoring case; a term of five
or more letters also matches with a German ending ("Prüfungsordnungen", "des Rücktritts")."""

import re
from collections.abc import Sequence
from functools import cache

ENDING = r"(?:e|en|s|es|n)?"
MIN_INFLECTED = 5


@cache
def _pattern(term: str) -> re.Pattern[str]:
    body = re.escape(term).replace(r"\ ", r"\s+")
    ending = ENDING if len(term) >= MIN_INFLECTED and term[-1].isalpha() else ""
    return re.compile(rf"(?<!\w){body}{ending}(?!\w)", re.IGNORECASE)


def expand(text: str, groups: Sequence[Sequence[str]]) -> tuple[str, list[tuple[str, ...]]]:
    """(the text with every named term cut out, the groups named, each once, in order of appearance). Where terms overlap the
    longest wins ("PO-Wechsel" over "PO")."""
    found = [
        (m.start(), m.end(), i)
        for i, group in enumerate(groups)
        for term in group
        for m in _pattern(term).finditer(text)
    ]
    taken: list[tuple[int, int, int]] = []
    for span in sorted(found, key=lambda s: (s[0] - s[1], s[0])):  # longest first
        if all(span[1] <= other[0] or other[1] <= span[0] for other in taken):
            taken.append(span)
    taken.sort()
    rest: list[str] = []
    position = 0
    for start, end, _ in taken:
        rest.append(text[position:start])
        position = end
    rest.append(text[position:])
    named = list(dict.fromkeys(i for _, _, i in taken))
    return " ".join("".join(rest).split()) if taken else text, [tuple(groups[i]) for i in named]
