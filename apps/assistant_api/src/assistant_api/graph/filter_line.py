"""The line that closes an answer to a search that found nothing under a hard filter: it names what was applied ("Für den
M.Sc. Informatik, PO 2015, habe ich nichts gefunden."), because the search is never repeated without the filter. The wording
is the profile's (`answer.no_match` and the `filter_*` texts of messages.<language>.yaml), the programmes' labels come from
roles."""

from collections.abc import Mapping
from typing import Any

from rag_core.domain.filters import Filter

from assistant_api.settings import AnswerMessages


def filter_parts(applied: Filter, roles: Mapping[str, Any], texts: AnswerMessages, english: bool) -> list[str]:
    """What the filter names, in the order scope, PO: the programmes (else the degree) and the PO years. Regulation and date
    come from the question's own words and are not named."""
    parts: list[str] = []
    label = "label_en" if english else "label_de"
    programmes = roles.get("programmes", {})
    if applied.programme:
        labels = dict.fromkeys(programmes[name][label] for name in applied.programme if name in programmes)
        if labels and texts.filter_programmes:
            parts.append(texts.filter_programmes.format(labels=" / ".join(labels)))
    elif applied.degree:
        parts += [t for degree in applied.degree if (t := getattr(texts, f"filter_{degree}", ""))]
    if applied.po_version and texts.filter_po:
        parts.append(texts.filter_po.format(years="/".join(map(str, applied.po_version))))
    return parts


def no_match_line(applied: Filter | None, roles: Mapping[str, Any] | None, texts: AnswerMessages, english: bool) -> str:
    """The closing line for the filter, "" when nothing is filtered by scope or PO or the profile has no wording."""
    if applied is None or not roles:
        return ""
    parts = filter_parts(applied, roles, texts, english)
    if not parts or not texts.no_match:
        return ""
    if len(parts) == 1:
        return texts.no_match.format(filter=parts[0])
    return texts.no_match_more.format(first=parts[0], rest=", ".join(parts[1:])) if texts.no_match_more else ""
