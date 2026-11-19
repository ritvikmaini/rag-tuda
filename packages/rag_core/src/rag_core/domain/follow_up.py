# pyright: standard
"""The iterative round's pure parts, standard library only: the schema of the follow-up model's answer, the digest of the
kept sections it is shown, and the parser of its answer.

The follow-up prompt ("follow-up-queries") asks for {"reasoning": str, "queries": [str], "done": bool}."""

from collections.abc import Iterable, Sequence
from typing import Any

from rag_core.domain.query_analysis import THINK_BLOCK, _dedupe, _first_json_value, _norm

MAX_FOLLOW_UP_QUERIES = 2
DIGEST_TOKENS = 1500  # what the model is shown of the kept sections, in tokens
CHILD_CHARS = 300  # of each section's matched child


def follow_up_schema() -> dict[str, Any]:
    """The JSON schema of the follow-up answer, in the strict shape (every property required, no extra keys, no length
    keywords: parse_follow_up holds the query count)."""
    properties = {
        "reasoning": {"type": "string"},
        "queries": {"type": "array", "items": {"type": "string"}},
        "done": {"type": "boolean"},
    }
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def sections_digest(sections: Sequence[Any], max_chars: int) -> str:
    """One entry per kept section, best first as given: its title and section path (the header) and the child that matched,
    as many entries as fit in max_chars."""
    entries: list[str] = []
    used = 0
    for section in sections:
        meta = section.metadata or {}
        header = " > ".join(str(meta[key]) for key in ("title", "section_path") if meta.get(key)) or "Dokument"
        child = " ".join(str(meta.get("matched_child_text") or "").split())[:CHILD_CHARS].rstrip()
        entry = f"- {header}" + (f"\n  {child}" if child else "")
        if used + len(entry) > max_chars:
            break
        entries.append(entry)
        used += len(entry) + 1
    return "\n".join(entries)


def parse_follow_up(content: str | None, searched: Iterable[str]) -> list[str]:
    """The follow-up queries of the model's raw text: at most MAX_FOLLOW_UP_QUERIES, none that was searched already; none when
    the model says it is done, answers without a JSON object or names nothing. Never raises."""
    value = _first_json_value(THINK_BLOCK.sub("", content or ""))
    if not isinstance(value, dict) or value.get("done") is True:
        return []
    raw = value.get("queries")
    seen = {_norm(q) for q in searched}
    return [q for q in _dedupe(raw if isinstance(raw, list) else []) if _norm(q) not in seen][:MAX_FOLLOW_UP_QUERIES]
