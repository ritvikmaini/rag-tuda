# pyright: standard
"""
Query analysis: parse the rewriter's JSON answer into a QueryAnalysis and derive the search queries.
Pure functions, standard library only.

The rewriter prompt ("question-rewriting") of a profile with roles asks for the analysis_schema(roles) fields:
  {"reasoning": str, "language": "de" | "en", "standalone_question": str, "intent": str, "queries_de": [str],
   "queries_en": [str], "degree": [str] | null, "programme": [str] | null, "po_version": [int] | null,
   "regulation": [str] | null, "enrolment_term": str | null, "date_range": {"from": ..., "to": ...} | null,
   "shape": "text" | "table" | "steps", "needs_more": bool}
A profile without roles asks for the base shape, which the parser reads as well:
  {"standalone_question": str, "intent": str, "search_queries": [str], "language": str,
   "date_from": "YYYY-MM-DD" | null, "date_to": ..., "answer_shape": "text" | "table" | "steps"}
"""

import asyncio
import json
import re
from datetime import date
from dataclasses import dataclass, field
from collections.abc import Awaitable, Mapping
from itertools import zip_longest
from typing import Any, List, Optional, TypeVar

from rag_core.domain.filters import REGULATIONS, degrees

INTENTS = ("lookup", "situation", "find_document", "conversation", "about_bot", "urgent", "out_of_scope")
RETRIEVAL_INTENTS = ("lookup", "situation", "find_document")
MAX_QUERIES = {"situation": 4}
DEFAULT_MAX_QUERIES = 3
MAX_ENGLISH_QUERIES = 2
ANSWER_SHAPES = ("text", "table", "steps")
LANGUAGES = {"de": "Deutsch", "en": "Englisch"}


def _nullable_array(items: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [{"type": "array", "items": items}, {"type": "null"}]}


def analysis_schema(roles: Mapping[str, Any] | None) -> dict[str, Any]:
    """The JSON schema of the rewriter's answer, in the strict shape (every property required, no extra keys, an optional
    value as a union with null, no length or pattern keywords: the parser and retrieval_queries hold the query counts).
    With roles (the dump of roles.yaml: `programmes` with `degree` and `po_versions`, `groups`) the fields are the
    profile's: the reasoning first, so the model thinks before it answers, and the degree, programme and PO year enums
    from the roles. Without roles it is the base shape of the example profile's prompt."""
    string = {"type": "string"}
    if roles is None:
        date_or_null = {"type": ["string", "null"]}
        properties = {
            "standalone_question": string,
            "intent": {"type": "string", "enum": list(INTENTS)},
            "search_queries": {"type": "array", "items": string},
            "language": string,
            "date_from": date_or_null,
            "date_to": date_or_null,
            "answer_shape": {"type": "string", "enum": list(ANSWER_SHAPES)},
        }
    else:
        programmes = roles.get("programmes", {})
        years = sorted({version["year"] for programme in programmes.values() for version in programme["po_versions"]})
        day = {"type": ["string", "null"]}
        properties = {
            "reasoning": string,
            "language": {"type": "string", "enum": list(LANGUAGES)},
            "standalone_question": string,
            "intent": {"type": "string", "enum": list(INTENTS)},
            "queries_de": {"type": "array", "items": string},
            "queries_en": {"type": "array", "items": string},
            "degree": _nullable_array({"type": "string", "enum": sorted(degrees(roles))}),
            "programme": _nullable_array({"type": "string", "enum": sorted(programmes)}),
            "po_version": _nullable_array({"type": "integer", "enum": years}),
            "regulation": _nullable_array({"type": "string", "enum": list(REGULATIONS)}),
            "enrolment_term": {"type": ["string", "null"]},
            "date_range": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"from": day, "to": day},
                        "required": ["from", "to"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ]
            },
            "shape": {"type": "string", "enum": list(ANSWER_SHAPES)},
            "needs_more": {"type": "boolean"},
        }
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
YEAR_TEXT = re.compile(r"[0-9]{4}")  # ASCII digits only: int() takes others ("²" raises, "２０２３" is 2023)
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
LANGUAGE_NAME = re.compile(r"[A-Za-zÄÖÜäöüß]{2,20}")
VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_T = TypeVar("_T")


@dataclass
class QueryAnalysis:
    standalone_question: str
    intent: str = "lookup"
    search_queries: List[str] = field(default_factory=list)
    language: str = "Deutsch"
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    parsed: str = "json"  # json | fallback
    answer_shape: str = "text"  # text | table | steps | checklist
    queries_en: List[str] = field(default_factory=list)  # for an English question
    # the enum fields as the model wrote them (filters.resolve_filter checks the values); None = not given
    degree: Optional[List[str]] = None
    programme: Optional[List[str]] = None
    po_version: Optional[List[int]] = None
    regulation: Optional[List[str]] = None
    enrolment_term: Optional[str] = None
    needs_more: bool = False  # the question needs sources of more than one document


def _first_json_value(text):
    """The first JSON object in text (code fences and prose around it are ignored, so "[1]" in a preamble cannot
    hide it); None without one."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                value, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _iso_date(value):
    """value if it is a real calendar date written as YYYY-MM-DD, else None."""
    if not isinstance(value, str) or not ISO_DATE.fullmatch(value):
        return None
    try:
        date.fromisoformat(value)
    except ValueError:
        return None
    return value


def _language(value):
    """The language as a plain name: "de" and "en" become "Deutsch" and "Englisch", a name stays, anything else is "Deutsch"."""
    value = str(value or "").strip()
    if value.lower() in LANGUAGES:
        return LANGUAGES[value.lower()]
    return value if LANGUAGE_NAME.fullmatch(value) else "Deutsch"


def _values(value, number=False):
    """The strings (the whole numbers, with number) of an array, trimmed and in lower case; None for no array or no value in it."""
    if not isinstance(value, list):
        return None
    found = []
    for item in value:
        if number:
            item = int(item) if isinstance(item, str) and YEAR_TEXT.fullmatch(item.strip()) else item
            if isinstance(item, int) and not isinstance(item, bool):
                found.append(item)
        elif isinstance(item, str) and item.strip():
            found.append(item.strip().lower())
    return found or None


def _queries(value, limit):
    raw = value if isinstance(value, list) else []
    return _dedupe(raw)[:limit]


def _norm(q):
    return " ".join(q.casefold().split()).rstrip(" ?.!")


def _dedupe(queries):
    seen, out = set(), []
    for q in queries:
        if not isinstance(q, str):
            continue
        q = " ".join(q.split())
        if q and _norm(q) not in seen:
            seen.add(_norm(q))
            out.append(q)
    return out


def _answer_shape(value, intent):
    """The shape of the answer: a life event always gets a checklist, the model chooses only
    between text, table and steps for the searching intents, everything else is text."""
    if intent == "situation":
        return "checklist"
    if intent not in RETRIEVAL_INTENTS:
        return "text"
    value = str(value or "").strip().lower()
    return value if value in ANSWER_SHAPES else "text"


def parse_query_analysis(content: str | None, original_question: str) -> QueryAnalysis:
    """QueryAnalysis from the model's raw text; never raises."""
    text = THINK_BLOCK.sub("", content or "")
    value = _first_json_value(text)
    if isinstance(value, dict):
        intent = str(value.get("intent", "")).strip().lower()
        if intent not in INTENTS:
            intent = "lookup"
        standalone = " ".join(str(value.get("standalone_question") or "").split()) or original_question
        queries = _queries(
            value.get("queries_de", value.get("search_queries")), MAX_QUERIES.get(intent, DEFAULT_MAX_QUERIES)
        )
        window = value.get("date_range")
        window = window if isinstance(window, dict) else {}
        dates = [
            _iso_date(window.get("from", value.get("date_from"))),
            _iso_date(window.get("to", value.get("date_to"))),
        ]
        term = value.get("enrolment_term")
        return QueryAnalysis(
            standalone,
            intent,
            queries,
            _language(value.get("language")),
            dates[0],
            dates[1],
            "json",
            _answer_shape(value.get("shape", value.get("answer_shape")), intent),
            _queries(value.get("queries_en"), MAX_ENGLISH_QUERIES),
            _values(value.get("degree")),
            _values(value.get("programme")),
            _values(value.get("po_version"), number=True),
            _values(value.get("regulation")),
            " ".join(term.split()) or None if isinstance(term, str) else None,
            value.get("needs_more") is True,
        )
    return QueryAnalysis(original_question, "lookup", [original_question], parsed="fallback")


def retrieval_queries(analysis: QueryAnalysis, original_question: str, first_turn: bool) -> list[str]:
    """Queries for search and reranking: the standalone question first, then the model's German search queries, and for an
    English question its English ones between them (one after each German one, so a short budget keeps both languages).
    The raw user text is added only on the first turn of a conversation (where it cannot contain unresolved references)."""
    english = analysis.queries_en if analysis.language == "Englisch" else []
    searches = [q for pair in zip_longest(analysis.search_queries, english) for q in pair if q]
    queries = [analysis.standalone_question] + searches
    if first_turn:
        queries.append(original_question)
    return _dedupe(queries)


async def bounded(awaitable: Awaitable[_T], timeout_s: float) -> tuple[_T | None, str]:
    """(result, "") or (None, reason): the call is cancelled after timeout_s seconds,
    and any exception becomes a reason instead of propagating."""
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_s), ""
    except asyncio.TimeoutError:
        return None, f"TimeoutError after {timeout_s:g} s"
    except Exception as exc:  # the caller falls back to the plain question
        return None, f"{type(exc).__name__}: {exc}"


def fill_template(text: str, variables: Mapping[str, Any]) -> str:
    """Replace {{name}} placeholders; unknown names and all other braces are left untouched."""
    return VAR_PATTERN.sub(lambda m: str(variables.get(m.group(1), m.group(0))), text)
