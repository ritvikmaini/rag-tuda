# pyright: standard
"""
Query analysis: parse the rewriter's JSON answer into a QueryAnalysis and derive the search queries.
Pure functions, standard library only.

The rewriter prompt ("question-rewriting") asks for:
  {"standalone_question": str, "intent": str, "search_queries": [str],
   "language": str, "date_from": "YYYY-MM-DD" | null, "date_to": ...,
   "answer_shape": "text" | "table" | "steps"}
"""

import asyncio
import json
import re
from datetime import date
from dataclasses import dataclass, field
from collections.abc import Awaitable, Mapping
from typing import Any, List, Optional, TypeVar

INTENTS = ("lookup", "situation", "find_document", "conversation", "about_bot", "urgent", "out_of_scope")
RETRIEVAL_INTENTS = ("lookup", "situation", "find_document")
MAX_QUERIES = {"situation": 4}
DEFAULT_MAX_QUERIES = 3
ANSWER_SHAPES = ("text", "table", "steps")


def analysis_schema(profile: Any) -> dict[str, Any]:
    """The JSON schema of the rewriter's answer, in the strict shape (every property required, no extra keys, a nullable
    date as a union with null). The fields are the ones the prompt asks for and parse_query_analysis reads; the enums are
    the module's (profile is the argument for enums that come from a profile)."""
    string = {"type": "string"}
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
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
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
    """A plain language name ("Deutsch", "Englisch"); anything else becomes "Deutsch"."""
    value = str(value or "").strip()
    return value if LANGUAGE_NAME.fullmatch(value) else "Deutsch"


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
        raw = value.get("search_queries") or []
        queries = _dedupe(raw if isinstance(raw, list) else [])
        limit = MAX_QUERIES.get(intent, DEFAULT_MAX_QUERIES)
        dates = [_iso_date(value.get("date_from")), _iso_date(value.get("date_to"))]
        language = _language(value.get("language"))
        shape = _answer_shape(value.get("answer_shape"), intent)
        return QueryAnalysis(standalone, intent, queries[:limit], language, dates[0], dates[1], "json", shape)
    return QueryAnalysis(original_question, "lookup", [original_question], parsed="fallback")


def retrieval_queries(analysis: QueryAnalysis, original_question: str, first_turn: bool) -> list[str]:
    """Queries for search and reranking: the standalone question first, then the
    model's search queries. The raw user text is added only on the first turn of a
    conversation (where it cannot contain unresolved references)."""
    queries = [analysis.standalone_question] + list(analysis.search_queries)
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
