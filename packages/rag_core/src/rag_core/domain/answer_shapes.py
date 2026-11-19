# pyright: standard
"""
Response formats v1: which instruction line of an answer shape or variant a generation needs, cheap checks of what an answer
contains and the check whether a stored conversation can make a variant of its last answer.

The lines themselves are the profile's, one set per language (messages.<language>.yaml, `shapes`): they contain no braces (they
go through LangChain's template) and each restates the citation rule. The functions take them as a mapping from the name of
a shape (table, checklist, steps, mindmap), of a length or language variant (shorter, longer, simple, english) and
`english_reminder`. Standard library only: the frames, the message helpers and the store block of the variants live in
assistant_api (sse.events, graph.nodes.generate, graph.routes, chat.turns_store).
"""

import re
from collections.abc import Mapping
from typing import Any

from rag_core.domain.language import LANGUAGES, detect, looks_english_question

FORMATS_VERSION = "v1"
SHAPES = ("text", "table", "checklist", "steps", "mindmap")
SHAPE_VARIANTS = ("table", "checklist", "steps", "mindmap")
LINE_VARIANTS = ("shorter", "longer", "simple", "english")  # the variants that add a length or language line
VARIANTS = SHAPE_VARIANTS + LINE_VARIANTS


def shape_of(value) -> str:
    """A known shape name, else "text"."""
    value = str(value or "").strip().lower()
    return value if value in SHAPES else "text"


def formats_on(state) -> bool:
    return (state.get("formats") or "") == FORMATS_VERSION


def format_instruction(state, texts: Mapping[str, str]) -> str:
    """The format instruction of a generation (appended by with_instruction): the shape's form, then the variant's length or language.
    Empty without formats=v1 and for plain text, so those answers get the plain prompt."""
    if not formats_on(state):
        return ""
    lines = []
    shape = shape_of(state.get("answer_shape"))
    if shape in SHAPE_VARIANTS:
        lines.append(texts[shape])
    variant = state.get("variant")
    if variant in LINE_VARIANTS:
        lines.append(texts[variant])
    return "\n".join(lines)


def shape_reminder(state, texts: Mapping[str, str]) -> str:
    """The shape's form line for the end of the user message (repeated after the question, where the model reads it last),
    then the length line of a shorter or longer variant, so the length is still read after the form (with the form line
    alone the longer answers shrank); not the simple or English line. Empty without formats=v1 and for plain text."""
    shape = shape_of(state.get("answer_shape"))
    if not formats_on(state) or shape not in SHAPE_VARIANTS:
        return ""
    variant = state.get("variant")
    return "\n".join([texts[shape]] + ([texts[variant]] if variant in ("shorter", "longer") else []))


def language_reminder(state, texts: Mapping[str, str]) -> str:
    """The English reminder for the user message of an English answer (see answer_language); empty otherwise."""
    return texts["english_reminder"] if answer_language(state) == "en" else ""


def answer_language(state) -> str:
    """The language of a generation, "de" or "en": English for the English variant, for a question the analysis calls English
    and for one that looks English although the analysis said German. Where the analysis named none (it failed, or said
    something else) the detector reads the question: a tie, or a question without a stopword, is German."""
    if formats_on(state) and state.get("variant") == "english":
        return "en"
    messages = state.get("messages")
    question = str(getattr(messages[-1], "content", "")) if messages else ""
    language = state.get("language")
    if language not in LANGUAGES:
        return detect(question)
    return "en" if language == "de" and looks_english_question(question) else language


# ---- what an answer contains

ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
SEPARATOR = re.compile(r"^[ \t]*\|[ \t:|-]*-[ \t:|-]*\|[ \t]*$")
TASK = re.compile(r"^[ \t]*[-*][ \t]+\[[ xX]\][ \t]+\S")
STEP = re.compile(r"^[ \t]*\d{1,2}[.)][ \t]+\S")
ITEM = re.compile(r"^[ \t]*[-*][ \t]+\S")
MINDMAP = re.compile(r"```mindmap[ \t]*\n(.*?)(?:\n[ \t]*```|\Z)", re.S)
CITE = re.compile(r"\[\d{1,2}(?:\s*,\s*\d{1,2})*\]")


def _lines(text):
    return str(text or "").splitlines()


def _table_rows(text):
    """Data rows of the Markdown tables in text (header and separator rows left out)."""
    rows, in_table = [], False
    for line in _lines(text):
        if SEPARATOR.match(line):
            in_table = True
        elif in_table and ROW.match(line):
            rows.append(line)
        elif not ROW.match(line):
            in_table = False
    return rows


def detect_shape(text) -> str:
    """What an answer contains: mindmap, table, checklist (≥ 2 items), steps (≥ 2 numbered lines) or text."""
    if MINDMAP.search(str(text or "")):
        return "mindmap"
    if _table_rows(text):
        return "table"
    lines = _lines(text)
    if sum(1 for line in lines if TASK.match(line)) >= 2:
        return "checklist"
    if sum(1 for line in lines if STEP.match(line)) >= 2:
        return "steps"
    return "text"


def structure_lines(text, shape):
    """The lines that must each carry a citation in that shape: table data rows, checklist items, steps, mind-map
    points; the whole text for plain text."""
    if shape == "mindmap":
        found = MINDMAP.search(str(text or ""))
        return [line for line in _lines(found.group(1) if found else "") if ITEM.match(line)]
    if shape == "table":
        return _table_rows(text)
    if shape == "checklist":
        return [line for line in _lines(text) if TASK.match(line)]
    if shape == "steps":
        return [line for line in _lines(text) if STEP.match(line)]
    return [text] if str(text or "").strip() else []


def cited_share(text, shape):
    """(structure lines with a citation, structure lines)."""
    lines = structure_lines(text, shape)
    return sum(1 for line in lines if CITE.search(line)), len(lines)


def shape_log_line(requested, generation, variant=None) -> str:
    """One [Shape] or [Variant] log line: what was asked for, what the answer contains, how much of it is cited."""
    produced = detect_shape(generation)
    cited, total = cited_share(generation, produced)
    head = "[Variant] name=%s" % variant if variant else "[Shape]"
    return "%s requested=%s produced=%s cited=%d/%d" % (head, shape_of(requested), produced, cited, total)


# ---- variants of the last answer: what the stored conversation must hold; the changes are chat.turns_store's


def turn_variant_problem(variant: Any, variant_of: str | None, last: Any) -> str:
    """variant_problem's reasons on the turn store: last is the conversation's newest turn a
    variant may change (rag_core.domain.history.LastTurn) or None. "" when the variant can be made."""
    if variant not in VARIANTS:
        return "unknown variant"
    if not variant_of:
        return "no turn id"
    if last is None or not last.answered:
        return "no answer to change"
    if last.turn_id != variant_of:
        return "not the last question"
    if not last.has_sources:
        return "no sources"
    return ""
