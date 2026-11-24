# pyright: standard
"""Conversation history policy.

The model sees only completed turns (a question followed by an answer with text): failed, interrupted or stopped
questions and fallback replies never reach it. The newest turns count, whole and without gaps, within a turn count, an
age limit and a token budget; the citation markers of earlier answers are removed. An edit or a retry cuts the stored
conversation at the edited question (replace_from), and every new question carries the page's turn id and its time.
Messages are read by attribute (rag_core.domain.types.Message): LangChain's fit, and nothing here imports it; the LangChain
side (constructors, insertion) is assistant_api.graph's, the cut is the turn store's (assistant_api.chat)."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Tuple

from rag_core.domain.filters import Filter
from rag_core.domain.source_meta import CITATION_GROUP
from rag_core.domain.types import Message

Turn = Tuple[Any, Any]
Count = Callable[[str], Awaitable[int]]

TS_KEY = "ts"
MAX_KEPT = 5  # the page sends at most its five newest kept questions
NO_HISTORY = "Kein Verlauf"
MESSAGE_OVERHEAD_TOKENS = 8  # the per-message allowance context_budget uses
ELLIPSIS = " …"
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# "[2]", "[1, 3]": the markers the page links (its CITATION_PATTERN), with the space before them
CITATION_MARKER = re.compile(r"\s?" + CITATION_GROUP.pattern)
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class Answer:
    """An earlier answer as select_turns keeps it (cleaned or shortened); only .content is read afterwards."""

    content: str
    type: str = "ai"


@dataclass(frozen=True)
class PastTurn:
    """A completed, not superseded turn as the history reads it from the turn store: the answer on
    screen (the active version, else the original) and the time the question was stored."""

    turn_id: str
    question: str
    answer: str
    asked_at: datetime
    applied_filter: Optional[dict[str, Any]] = (
        None  # the turn's hard filter as stored (Filter.as_json()); None = it ran unfiltered
    )


@dataclass(frozen=True)
class LastTurn:
    """The conversation's newest turn a variant may change. Too-long and busy turns are skipped: they never reached the model,
    so they are not part of the conversation a variant may change."""

    turn_id: str
    answered: bool  # completed with answer text
    has_sources: bool  # a turn_context row (a completed lookup)


@dataclass(frozen=True)
class VariantInput:
    """What a variant's run needs from the store: the target question, the history before it, its stored candidates in
    their grouped order with what the query analysis found, the shape the variant is written in and the hard filter the turn
    ran with."""

    turn_id: str
    question: str
    asked_at: datetime
    history: tuple[PastTurn, ...]
    candidates: tuple[dict[str, Any], ...]
    standalone_question: str
    intent: str
    language: str
    shape: str
    scope_fallback: bool = False  # the turn was an out-of-scope question answered from its documents
    filter: Optional[Filter] = None  # the hard filter the turn ran with (None = unfiltered): the variant runs on it


def valid_turn_id(value) -> Optional[str]:
    """A canonical UUID in lower case, or None (the caller ignores anything else)."""
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if UUID_PATTERN.match(value) else None


def kept_ids(value) -> Optional[List[str]]:
    """The page's kept questions (`kept`: ids, oldest first, comma-separated) as at most the newest MAX_KEPT ids; [] for an
    empty value (the page kept nothing); None when absent or when any entry is not a canonical UUID (never "nothing kept")."""
    if not isinstance(value, str):
        return None
    parts = [p for p in (part.strip() for part in value.split(",")) if p]
    ids = [valid_turn_id(p) for p in parts]
    if any(i is None for i in ids):
        return None
    return ids[-MAX_KEPT:]  # pyright: ignore[reportReturnType]


def completed_turns(messages: Sequence[Message]) -> List[Turn]:
    """Every question directly followed by an answer with text, oldest first. The current question (the last message)
    and every question without an answer (failed, interrupted, stopped, fallback reply) drop out."""
    messages = list(messages)
    return [
        (question, answer)
        for question, answer in zip(messages, messages[1:])
        if question.type == "human" and answer.type == "ai" and str(answer.content).strip()
    ]


def message_time(message: Message) -> Optional[datetime]:
    """The time a question was asked (stored by question_message), or None."""
    raw = (getattr(message, "additional_kwargs", None) or {}).get(TS_KEY)
    if not isinstance(raw, str):
        return None
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else None


def fresh_turns(turns: Sequence[Turn], max_age_hours: float, now: datetime) -> List[Turn]:
    """Turns whose question is at most max_age_hours old; 0 or less switches the limit off. A question without a
    readable time counts as stale."""
    if not max_age_hours or max_age_hours <= 0:
        return list(turns)
    limit = now - timedelta(hours=max_age_hours)
    kept = []
    for question, answer in turns:
        stamp = message_time(question)
        if stamp is not None and stamp >= limit:
            kept.append((question, answer))
    return kept


def context_turns(messages: Sequence[Message], max_turns: int, max_age_hours: float, now: datetime) -> List[Turn]:
    """The completed, fresh turns the model may see: at most max_turns, oldest first."""
    if max_turns <= 0:
        return []
    return fresh_turns(completed_turns(messages), max_age_hours, now)[-max_turns:]


def clean_answer(text) -> str:
    """An earlier answer without its citation markers: they point into that turn's source list, not the new one."""
    return CITATION_MARKER.sub("", str(text)).strip()


async def shorten(text: str, budget: int, count: Count) -> str:
    """The longest run of whole leading sentences that fits budget tokens, plus an ellipsis; '' when none fits."""
    kept = ""
    for sentence in SENTENCE_END.split(text.strip()):
        candidate = (kept + " " + sentence).strip()
        if await count(candidate + ELLIPSIS) > budget:
            break
        kept = candidate
    return kept + ELLIPSIS if kept else ""


async def select_turns(turns: Sequence[Turn], max_tokens: int, count: Count) -> List[Turn]:
    """Newest first, whole turns, no gaps: stop at the first turn that no longer fits. When even the newest turn is too
    long, keep it with its answer shortened at a sentence end (the latest turn matters most for a follow-up).
    The answers come back cleaned (no citation markers)."""
    chosen: List[Turn] = []
    used = 0
    for question, answer in reversed(list(turns)):
        answer_text = clean_answer(answer.content)
        q_cost = await count(str(question.content)) + MESSAGE_OVERHEAD_TOKENS
        a_cost = await count(answer_text) + MESSAGE_OVERHEAD_TOKENS
        if used + q_cost + a_cost <= max_tokens:
            chosen.insert(0, (question, Answer(answer_text)))
            used += q_cost + a_cost
            continue
        if not chosen:
            short = await shorten(answer_text, max_tokens - q_cost - MESSAGE_OVERHEAD_TOKENS, count)
            if short:
                chosen.insert(0, (question, Answer(short)))
        break
    return chosen


def history_text(turns: Sequence[Turn], max_chars: int = 500) -> str:
    """The query analysis' "User:/Assistant:" text, each message cut at max_chars."""
    if not turns:
        return NO_HISTORY
    lines = []
    for question, answer in turns:
        for role, text in (("User", str(question.content)), ("Assistant", clean_answer(answer.content))):
            cut = text[:max_chars] + "..." if len(text) > max_chars else text
            lines.append("%s: %s" % (role, cut))
    return "\n\n".join(lines)


def conversation_state(turns_seen: int) -> str:
    """One sentence for the direct reply: how many earlier turns it sees before the current message; the model is told, not
    left to infer, whether there is anything to rework."""
    if turns_seen <= 0:
        return "Vor der aktuellen Nachricht gibt es keine früheren Fragen und Antworten."
    if turns_seen == 1:
        return "Vor der aktuellen Nachricht steht eine frühere Frage mit Antwort."
    return "Vor der aktuellen Nachricht stehen %d frühere Fragen mit Antworten." % turns_seen


def cut_turns(turn_ids: Sequence[str], replace_from: str | None, kept: Sequence[str] | None = None) -> list[str]:
    """The turns an edit or a retry supersedes, oldest first: replace_from and every later turn. When replace_from was never
    stored, the page's kept questions place the cut: after the newest stored one, before everything when the page kept
    nothing, nowhere without kept. An answer belongs to its turn."""
    ids = list(turn_ids)
    if not replace_from:
        return []
    if replace_from in ids:
        return ids[ids.index(replace_from) :]
    if kept is None:
        return []
    if not kept:
        return ids
    found = [n for n, i in enumerate(ids) if i in kept]
    return ids[found[-1] + 1 :] if found else []
