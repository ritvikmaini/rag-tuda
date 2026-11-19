"""The conversation's LangChain messages: the history policy's turns as prompt messages, the new question as stored, the
stored turns as messages, the filter the next question starts from, and a variant's run input."""

import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from rag_core.domain.filters import Filter
from rag_core.domain.history import TS_KEY, Answer, PastTurn, Turn, VariantInput, clean_answer
from rag_core.domain.notes import Texts, strip_lines

from assistant_api.chat.turns_store import stored_filter


def to_messages(pairs: Sequence[tuple[str, str]]) -> list[BaseMessage]:
    """(role, text) pairs, as the prompt service compiles them, as chat messages."""
    classes: dict[str, Any] = {"system": SystemMessage, "assistant": AIMessage}
    return [classes.get(role, HumanMessage)(content=text) for role, text in pairs]


def as_messages(turns: Sequence[Turn]) -> list[BaseMessage]:
    """The turns as new chat messages for a prompt (answers without citation markers)."""
    out: list[BaseMessage] = []
    for question, answer in turns:
        out.append(HumanMessage(content=str(question.content)))
        out.append(AIMessage(content=clean_answer(answer.content)))
    return out


def line_texts(deps: Any) -> list[Texts]:
    """The wording of the appended lines in each language the deps hold (a stand-in without messages has none)."""
    found = (getattr(getattr(deps, name, None), "answer", None) for name in ("messages", "messages_en"))
    return [texts for texts in found if texts is not None]


def without_lines(turns: Sequence[Turn], texts: Sequence[Texts]) -> list[Turn]:
    """The turns with the lines the code appended to each answer removed (the stored turn and the page keep them)."""
    return [(question, Answer(strip_lines(str(answer.content), texts))) for question, answer in turns]


def insert_history(
    messages: Sequence[BaseMessage], history: Sequence[BaseMessage], position: str = "after_system"
) -> list[BaseMessage]:
    """The prompt with the earlier turns after the system message (default), or first ("first": selectable by config)."""
    messages, history = list(messages), list(history)
    if position == "first":
        return history + messages
    split = 1 if messages and isinstance(messages[0], SystemMessage) else 0
    return messages[:split] + history + messages[split:]


def question_message(question: str, turn_id: Optional[str], now: datetime) -> HumanMessage:
    """The new question as stored: the page's turn id (or a fresh UUID) and the time it was asked (UTC, ISO 8601)."""
    return HumanMessage(content=question, id=turn_id or str(uuid.uuid4()), additional_kwargs={TS_KEY: now.isoformat()})


def history_messages(past: Sequence[PastTurn]) -> list[BaseMessage]:
    """The stored turns as messages: each question with its id and time, its answer on screen. The history policy
    (context_turns, select_turns, as_messages) reads them."""
    out: list[BaseMessage] = []
    for p in past:
        out.append(HumanMessage(content=p.question, id=p.turn_id, additional_kwargs={TS_KEY: p.asked_at.isoformat()}))
        out.append(AIMessage(content=p.answer))
    return out


def previous_filter(past: Sequence[PastTurn]) -> Filter | None:
    """The hard filter the newest of the earlier turns ran with: the start of the next question's filter. None when there is
    no earlier turn, when it ran unfiltered or when its stored filter is unreadable (logged)."""
    return stored_filter(past[-1].applied_filter) if past else None


def documents(candidates: Sequence[Mapping[str, Any]]) -> list[Document]:
    return [Document(page_content=c["page_content"], metadata=dict(c.get("metadata") or {})) for c in candidates]


def variant_inputs(v: VariantInput, variant: str, formats: str) -> dict[str, Any]:
    """A variant's run input from the store: the history before the target, the target question last, its sources in their
    grouped order, and what the analysis found."""
    docs = documents(v.candidates)
    return {
        "messages": history_messages(v.history)
        + [HumanMessage(content=v.question, id=v.turn_id, additional_kwargs={TS_KEY: v.asked_at.isoformat()})],
        "variant": variant,
        "answer_shape": v.shape,
        "formats": formats,
        "raw_docs": docs,
        "grouped_raw_docs": docs,
        "standalone_question": v.standalone_question,
        "intent": v.intent,
        "language": v.language,
        "scope_fallback": v.scope_fallback,
        "filter": v.filter,
    }
