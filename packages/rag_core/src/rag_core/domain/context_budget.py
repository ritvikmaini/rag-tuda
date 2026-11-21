# pyright: standard
"""
Context budget guard for the generation step.

Fits the conversation history and the retrieved sources into the model's
context window before the LLM call, so that no request is rejected for being
too long. Token counts come from vLLM's /tokenize endpoint; for every other
provider they are estimated from the card's characters per token. When the
/tokenize call fails, a conservative estimate of 2 characters per token (the
worst ratio measured on German source text) is used, which can only under-fill.
"""

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from rag_core.domain.child_text import child_body
from rag_core.domain.source_meta import SourceLabels, describe_source, strip_source_text
from rag_core.domain.types import Doc, Message

logger = logging.getLogger(__name__)

PAGE_MARKER = re.compile(r"\[Page (\d+)\]")
CHARS_PER_TOKEN_FALLBACK = 2.0
MESSAGE_OVERHEAD_TOKENS = 8
SOURCE_OVERHEAD_TOKENS = 6  # "[nn] " in a source header and the blank line before it
CONTEXT_LEAD_TOKENS = 12  # the lead line of join_sources (labels.sources_lead)


def format_source(doc: Doc, text: Optional[str] = None, page: Optional[int] = None, *, labels: SourceLabels) -> str:
    """One source as the prompt shows it: title, origin | type | date | page, the document's scope statement (section metadata
    "scope"), link, then the text. fit_context() adds the number [n] in front of the title."""
    metadata = doc.metadata or {}
    info = describe_source(metadata, labels)
    if page is None:
        page = metadata.get("page_number", -1)
    if text is None:
        text = doc.page_content
    fields = [
        info["origin"],
        info["type"],
        f"{labels.date}: {info['date']}" if info["date"] else "",
        f"{labels.page} {page}" if isinstance(page, int) and page > 0 else "",
    ]
    lines = ["### " + info["title"]]
    if any(fields):
        lines.append(" | ".join(f for f in fields if f))
    if labels.scope and metadata.get("scope"):
        lines.append(labels.scope + ": " + metadata["scope"])
    if info["url"]:
        lines.append(f"{labels.link}: {info['url']}")
    return "\n".join(lines) + "\n\n" + strip_source_text(text)


def join_sources(parts: Sequence[str], labels: SourceLabels) -> str:
    """The context string the prompt receives, its framing words the labels'."""
    if not parts:
        return labels.nothing_found
    return labels.sources_lead.format(n=len(parts)) + "\n\n" + "\n\n".join(parts)


def page_of_position(parent_text: str, pos: int, default: int = -1) -> int:
    """Page of a position in a parent: the last [Page N] marker before it, else the first marker, else default."""
    markers = list(PAGE_MARKER.finditer(parent_text))
    if not markers:
        return default
    page = None
    for m in markers:
        if m.start() <= pos:
            page = int(m.group(1))
        else:
            break
    return page if page is not None else int(markers[0].group(1))


def _window_and_page(doc: Doc, window_chars: int) -> Tuple[str, int]:
    """The matched child text plus window_chars of the parent on each side, and its page."""
    parent = doc.page_content or ""
    metadata = doc.metadata or {}
    # Children are indexed with '<title>: ' or a context header in front; the parent text has neither.
    child = child_body(metadata, metadata.get("matched_child_text") or "")
    anchor = child[:120]
    pos = parent.find(anchor) if anchor else -1
    if pos < 0:
        pos, start = 0, 0
        end = min(len(parent), max(len(child), 400) + 2 * window_chars)
    else:
        start = max(0, pos - window_chars)
        end = min(len(parent), pos + len(child) + window_chars)
    page = page_of_position(parent, pos, metadata.get("page_number", -1))
    return parent[start:end], page


def to_chat_dicts(messages: Sequence[Message]) -> List[Dict[str, str]]:
    role = {"human": "user", "ai": "assistant", "system": "system"}
    return [{"role": role.get(m.type, "user"), "content": str(m.content)} for m in messages]


class TokenCounter(Protocol):
    """What fit_context needs from the token counter (vLLM's /tokenize behind rag_core.adapters.vllm_tokenize, or the
    estimate of CharsPerTokenCounter)."""

    max_model_len: int
    exact: bool

    async def count_text(self, text: str) -> int: ...

    async def count_chat(self, messages: List[Dict[str, str]]) -> int: ...


class CharsPerTokenCounter:
    """The token count of a provider with no tokenizer endpoint: the characters divided by the card's characters per token,
    rounded up."""

    exact = False

    def __init__(self, chars_per_token: float, window: int) -> None:
        self.chars_per_token = chars_per_token
        self.max_model_len = window

    async def count_text(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token)

    async def count_chat(self, messages: List[Dict[str, str]]) -> int:
        return (
            sum(math.ceil(len(m["content"]) / self.chars_per_token) + MESSAGE_OVERHEAD_TOKENS for m in messages)
            + MESSAGE_OVERHEAD_TOKENS
        )


@dataclass(frozen=True)
class ContextBudget:
    reserve: int  # card.budget.answer_tokens: the answer's tokens
    margin: int  # profile.guard.safety_margin
    window_chars: int  # profile.guard.source_window_chars
    full_sources: int  # card.budget.full_sources


def rank_score(doc: Doc) -> float:
    """A candidate's ranking score (rank_score, else the cross-encoder's; 0 without either)."""
    metadata = doc.metadata or {}
    return metadata.get("rank_score", metadata.get("cross_encoder_score")) or 0


async def fit_context(
    budget: ContextBudget,
    counter: TokenCounter,
    fixed_prompt: Sequence[Message],
    history: Sequence[Message],
    candidates: Sequence[Doc],
    *,
    labels: SourceLabels,
) -> Dict[str, Any]:
    """
    Fit history and sources into the window.

    fixed_prompt: the rendered generation prompt with an empty context and no history.
    history: the earlier turns chosen by history.select_turns, oldest first (counted, never trimmed).
    candidates: ranked parent documents in the order they should appear (grouped_raw_docs).
    Returns {"history": kept messages, "context": formatted sources, "report": numbers}.
    """
    reserve = budget.reserve
    margin = budget.margin
    window_chars = budget.window_chars

    fixed = await counter.count_chat(to_chat_dicts(fixed_prompt))
    window = counter.max_model_len

    # 1. History: chosen before this call (history.select_turns: whole earlier turns within the card's
    #    budget.history_tokens); counted here, never trimmed.
    history = list(history)
    counts = await asyncio.gather(*(counter.count_text(str(m.content)) for m in history))
    kept: List[Message] = history
    history_tokens = sum(n + MESSAGE_OVERHEAD_TOKENS for n in counts)

    # 2. Sources, breadth first: the top `full_sources` in full, every further candidate as a
    #    window around its matched passage, then windows upgraded to full parents while budget remains.
    available = window - reserve - margin - fixed - history_tokens - CONTEXT_LEAD_TOKENS
    full_min = budget.full_sources
    ranked = sorted(
        candidates,
        key=rank_score,
        reverse=True,
    )
    windows = [_window_and_page(d, window_chars) for d in ranked]
    pages = {id(d): page for d, (_, page) in zip(ranked, windows)}
    full_texts = [format_source(d, page=page, labels=labels) for d, (_, page) in zip(ranked, windows)]
    win_texts = [format_source(d, text=text, page=page, labels=labels) for d, (text, page) in zip(ranked, windows)]
    full_counts, win_counts = await asyncio.gather(
        asyncio.gather(*(counter.count_text(t) for t in full_texts)),
        asyncio.gather(*(counter.count_text(t) for t in win_texts)),
    )
    full_counts = [n + SOURCE_OVERHEAD_TOKENS for n in full_counts]
    win_counts = [n + SOURCE_OVERHEAD_TOKENS for n in win_counts]
    chosen: Dict[int, str] = {}
    kind: Dict[int, str] = {}
    # ids cut to fit: their counted cost is the cut text's, so the upgrade below must not price them as the window
    truncated: set[int] = set()
    remaining = available
    for i, doc in enumerate(ranked):
        if i < full_min and full_counts[i] <= remaining:
            chosen[id(doc)], kind[id(doc)] = full_texts[i], "full"
            remaining -= full_counts[i]
        elif win_counts[i] <= remaining:
            chosen[id(doc)], kind[id(doc)] = win_texts[i], "window"
            remaining -= win_counts[i]
        elif not chosen:
            # Nothing fits yet: the best source goes in truncated rather than not at all.
            text, page = windows[i]
            header_len = len(format_source(doc, text="", page=page, labels=labels))
            cut = max(200, int((remaining - SOURCE_OVERHEAD_TOKENS) * CHARS_PER_TOKEN_FALLBACK) - header_len)
            chosen[id(doc)], kind[id(doc)] = format_source(doc, text=text[:cut], page=page, labels=labels), "window"
            truncated.add(id(doc))
            remaining -= await counter.count_text(chosen[id(doc)]) + SOURCE_OVERHEAD_TOKENS
    for i, doc in enumerate(ranked):  # upgrades, best first
        if kind.get(id(doc)) == "window" and id(doc) not in truncated and full_counts[i] - win_counts[i] <= remaining:
            chosen[id(doc)], kind[id(doc)] = full_texts[i], "full"
            remaining -= full_counts[i] - win_counts[i]
    n_full = sum(1 for k in kind.values() if k == "full")
    n_window = sum(1 for k in kind.values() if k == "window")
    n_dropped = len(ranked) - n_full - n_window

    parts, sources = [], []
    for n, d in enumerate([d for d in candidates if id(d) in chosen], 1):
        parts.append("### [%d] %s" % (n, chosen[id(d)][len("### ") :]))
        sources.append(
            {
                "n": n,
                **describe_source(d.metadata, labels),
                "page": pages.get(id(d)),
                "section_id": (d.metadata or {}).get("section_id"),
                "kind": kind[id(d)],
            }
        )
    report = {
        "window": window,
        "reserve": reserve,
        "margin": margin,
        "fixed": fixed,
        "history_tokens": history_tokens,
        "history_kept": len(kept),
        "history_total": len(history),
        "sources_full": n_full,
        "sources_windowed": n_window,
        "sources_dropped": n_dropped,
        "candidates": len(candidates),
        "source_tokens": available - remaining,
        "left": remaining,
        "exact": counter.exact,
    }
    logger.info(
        "[ContextBudget] window=%d reserve=%d margin=%d fixed=%d history=%d (%d/%d msgs) "
        "sources full=%d windowed=%d dropped=%d of %d | source_tokens=%d left=%d exact=%s",
        window,
        reserve,
        margin,
        fixed,
        history_tokens,
        len(kept),
        len(history),
        n_full,
        n_window,
        n_dropped,
        len(candidates),
        available - remaining,
        remaining,
        counter.exact,
    )
    return {"history": kept, "context": join_sources(parts, labels), "report": report, "sources": sources}
