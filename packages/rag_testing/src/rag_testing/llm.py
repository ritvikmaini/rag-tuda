"""The fake LLM's logic: which prompt a call carries, what the
per-test script answers, how the answer streams, and /tokenize's count. Deterministic; standard library only.

A script is a list of rules; the first rule whose purpose, `match` (a part of the question) and `system` (a part of the
system message) fit a call answers it:
  {"purpose": "analysis", "match": "Witz", "reply": {"intent": "out_of_scope"}}       analysis JSON fields to change
  {"purpose": "follow_up", "reply": {"queries": ["APB Rücktritt"], "done": false}}    follow-up JSON fields to change
  {"purpose": "answer", "reply": "Text [1].", "first_delay_s": 8, "token_delay_s": 0}  answer text and its timing
  {"purpose": "answer", "fail": 500}                                                  an HTTP error instead
Without a fitting rule a call gets the default reply of its purpose. The analysis reply takes the shape its prompt asks for:
the profile's (the prompt names `queries_de`: reasoning, queries_de, queries_en, the filter fields, shape, needs_more) or the
base one; a rule written with the base keys (`search_queries`, `answer_shape`) is read into the profile's.
  {"purpose": "analysis", "match": "Vergleich", "reply": {"needs_more": true}}         the profile's shape: another round"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

MODEL = "Qwen/Qwen3-14B-AWQ"
MAX_MODEL_LEN = 8192
# (purpose, a text that only this prompt's system message carries; the profile's prompts/*.yaml, pinned by
# tests/contracts/test_harness_contracts.py)
MARKERS = (
    ("follow_up", "Folgesuche"),
    ("analysis", '"standalone_question"'),
    ("direct", "Die Nachricht wurde als"),
    ("no_docs", "keine passenden Informationen gefunden"),
    ("answer", "Belege jede Aussage"),
)
FOLLOW_UP_REPLY = {"reasoning": "", "queries": [], "done": True}  # the iterative round finds nothing more to ask
DEFAULT_REPLIES = {
    "answer": "Die gesuchte Angabe steht in der Quelle [1].",
    "direct": "Dabei kann ich Ihnen leider nicht helfen.",
    "no_docs": "Dazu habe ich keine passenden Dokumente gefunden.",
    "unknown": "OK.",
}
# a prompt that asks for the profile's analysis shape carries this key in its JSON contract
PROFILE_SHAPE_MARK = '"queries_de"'
PROFILE_KEYS = {"search_queries": "queries_de", "answer_shape": "shape"}
QUESTION_MARKS = {"analysis": "Nachricht: ", "answer": "# Frage\n"}
PIECE = re.compile(r"\S+\s*|\s+")

Message = Mapping[str, Any]
Rule = Mapping[str, Any]


def text_of(message: Message) -> str:
    content = message.get("content")
    if isinstance(content, list):  # OpenAI content parts
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def system_of(messages: Sequence[Message]) -> str:
    return next((text_of(m) for m in messages if m.get("role") == "system"), "")


def purpose(messages: Sequence[Message]) -> str:
    system = system_of(messages)
    return next((name for name, marker in MARKERS if marker in system), "unknown")


def question(messages: Sequence[Message], kind: str) -> str:
    """The user's question in a call: after the last 'Nachricht: ' (analysis) or '# Frage' (answer), else the last user
    message."""
    last = next((text_of(m) for m in reversed(messages) if m.get("role") == "user"), "")
    mark = QUESTION_MARKS.get(kind)
    if mark and mark in last:
        return last.rsplit(mark, 1)[1].strip()
    return last.strip()


def pick(rules: Sequence[Rule], kind: str, messages: Sequence[Message]) -> Rule:
    """The first rule for this purpose whose `match` is in the question and `system` in the system message; {} if none."""
    asked, system = question(messages, kind), system_of(messages)
    for rule in rules:
        if (
            rule.get("purpose", kind) == kind
            and str(rule.get("match", "")) in asked
            and str(rule.get("system", "")) in system
        ):
            return rule
    return {}


def reply(rule: Rule, kind: str, messages: Sequence[Message]) -> str:
    if kind == "analysis":
        asked = question(messages, kind)
        changes: dict[str, Any] = dict(rule.get("reply") or {})
        if PROFILE_SHAPE_MARK in system_of(messages):
            analysis: dict[str, Any] = {
                "reasoning": "",
                "language": "de",
                "standalone_question": asked,
                "intent": "lookup",
                "queries_de": [],
                "queries_en": [],
                "degree": None,
                "programme": None,
                "po_version": None,
                "regulation": None,
                "enrolment_term": None,
                "date_range": None,
                "shape": "text",
                "needs_more": False,
            }
            changes = {PROFILE_KEYS.get(key, key): value for key, value in changes.items()}
        else:
            analysis = {
                "standalone_question": asked,
                "intent": "lookup",
                "search_queries": [],
                "language": "de",
                "date_from": None,
                "date_to": None,
                "answer_shape": "text",
            }
        analysis.update(changes)
        return json.dumps(analysis, ensure_ascii=False)
    if kind == "follow_up":
        return json.dumps({**FOLLOW_UP_REPLY, **(rule.get("reply") or {})}, ensure_ascii=False)
    value = rule.get("reply")
    return DEFAULT_REPLIES.get(kind, "OK.") if value is None else str(value)


def pieces(text: str) -> list[str]:
    """The streamed pieces of a reply: one word with the white space after it; they join to the text."""
    return PIECE.findall(text)


def count_tokens(body: Mapping[str, Any]) -> int:
    """/tokenize: the white-space separated words of the prompt, or of every message plus four per message."""
    if "messages" in body:
        return sum(len(text_of(m).split()) + 4 for m in body["messages"])
    return len(str(body.get("prompt", "")).split())


def completion(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def usage_chunk(prompt_tokens: int, completion_tokens: int) -> str:
    """The last event of a streamed completion that asked for its usage (stream_options.include_usage): no choices."""
    body = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return "data: " + json.dumps(body) + "\n\n"


def chunk(delta: Mapping[str, str], finish: str | None = None) -> str:
    """One server-sent event of a streamed completion."""
    body = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
    }
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"
