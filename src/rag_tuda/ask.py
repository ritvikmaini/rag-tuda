"""`rag ask`: reads the answer stream the page uses (server-sent events, one JSON frame per `data:` line) and shows it in the
terminal: thinking steps, the answer, numbered sources.

Frames handled: meta, status, shape, docs, chunk, citations, variant, error, done. The terminal shows status (the steps),
chunk (the answer), citations (the sources), error and a done whose outcome is not `completed`; the others are read and
skipped."""

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from typing import Any

Frame = dict[str, Any]
DIM, RESET = "\x1b[2m", "\x1b[0m"
TIMEOUT_S = 300


class AskError(Exception):
    """The stream could not be opened; the message says what to do."""


def parse(lines: Iterable[str]) -> Iterator[Frame]:
    """Frames from SSE lines. Comment lines (`: progress`) and fields other than `data` are skipped, consecutive `data` lines
    form one event, and an event that is not a JSON object with a string `type` is dropped."""
    data: list[str] = []

    def event() -> Frame | None:
        text, data[:] = "\n".join(data), []
        try:
            frame = json.loads(text)
        except ValueError:
            return None
        return frame if isinstance(frame, dict) and isinstance(frame.get("type"), str) else None

    for line in lines:
        line = line.rstrip("\r\n")
        if not line:
            if data and (frame := event()):
                yield frame
            data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data and (frame := event()):
        yield frame


def _step(content: dict[str, Any]) -> str | None:
    step = content.get("step")
    if not step:
        return None
    if step == "understood" and content.get("question"):
        return f"understood: {content['question']}"
    if step == "searching" and content.get("queries"):
        return "searching: " + "; ".join(map(str, content["queries"]))
    if step == "found":
        return f"found: {content.get('documents', 0)} documents, {content.get('passages', 0)} passages"
    return str(step)


def _source(c: dict[str, Any]) -> str:
    return f"[{c.get('n', '?')}] " + " - ".join(x for x in (c.get("title"), c.get("url")) if x)


def pieces(frames: Iterable[Frame], color: bool = False) -> Iterator[str]:
    """The terminal text in the order it can be shown: steps and answer as they arrive, errors and sources at the end."""
    answered = False
    ended_with_newline = True
    steps = False
    sources: list[dict[str, Any]] = []
    tail: list[str] = []
    error = False
    for f in frames:
        kind, content = f.get("type"), f.get("content")
        if kind == "status" and isinstance(content, dict) and (text := _step(content)):
            line = f"- {text}"
            yield (f"{DIM}{line}{RESET}" if color else line) + "\n"
            steps = True
        elif kind == "chunk" and isinstance(content, str) and content:
            if not answered and steps:
                yield "\n"
            answered = True
            ended_with_newline = content.endswith("\n")
            yield content
        elif kind == "citations" and isinstance(content, list):
            sources = [c for c in content if isinstance(c, dict)]
        elif kind == "error":
            error = True
            tail.append(f"Error ({f.get('code', 'error')}): {content}")
        elif (
            kind == "done"
            and isinstance(content, dict)
            and content.get("outcome") not in (None, "completed")
            and not error
        ):
            tail.append(f"Outcome: {content['outcome']}")
    if answered and not ended_with_newline:
        yield "\n"
    if sources:
        yield ("\n" if answered or steps else "") + "".join(
            _source(c) + "\n" for c in sorted(sources, key=lambda c: c.get("n", 0))
        )
    if tail:
        yield "".join(t + "\n" for t in tail)


def render(frames: Iterable[Frame], color: bool = False) -> str:
    return "".join(pieces(frames, color))


def failed(frames: Iterable[Frame]) -> bool:
    """True when the stream carried an error frame or ended with outcome `failed`."""
    return any(
        f.get("type") == "error" or (f.get("type") == "done" and (f.get("content") or {}).get("outcome") == "failed")
        for f in frames
    )


def build_request(
    base_url: str, token: str, question: str, conversation_id: str, formats: str | None = None
) -> urllib.request.Request:
    query = {"question": question, "conversation_id": conversation_id}
    if formats:
        query["formats"] = formats
    return urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat?{urllib.parse.urlencode(query)}",
        headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
    )


def stream(
    base_url: str, token: str, question: str, conversation_id: str, formats: str | None = None
) -> Iterator[Frame]:
    req = build_request(base_url, token, question, conversation_id, formats)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            yield from parse(line.decode("utf-8") for line in resp)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise AskError(
                "the token was refused: use the token in .rag/token (open the link `rag up` printed)"
            ) from exc
        raise AskError(f"the assistant answered HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise AskError(f"cannot reach {base_url}: is the stack running? Start it with `rag up`") from exc
