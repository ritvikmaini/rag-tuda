"""Redacted stream transcripts.

The recordings (<plan>/<id>_t<n>.raw, the frames as the harness test tests/integration/tu/test_conversations.py or
`evals run --probes <plan> --record <dir>` kept them, and <id>_t<n>.json, status and parameters) stay local in
data/transcripts/. This writes the committed copies: every frame decoded, the
document text replaced by placeholders of equal length and character class, and written back in the legacy
frame form ("data: " + json.dumps({...}) with Python's defaults, no terminator).

Replaced: chunk contents, docs[].page_content, docs[].metadata's TEXT_KEYS (every string inside them, also in lists and
dicts), citations[].quote, and on every turn after a conversation's first the status frames' question and queries (the
rewritten question and the search queries can carry an earlier answer's terms or figures). A letter becomes x or X by
case, a digit 0, except the digits of a citation marker ([2], [1, 3]) and a line-leading step number; white space,
punctuation, markup and a code fence's info string (```mindmap) stay, so shapes, citations and numbering still parse.
Titles, URLs, types, origins and a first turn's status question and queries (the probe question restated) stay as they are.
leaks() is the check of a committed frame (tests/contracts/test_transcripts.py).

  python -m rag_testing.redact <raw dir> <plan dir> <out dir> [--check-round-trip]"""

import argparse
import json
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from rag_testing.harness import frames, serialise

# evals.checks' FMT_CITE and FMT_STEP (tests/contracts/test_transcripts.py keeps them equal)
FMT_CITE = re.compile(r"\[\d{1,2}(?:\s*,\s*\d{1,2})*\]")
FMT_STEP = re.compile(r"^[ \t]*\d{1,2}[.)][ \t]+\S")
FENCE = re.compile(r"^[ \t]*```[A-Za-z0-9_-]*")
STEP_NUMBER = re.compile(r"[ \t]*(\d{1,2})")
# docs metadata keys whose values quote the document's text
TEXT_KEYS = ("matched_child_text", "context_header", "section_path")
# docs metadata keys that may be long and stay as they are (names and addresses, not text)
LONG_KEYS = (
    "title",
    "filename",
    "source",
    "url",
    "download_url",
    "page_url",
    "public_url",
    "official_url",
    "twin_key",
    "doc_key",
    "title_path",
    "section",
)
# status frame keys that are masked on every turn after a conversation's first
STATUS_TEXT_KEYS = ("question", "queries")
PLANS = ("plain", "formats", "busy")


def kept(text: str) -> set[int]:
    """Positions that stay: citation markers, line-leading step numbers, code fence openings with their info string."""
    keep: set[int] = set()
    for match in FMT_CITE.finditer(text):
        keep.update(range(match.start(), match.end()))
    offset = 0
    for line in text.splitlines(keepends=True):
        if FMT_STEP.match(line):
            number = STEP_NUMBER.match(line)
            if number:
                keep.update(range(offset + number.start(1), offset + number.end(1)))
        fence = FENCE.match(line)
        if fence:
            keep.update(range(offset, offset + fence.end()))
        offset += len(line)
    return keep


def mask(text: str) -> str:
    keep = kept(text)
    out = []
    for i, ch in enumerate(text):
        if i in keep:
            out.append(ch)
        elif ch.isalpha():
            out.append("X" if ch.isupper() else "x")
        elif ch.isdigit():
            out.append("0")
        else:
            out.append(ch)
    return "".join(out)


def mask_all(value: Any) -> Any:
    """Every string in value masked, also inside lists and dicts; any other value unchanged."""
    if isinstance(value, str):
        return mask(value)
    if isinstance(value, list):
        return [mask_all(item) for item in value]
    if isinstance(value, dict):
        return {key: mask_all(item) for key, item in value.items()}
    return value


def strings(value: Any) -> list[str]:
    """Every string inside value, also inside lists and dicts."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in strings(item)]
    return []


def redact_frame(frame: dict[str, Any], follow_up: bool = False) -> dict[str, Any]:
    """The frame with its document text masked; follow_up: a turn after the conversation's first."""
    kind, content = frame.get("type"), frame.get("content")
    if kind == "chunk" and isinstance(content, str):
        return dict(frame, content=mask(content))
    if kind == "docs" and isinstance(content, list):
        docs = []
        for doc in content:
            meta = dict(doc.get("metadata") or {})
            for key in TEXT_KEYS:
                if key in meta:
                    meta[key] = mask_all(meta[key])
            docs.append(dict(doc, page_content=mask(str(doc.get("page_content") or "")), metadata=meta))
        return dict(frame, content=docs)
    if kind == "citations" and isinstance(content, list):
        return dict(frame, content=[dict(c, quote=mask(str(c.get("quote") or ""))) for c in content])
    if kind == "status" and follow_up and isinstance(content, dict):
        return dict(frame, content={k: mask_all(v) if k in STATUS_TEXT_KEYS else v for k, v in content.items()})
    return frame


def leaks(frame: dict[str, Any], follow_up: bool = False) -> list[str]:
    """What a committed frame must not carry: a text the mask would still change, or a docs metadata value of more than 80
    characters (measured as JSON when it is no string) under a key that is neither redacted nor listed in LONG_KEYS."""
    kind, content = frame.get("type"), frame.get("content")
    texts: list[str] = []
    problems: list[str] = []
    if kind == "chunk":
        texts += strings(content)
    elif kind == "citations" and isinstance(content, list):
        texts += [text for entry in content if isinstance(entry, dict) for text in strings(entry.get("quote"))]
    elif kind == "status" and follow_up and isinstance(content, dict):
        texts += [text for key in STATUS_TEXT_KEYS for text in strings(content.get(key))]
    elif kind == "docs" and isinstance(content, list):
        for doc in content:
            texts += strings(doc.get("page_content"))
            for key, value in (doc.get("metadata") or {}).items():
                if key in TEXT_KEYS:
                    texts += strings(value)
                elif key not in LONG_KEYS:
                    size = len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))
                    if size > 80:
                        problems.append("docs metadata %r carries %d characters: redact it or list it" % (key, size))
    problems += ["not masked: %r" % text[:60] for text in texts if mask(text) != text]
    return problems


def turns(plan_dir: Path) -> Iterator[tuple[str, str, int, dict[str, Any]]]:
    """(plan, conversation id, turn number, turn) of every turn of the three plan files."""
    for plan in PLANS:
        spec = json.loads((plan_dir / ("plan_%s.json" % plan)).read_text(encoding="utf-8"))
        for conv in spec["conversations"]:
            for n, turn in enumerate(conv["turns"], 1):
                yield plan, conv["id"], n, turn


def chosen_busy(raw_dir: Path) -> dict[str, str]:
    """{stem: path} for the busy plan: the first reply that was busy and the first that was answered."""
    picked: dict[str, str] = {}
    for raw in sorted((raw_dir / "busy").glob("*.raw")):
        types = [f.get("type") for f in frames(raw.read_text(encoding="utf-8"))[0]]
        path = "busy" if types[:1] == ["error"] else "lookup"
        picked.setdefault(path, raw.stem)
    return {stem: path for path, stem in picked.items()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rag_testing.redact", description="Write redacted transcripts")
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("plan_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument(
        "--check-round-trip", action="store_true", help="only check that re-serialising gives the raw bytes"
    )
    args = ap.parse_args(argv)
    busy = chosen_busy(args.raw_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written, mismatched = 0, 0
    for plan, cid, n, turn in turns(args.plan_dir):
        stem = "%s_t%d" % (cid, n)
        if plan == "busy" and stem not in busy:
            continue
        raw_path = args.raw_dir / plan / (stem + ".raw")
        raw = raw_path.read_text(encoding="utf-8")
        items, end = frames(raw)
        if args.check_round_trip:
            if serialise(items) != raw[:end]:
                mismatched += 1
                print("ROUND TRIP DIFFERS: %s" % raw_path)
            elif end < len(raw):
                print(
                    "%s: %d bytes after the last complete frame left out (a stopped request)"
                    % (raw_path, len(raw) - end)
                )
            continue
        sidecar = json.loads((args.raw_dir / plan / (stem + ".json")).read_text(encoding="utf-8"))
        params = {k: v for k, v in (sidecar.get("params") or {}).items() if k != "conversation_id"}
        path = busy.get(stem) if plan == "busy" else turn.get("path")
        name = "%s_%s" % (plan, stem)
        (args.out_dir / (name + ".sse")).write_text(
            serialise([redact_frame(f, follow_up=n > 1) for f in items]), encoding="utf-8"
        )
        meta = {"plan": plan, "id": cid, "turn": n, "path": path, "status": sidecar.get("status"), "params": params}
        (args.out_dir / (name + ".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", "utf-8")
        written += 1
    print("round trip: %d differ" % mismatched if args.check_round_trip else "%d transcripts written" % written)
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
