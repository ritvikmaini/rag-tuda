"""Command line entry point."""

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from rag_tuda import ask, doctor, sources, tiers

DEFAULT_URL = "http://127.0.0.1:8700"


def _doctor(args: argparse.Namespace) -> int:
    facts = doctor.gather()
    tier = tiers.pick_tier(facts)
    print(doctor.report_json(facts, tier) if args.json else doctor.report(facts, tier))
    return 1 if doctor.problems(facts) else 0


def _ask(args: argparse.Namespace) -> int:
    token_file = Path.cwd() / ".rag" / "token"
    if not token_file.is_file():
        print("no .rag/token here: run `rag up` in this folder first", file=sys.stderr)
        return 1
    seen: list[ask.Frame] = []

    def frames():
        for frame in ask.stream(args.url, token_file.read_text().strip(), args.question, f"cli-{secrets.token_hex(4)}"):
            seen.append(frame)
            yield frame

    try:
        if args.json:
            for frame in frames():
                print(json.dumps(frame, ensure_ascii=False), flush=True)
        else:
            color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
            for piece in ask.pieces(frames(), color):
                print(piece, end="", flush=True)
    except ask.AskError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 1 if ask.failed(seen) else 0


def _add(args: argparse.Namespace) -> int:
    try:
        source = sources.register(Path.cwd(), args.target, name=args.name, only=args.only, topic=args.topic)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"registered {source.name} ({source.connector}: {source.target})")
    print("run `rag reindex` to index it")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Local RAG assistant with cited answers")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="check this machine and pick a tier")
    d.add_argument("--json", action="store_true", help="print the result as JSON")
    d.set_defaults(run=_doctor)
    a = sub.add_parser("ask", help="ask one question and stream the cited answer")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="print the stream's frames, one JSON object per line")
    a.add_argument("--url", default=DEFAULT_URL, help="where the assistant listens (default %(default)s)")
    a.set_defaults(run=_ask)
    s = sub.add_parser("add", help="register a folder, a web site or a CSV file as a source")
    s.add_argument("target", help="a folder, an http(s) URL or a .csv file")
    s.add_argument("--name", help="the source's name (default: from the target)")
    s.add_argument("--only", action="store_true", help="index only your sources, not the default corpus")
    s.add_argument("--topic", help="what the assistant says it is about")
    s.set_defaults(run=_add)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
