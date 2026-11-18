"""Command line entry point."""

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from rag_tuda import ask, cards, classes, doctor, runtime, sources, stack, state

DEFAULT_URL = "http://127.0.0.1:8700"


def _doctor(args: argparse.Namespace) -> int:
    facts = doctor.gather()
    try:
        cls = classes.pick_class(facts)
        text = doctor.report_json(facts, cls) if args.json else doctor.report(facts, cls)
    except ValueError as exc:  # the model cards are missing or unreadable
        print(str(exc), file=sys.stderr)
        return 1
    print(text)
    return 1 if doctor.problems(facts) else 0


def _confirm(steps: list[stack.Step], yes: bool) -> bool:
    """The downloads of a plan are listed once and need a yes (--yes, or an answer at the terminal)."""
    text = stack.confirm_text(steps)
    if text is None:
        return True
    print(text)
    if yes:
        return True
    if not sys.stdin.isatty():
        print("downloads need your yes: run `rag up --yes`", file=sys.stderr)
        return False
    return input("Continue? [y/N] ").strip().lower() in ("y", "yes")


def _up(args: argparse.Namespace) -> int:
    root = Path.cwd()
    if not (root / "deploy" / "compose.yaml").is_file():
        print("run `rag up` from the root of the repository (the folder with deploy/compose.yaml)", file=sys.stderr)
        return 1
    hardware = doctor.gather()
    try:
        card, warning = cards.choose(args.card, args.model, hardware)
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
        current = state.ensure(root)
        up = stack.Args(card["name"], args.mode, args.profile, args.model, args.llm_url)
        steps = stack.plan_up(current, stack.gather(root, current, card, hardware, up), up)
        if not _confirm(steps, args.yes):
            return 1
        stack.execute(steps, stack.System())
    except (ValueError, stack.StackError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"ready: {state.page_url(current)}")
    return 0


def _down(args: argparse.Namespace) -> int:
    root = Path.cwd()
    current = state.ensure(root)
    steps = stack.plan_down(current, stack.gather_down(root, current, doctor.gather()), args.purge)
    if args.purge and not args.yes:
        if not sys.stdin.isatty() or input(
            "Delete the crawled content in .rag/crawl and .rag/snapshots? [y/N] "
        ).strip().lower() not in ("y", "yes"):
            print("--purge deletes the crawled content: confirm with --yes", file=sys.stderr)
            return 1
    try:
        stack.execute(steps, stack.System())
    except stack.StackError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


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
    d = sub.add_parser("doctor", help="check this machine and pick a class")
    d.add_argument("--json", action="store_true", help="print the result as JSON")
    d.set_defaults(run=_doctor)
    u = sub.add_parser("up", help="bring the stack up (a second run changes nothing)")
    u.add_argument("--card", help="the model card (default: the card of the class this machine fits)")
    u.add_argument("--model", help="a model for the answers: the Ollama model the card's derived model is made from")
    u.add_argument("--llm-url", help="an OpenAI-compatible server you run; the stack does not manage it")
    u.add_argument("--profile", default="example", help="the profile to index and answer from (default %(default)s)")
    u.add_argument("--mode", choices=runtime.MODES, default="native", help="native (Mac) or compose (Linux, WSL2)")
    u.add_argument("--yes", action="store_true", help="confirm the model downloads")
    u.set_defaults(run=_up)
    w = sub.add_parser("down", help="stop the stack (safe to repeat; data stays)")
    w.add_argument(
        "--purge", action="store_true", help="also delete the crawled content in .rag/crawl and .rag/snapshots"
    )
    w.add_argument("--yes", action="store_true", help="confirm --purge")
    w.set_defaults(run=_down)
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
