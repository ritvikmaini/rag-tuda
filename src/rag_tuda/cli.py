"""Command line entry point."""

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from rag_tuda import ask, cards, classes, doctor, runtime, sources, stack, state
from rag_tuda import eval as rag_eval

DEFAULT_URL = "http://127.0.0.1:8700"


def _doctor(args: argparse.Namespace) -> int:
    facts = doctor.gather()
    try:
        cls = classes.pick_class(facts)
        card = cards.named(args.card) if args.card else None
        text = doctor.report_json(facts, cls, card) if args.json else doctor.report(facts, cls, card)
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


def _at_root() -> Path | None:
    """The current folder when it is the repository root (the folder with deploy/compose.yaml)."""
    root = Path.cwd()
    if (root / "deploy" / "compose.yaml").is_file():
        return root
    print("run this from the root of the repository (the folder with deploy/compose.yaml)", file=sys.stderr)
    return None


def _up(args: argparse.Namespace) -> int:
    if (root := _at_root()) is None:
        return 1
    hardware = doctor.gather()
    try:
        card, warning = cards.choose(args.card, args.model, hardware)
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
        current = state.ensure(root)
        up = stack.Args(
            card["name"], args.mode or doctor.default_mode(hardware), args.profile, args.model, args.llm_url
        )
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
    if (root := _at_root()) is None:
        return 1
    rag = root / ".rag"
    if not rag.is_dir():
        print("no .rag/ here: nothing was started from this folder")
        return 0
    steps = stack.plan_down(rag, stack.gather_down(root, rag, doctor.gather()), args.purge)
    if args.purge and not args.yes:
        question = "Delete the crawled content in .rag/crawl and .rag/snapshots? [y/N] "
        if not sys.stdin.isatty() or input(question).strip().lower() not in ("y", "yes"):
            print("--purge deletes the crawled content: confirm with --yes", file=sys.stderr)
            return 1
    try:
        stack.execute(steps, stack.System())
    except stack.StackError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _index_job(plan) -> int:
    """Runs a reindex or rollback plan with what the last `up` recorded in .rag/state.json."""
    if (root := _at_root()) is None:
        return 1
    if not (root / ".rag" / "state.json").is_file():
        print("nothing was started from this folder yet: run `rag up` first", file=sys.stderr)
        return 1
    try:
        current = state.ensure(root)
        data = current.data
        card = cards.named(data["card"]) if data.get("card") else None
        if card is None or not data.get("mode") or not data.get("profile"):
            raise stack.StackError("nothing was started from this folder yet: run `rag up` first")
        up = stack.Args(card["name"], data["mode"], data["profile"], data.get("model"), data.get("llm_url"))
        if up.mode == "compose" and sources.load(root):
            print(
                "note: the compose indexer does not mount the folders of `rag add`; they are indexed by the native stack",
                file=sys.stderr,
            )
        steps = plan(current, stack.gather(root, current, card, doctor.gather(), up), up)
        stack.execute(steps, stack.System())
    except (ValueError, stack.StackError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _reindex(args: argparse.Namespace) -> int:
    return _index_job(lambda *a: stack.plan_reindex(*a, full=args.full, resume=args.resume))


def _rollback(args: argparse.Namespace) -> int:
    return _index_job(stack.plan_rollback)


def _eval(args: argparse.Namespace) -> int:
    """The eval on the recorded card, or on --card after bringing the stack onto it (downloads need a yes)."""
    if (root := _at_root()) is None:
        return 1
    if not (root / ".rag" / "state.json").is_file():
        print("nothing was started from this folder yet: run `rag up` first", file=sys.stderr)
        return 1
    try:
        current = state.ensure(root)
        data = current.data
        name = args.card or data.get("card")
        if not name or not data.get("mode"):
            raise stack.StackError("nothing was started from this folder yet: run `rag up` first")
        same = name == data.get("card")  # the model and server overrides belong to the card they were given for
        up = stack.Args(
            name,
            data["mode"],
            data["profile"],
            data.get("model") if same else None,
            data.get("llm_url") if same else None,
        )
        card = cards.named(name)
        facts = stack.gather(root, current, card, doctor.gather(), up)
        steps = [] if same else stack.plan_up(current, facts, up)
        if not _confirm(steps, args.yes):
            return 1
        steps += rag_eval.plan(
            current, facts, up, retrieval_only=args.retrieval_only, ablation=args.ablation, check_ready=same
        )
        stack.execute(steps, stack.System())
    except (ValueError, stack.StackError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _ask(args: argparse.Namespace) -> int:
    token_file = Path.cwd() / ".rag" / "token"
    if not token_file.is_file():
        print("no .rag/token here: run `rag up` in this folder first", file=sys.stderr)
        return 1
    filters = {k: v for k in ("degree", "programme", "po", "lang") if (v := getattr(args, k)) is not None}
    seen: list[ask.Frame] = []

    def frames():
        for frame in ask.stream(
            args.url, token_file.read_text().strip(), args.question, f"cli-{secrets.token_hex(4)}", filters=filters
        ):
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
        source = sources.register(Path.cwd(), args.target, name=args.name)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"registered {source.name} ({source.target})")
    print("run `rag reindex` to index it")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Local RAG assistant with cited answers")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="check this machine and pick a class")
    d.add_argument("--card", help="also show this card's download size and whether it fits this machine")
    d.add_argument("--json", action="store_true", help="print the result as JSON")
    d.set_defaults(run=_doctor)
    u = sub.add_parser("up", help="bring the stack up (a second run changes nothing)")
    u.add_argument("--card", help="the model card (default: the card of the class this machine fits)")
    u.add_argument("--model", help="a model for the answers: the Ollama model the card's derived model is made from")
    u.add_argument("--llm-url", help="an OpenAI-compatible server you run; the stack does not manage it")
    u.add_argument("--profile", default="example", help="the profile to index and answer from (default %(default)s)")
    u.add_argument(
        "--mode",
        choices=runtime.MODES,
        help="native or compose (default: native on an Apple-silicon Mac, else compose)",
    )
    u.add_argument("--yes", action="store_true", help="confirm the model downloads")
    u.set_defaults(run=_up)
    w = sub.add_parser("down", help="stop the stack (safe to repeat; data stays)")
    w.add_argument(
        "--purge", action="store_true", help="also delete the crawled content in .rag/crawl and .rag/snapshots"
    )
    w.add_argument("--yes", action="store_true", help="confirm --purge")
    w.set_defaults(run=_down)
    r = sub.add_parser(
        "reindex", help="build a new index generation from a re-crawl, swap it in and restart the backend"
    )
    r.add_argument("--full", action="store_true", help="also crawl the full source set (takes hours)")
    r.add_argument("--resume", action="store_true", help="continue an interrupted crawl")
    r.set_defaults(run=_reindex)
    b = sub.add_parser("rollback", help="make the previous index generation live again and restart the backend")
    b.set_defaults(run=_rollback)
    e = sub.add_parser("eval", help="run the eval against the running stack (the smoke unless told otherwise)")
    kind = e.add_mutually_exclusive_group()
    kind.add_argument("--smoke", action="store_true", help="the smoke questions, answers included (default)")
    kind.add_argument("--retrieval-only", action="store_true", help="the gold sources' rank after fusion and reranking")
    e.add_argument("--ablation", action="store_true", help="with --retrieval-only: one row per retrieval switch")
    e.add_argument("--card", help="run on this card, bringing the stack onto it first (default: the running card)")
    e.add_argument("--yes", action="store_true", help="confirm the downloads of a card that is not on this machine yet")
    e.set_defaults(run=_eval)
    a = sub.add_parser("ask", help="ask one question and stream the cited answer")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="print the stream's frames, one JSON object per line")
    a.add_argument("--degree", choices=("bachelor", "master", "promotion"), help="answer for this degree")
    a.add_argument("--programme", help="answer for this programme (a name of roles.yaml)")
    a.add_argument("--po", type=int, help="answer for this version of the examination regulations, e.g. 2023 or 2015")
    a.add_argument("--lang", choices=("de", "en"), help="the language of the page: it sets the texts before the answer")
    a.add_argument("--url", default=DEFAULT_URL, help="where the assistant listens (default %(default)s)")
    a.set_defaults(run=_ask)
    s = sub.add_parser("add", help="register a folder of your own documents (Markdown, text, PDF) as a source")
    s.add_argument("target", help="the folder")
    s.add_argument("--name", help="the source's name (default: from the target)")
    s.set_defaults(run=_add)
    args = parser.parse_args(argv)
    if getattr(args, "ablation", False) and not args.retrieval_only:
        parser.error("--ablation is a retrieval run: add --retrieval-only")
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
