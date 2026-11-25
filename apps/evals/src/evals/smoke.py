"""The deploy smoke: the fixed probes of gate.yaml, structure only, so a correct deploy never
fails on answer variance. One request per probe turn of gate.yaml (the example profile: 6 requests, about 2 minutes). Exit 0 = every check passed, 10 = at least one failed, 2 = no token;
any other code = the smoke could not run (the deploy keeps that apart from a failed smoke). Whether every service is ready is
the deploy's part: the container cannot see Docker."""

import argparse
import json
import os
import sys

import httpx

from evals.gate import conversations, load_config
from evals.runner import ask, run_conversation, run_stamp
from evals.settings import load_settings

STEPS = ("analysing", "understood", "searching", "found", "writing")  # a lookup's status steps, in order
OPTIONAL_STEPS = (
    "searching_more",
)  # the iterative round's step: it appears, after "searching", only for a question that needs more
NEW_TYPES = ("meta", "done")  # stream format 1: the smoke judges the frames an old reader sees


def structural(kind, last, row, steps, max_ttft):
    """The failed structural checks of one turn; the probe kind's own checks apply to its last turn."""
    problems = []
    if row["checks"]["errors"]:  # error frames, HTTP status other than 200, runner errors
        problems.append("error")
    types = row["event_types"]
    answered = "chunk" in types
    if answered and (row["first_token_s"] is None or row["first_token_s"] >= max_ttft):
        problems.append("ttft")
    if last and kind == "lookup":
        if [step for step in steps if step not in OPTIONAL_STEPS] != list(STEPS):
            problems.append("status steps")
        problems += ["no " + frame for frame in ("docs", "chunk", "citations") if frame not in types]
    elif last and kind == "direct":
        if "docs" in types:
            problems.append("docs")
        if not answered:
            problems.append("no chunk")
    elif last and kind == "too_long":
        seen = [t for t in types if t not in NEW_TYPES]
        if seen != ["chunk"]:
            problems.append("frames " + ",".join(str(t) for t in seen))
    elif last and not answered:  # edit: the edited turn is answered
        problems.append("no chunk")
    return problems


def run_smoke(settings, cfg, index, client, user_label, max_ttft, stamp):
    token = settings.token.get_secret_value()
    rows = []
    for probe in cfg.smoke.probes:
        conv = index[probe.id][1]
        played = {"id": probe.id, "set": conv.get("set"), "turns": [conv["turns"][n - 1] for n in probe.turns]}
        steps = []

        def ask_fn(question, user, opts, steps=steps):
            result = ask(client, settings.assistant_url, token, question, user, settings.timeout_s, **opts)
            steps.append((opts.get("meta") or {}).get("steps", []))
            return result

        played_rows = run_conversation(played, ask_fn, settings.min_interval_s, user_label, stamp, lambda row: None)
        for n, row in enumerate(played_rows):
            row["smoke"] = structural(probe.kind, n == len(played_rows) - 1, row, steps[n], max_ttft)
            print(
                "%-6s t%d %-8s %-4s %5.1fs %s"
                % (
                    probe.id,
                    probe.turns[n],
                    probe.kind,
                    "FAIL" if row["smoke"] else "ok",
                    row["total_s"],
                    ", ".join(row["smoke"]),
                ),
                flush=True,
            )
            rows.append(row)
    return rows


def main(argv=None, client_factory=httpx.Client):
    ap = argparse.ArgumentParser(
        prog="evals smoke", description="The deploy smoke: the fixed probes of gate.yaml, structure only"
    )
    ap.add_argument("--label", default="smoke")
    ap.add_argument("--max-ttft", type=float, default=None, help="seconds; default: gate.yaml smoke.max_ttft_s")
    args = ap.parse_args(argv)
    settings = load_settings().evals
    if settings.token is None:
        print("no backend token: mount the compose secret backend_token or set RAG__EVALS__TOKEN", file=sys.stderr)
        return 2
    cfg = load_config(settings.profile_dir)
    index = conversations(settings.profile_dir, cfg)
    max_ttft = cfg.smoke.max_ttft_s if args.max_ttft is None else args.max_ttft
    stamp = run_stamp()
    run_dir = os.path.join(settings.out_dir, "%s-smoke_%s" % (args.label, stamp))
    os.makedirs(run_dir)
    with client_factory() as client:
        rows = run_smoke(settings, cfg, index, client, args.label + "-smoke", max_ttft, stamp)
    with open(os.path.join(run_dir, "results.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    failed = [r for r in rows if r["smoke"]]
    print(
        "RESULT %s: %d of %d turns failed a check (max TTFT %g s)"
        % ("PASS" if not failed else "FAIL", len(failed), len(rows), max_ttft)
    )
    print("Results: " + run_dir)
    return 10 if failed else 0
