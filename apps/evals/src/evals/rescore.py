"""Re-score saved probe runs with the current checks.

  evals rescore <run folder>... [--check]

Every input evaluate_turn had at run time is rebuilt from the row: the answer, the sources, the errors, the expectation that
was applied, and meta (the shape frame, and variant_ok: stored, or else read from whether the stored checks
failed "variant"); a client stop (abort_after_s in the profile's probe files) stays a passed set-up step. Without --check each
run gets <out_dir>/<run name>_rescored/results.jsonl (the run itself is untouched; in the container the runs are a read-only
mount); with --check nothing is written and the exit code is 1 when any turn's failed checks changed."""

import argparse
import json
import os
from typing import Any

from evals.checks import evaluate_turn
from evals.gate import abort_turns, conversations, load_config, read_rows
from evals.settings import load_settings


def rescore_rows(rows, aborts):
    """(rows with the current checks, the number of turns whose failed checks changed)."""
    out, changed = [], 0
    for row in rows:
        stored = row.get("checks") or {}
        old = stored.get("failed", [])
        meta = {
            "shape": row.get("shape"),
            "variant_ok": row["variant_ok"] if "variant_ok" in row else "variant" not in old,
        }
        checks: dict[str, Any] = evaluate_turn(
            row.get("answer") or "", row.get("sources") or [], stored.get("errors") or [], row.get("expect") or {}, meta
        )
        if (row.get("id"), row.get("turn")) in aborts:
            checks = dict(checks, failed=[], auto_pass=True, manual_review=False)
        changed += sorted(old) != sorted(checks["failed"])
        out.append(dict(row, checks=checks))
    return out, changed


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evals rescore", description="Re-score saved runs with the current checks")
    ap.add_argument("runs", nargs="+", metavar="DIR")
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 when a turn's failed checks changed")
    args = ap.parse_args(argv)
    settings = load_settings().evals
    aborts = abort_turns(conversations(settings.profile_dir, load_config(settings.profile_dir)))
    changed_any = False
    for run in args.runs:
        src = os.path.normpath(run)
        rows, changed = rescore_rows(read_rows(src), aborts)
        changed_any = changed_any or changed > 0
        if args.check:
            print("%s: rescored %d turns (%d changed)" % (src, len(rows), changed))
            continue
        dst = os.path.join(settings.out_dir, os.path.basename(src) + "_rescored")
        os.makedirs(dst, exist_ok=True)
        with open(os.path.join(dst, "results.jsonl"), "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        print("rescored %d turns (%d changed) -> %s" % (len(rows), changed, dst))
    return 1 if args.check and changed_any else 0
