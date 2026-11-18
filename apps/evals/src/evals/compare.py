"""Compare probe runs turn by turn.

  evals compare <run folder>...

Per run and inner group: answers, passes without emoji, expected-fact failures, sources where none belong, "du", leaks,
error events, empty answers, latency (mean, first-token median on turns that retrieved documents); then the checks fixed and
newly failed from the first to the last run on the turns both share (emoji ignored), and intent-label echoes in answers."""

import argparse
import json
import re
import statistics
from collections import defaultdict

FACT = {"contains_any", "contains_none", "digits_any", "source_url_any", "max_words"}
LABEL = re.compile(r"^\W*(about_bot|out_of_scope|conversation|urgent|lookup|situation|find_document)\b")


def load(run_dir):
    with open(run_dir.rstrip("/") + "/results.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {(r["id"], r["turn"]): r for r in rows}


def failed(row):
    return {x.split(":")[0] for x in row["checks"]["failed"]}


def summary(rows):
    by_set = defaultdict(list)
    for r in rows.values():
        by_set[r["set"]].append(r)
        by_set["total"].append(r)
    out = {}
    for s, rs in by_set.items():
        out[s] = dict(
            n=len(rs),
            pass_no_emoji=sum(not (failed(r) - {"emoji"}) for r in rs),
            fact_fail=sum(bool(failed(r) & FACT) for r in rs),
            no_sources=sum("no_sources" in failed(r) for r in rs),
            du=sum("informal_address" in failed(r) for r in rs),
            leak=sum("leak" in failed(r) for r in rs),
            error=sum(bool(r["checks"]["errors"]) for r in rs),
            empty=sum("empty_answer" in failed(r) for r in rs),
        )
    t = [r["total_s"] for r in rows.values() if r["total_s"] is not None]
    ft = [r["first_token_s"] for r in rows.values() if r["first_token_s"] and r["checks"]["n_sources"]]
    out["latency"] = "mean %.1f s, max %.1f s, first token median (retrieval turns) %.1f s" % (
        statistics.mean(t),
        max(t),
        statistics.median(ft) if ft else 0.0,
    )
    out["label_echoes"] = sum(bool(LABEL.search(r["answer"])) for r in rows.values())
    return out


def changes(first, last):
    """(fixed, new failures) from the first to the last run on the turns both share, emoji ignored."""
    common = [k for k in last if k in first]
    fixed = [
        "%st%d:%s" % (k[0], k[1], ",".join(sorted((failed(first[k]) - failed(last[k])) - {"emoji"})))
        for k in common
        if (failed(first[k]) - failed(last[k])) - {"emoji"}
    ]
    new = [
        "%st%d:%s" % (k[0], k[1], ",".join(sorted((failed(last[k]) - failed(first[k])) - {"emoji"})))
        for k in common
        if (failed(last[k]) - failed(first[k])) - {"emoji"}
    ]
    return fixed, new


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evals compare", description="Compare probe runs turn by turn")
    ap.add_argument("runs", nargs="+", metavar="DIR")
    args = ap.parse_args(argv)
    runs = [(path, load(path)) for path in args.runs]
    for path, rows in runs:
        print("==", path)
        for key, value in summary(rows).items():
            print("  ", key, value)
    if len(runs) >= 2:
        fixed, new = changes(runs[0][1], runs[-1][1])
        print("FIXED (first -> last):", fixed)
        print("NEW FAILURES (first -> last):", new)
    return 0
