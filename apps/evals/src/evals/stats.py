"""Retrieval agreement, flips and TTFT of probe-run pairs.

usage:
  evals stats --pair <set> RUN_1 RUN_2 [--pair <set> RUN_1 RUN_2 ...] --out data/eval_runs/<label>-stats.json

RUN_1 and RUN_2 are run folders (results.jsonl) recorded back to back on one index generation by the probe runner
(every source carries its section identity). Per pair and pooled over all pairs:
- retrieval agreement of the first turns: per question, the share of the larger top-14 set of section identities found in
  both runs (order-insensitive); questions where neither run retrieved anything are counted, not scored;
- the questions found in only one run (an incomplete run: its pair must be recorded again);
- the turns whose pass/fail differs between the runs (candidates for the known-flaky list);
- auto-pass counts and error events per run;
- the TTFT medians of the pooled first turns of both runs and their difference (the spread behind the latency margin in gate.yaml)."""

import argparse
import json
import statistics
from typing import Any


def load_rows(run_dir):
    with open(run_dir.rstrip("/") + "/results.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    sources = [s for r in rows for s in (r.get("sources") or [])]
    if sources and not any(s.get("section") for s in sources):
        raise SystemExit("%s: the sources carry no section identity (recorded by an older runner)" % run_dir)
    return rows


def first_turns(rows):
    return {r["id"]: r for r in rows if r["turn"] == 1}


def sections(row):
    return {s["section"] for s in row.get("sources") or [] if s.get("section")}


def overlap(a, b):
    """The share of the larger set found in both; None when neither run retrieved anything."""
    if not a and not b:
        return None
    return len(a & b) / max(len(a), len(b))


def ttfts(firsts):
    return [r["first_token_s"] for r in firsts.values() if r.get("first_token_s") is not None]


def compare_pair(rows_1, rows_2):
    first_1, first_2 = first_turns(rows_1), first_turns(rows_2)
    scores, differing, no_retrieval = [], [], []
    for cid in sorted(set(first_1) & set(first_2)):
        score = overlap(sections(first_1[cid]), sections(first_2[cid]))
        if score is None:
            no_retrieval.append(cid)
            continue
        scores.append(score)
        if score < 1.0:
            differing.append({"id": cid, "overlap": round(score, 4)})
    passed_1 = {(r["id"], r["turn"]): bool(r["checks"]["auto_pass"]) for r in rows_1}
    passed_2 = {(r["id"], r["turn"]): bool(r["checks"]["auto_pass"]) for r in rows_2}
    flips = ["%s t%d" % key for key in sorted(set(passed_1) & set(passed_2)) if passed_1[key] != passed_2[key]]
    missing = ["%s (run %d only)" % (cid, 1 if cid in first_1 else 2) for cid in sorted(set(first_1) ^ set(first_2))]
    return {
        "scores": scores,
        "differing": differing,
        "no_retrieval": no_retrieval,
        "flips": flips,
        "missing": missing,
        "pass": [sum(passed_1.values()), sum(passed_2.values())],
        "turns": [len(rows_1), len(rows_2)],
        "errors": [sum(bool(r["checks"]["errors"]) for r in rows) for rows in (rows_1, rows_2)],
        "ttft": [ttfts(first_1), ttfts(first_2)],
    }


def mean(values):
    return sum(values) / len(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def summarise(pairs):
    """pairs: {set name: compare_pair(...)}; returns the statistics written to --out."""
    out: dict[str, Any] = {"sets": {}}
    scores, ttft_1, ttft_2 = [], [], []
    for name, p in pairs.items():
        scores += p["scores"]
        ttft_1 += p["ttft"][0]
        ttft_2 += p["ttft"][1]
        out["sets"][name] = {
            "agreement": mean(p["scores"]),
            "scored": len(p["scores"]),
            "identical": sum(s == 1.0 for s in p["scores"]),
            "no_retrieval": p["no_retrieval"],
            "differing": p["differing"],
            "flips": p["flips"],
            "missing": p["missing"],
            "pass": p["pass"],
            "turns": p["turns"],
            "errors": p["errors"],
            "ttft_median": [median(t) for t in p["ttft"]],
        }
    out["noise_floor"] = mean(scores)
    out["noise_floor_meets_97"] = out["noise_floor"] is not None and out["noise_floor"] >= 0.97
    out["ttft_median"] = [median(ttft_1), median(ttft_2)]
    both = None not in out["ttft_median"]
    # `both` excludes None; the type checker cannot see through the subscripts
    out["ttft_spread"] = abs(out["ttft_median"][0] - out["ttft_median"][1]) if both else None  # pyright: ignore[reportOptionalOperand]
    out["ttft_margin"] = max(0.3, out["ttft_spread"]) if both else None  # pyright: ignore[reportArgumentType]
    return out


def fmt(value, pattern="%.4f"):
    return pattern % value if value is not None else "-"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="evals stats", description="Noise floor and TTFT spread of back-to-back probe runs"
    )
    ap.add_argument("--pair", nargs=3, action="append", required=True, metavar=("SET", "RUN_1", "RUN_2"))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    stats = summarise({name: compare_pair(load_rows(a), load_rows(b)) for name, a, b in args.pair})
    for name, s in stats["sets"].items():
        print(
            "%-8s agreement %s over %d questions (%d identical, %d without retrieval); pass %d/%d and %d/%d; "
            "errors %d and %d; flips: %s; missing: %s"
            % (
                name,
                fmt(s["agreement"]),
                s["scored"],
                s["identical"],
                len(s["no_retrieval"]),
                s["pass"][0],
                s["turns"][0],
                s["pass"][1],
                s["turns"][1],
                s["errors"][0],
                s["errors"][1],
                ", ".join(s["flips"]) or "none",
                ", ".join(s["missing"]) or "none",
            )
        )
    print(
        "noise floor %s (meets 97 %%: %s); TTFT median %s / %s s, spread %s s, latency margin %s s"
        % (
            fmt(stats["noise_floor"]),
            stats["noise_floor_meets_97"],
            fmt(stats["ttft_median"][0], "%.2f"),
            fmt(stats["ttft_median"][1], "%.2f"),
            fmt(stats["ttft_spread"], "%.2f"),
            fmt(stats["ttft_margin"], "%.2f"),
        )
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
    return 0
