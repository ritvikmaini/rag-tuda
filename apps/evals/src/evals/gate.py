"""gate.yaml and the structural gate of a round.

  evals gate <cand-label> --against <ref-label> [--sets core|<set>,...] [--reruns DIR...]
  evals gate --cand DIR... --ref DIR... [--reruns DIR...]

A label stands for the newest <label>-<set>_<stamp> folder per declared set under runs_dir (--sets, default `core` of gate.yaml:
the gate scores only what it declares), its re-runs for the <label>-rerun<k>-<set>_<stamp> folders newer than the candidate run
of their set. A folder's set comes from its name when that names a set whose ids cover the rows (a panel is a view on a probe
file: gate.yaml `ids`), else from its conversation ids (unique across the probe files). A pending turn is re-run with `evals run
<set> --replay <candidate folder> --turns <id>:t<n>,...` (from the earliest listed turn to the last listed
one, on the stored earlier turns; no spaces, so the list stays one argument), or with `--ids` when its conversation cannot replay. Exit 0
pass; 1 fail; 2 re-runs needed (the commands are printed); 3 pass once each differing question listed is explained
(retrieval agreement between gate.yaml's hard floor and the noise floor); 4 void, ahead of every other result: the rows
span index generations, or a side (the reference; the candidate with its re-runs) holds a prompt in two versions - rows without
`generation` are unrecorded and never void; a usage error exits 64 and a run that could not be read 70 (evals.cli). Questions
gate.yaml names retrieval-volatile are reported and never counted. Latency hygiene stays the round's job: warmed
services, 40 minutes after a restart, no build beside a run. The report goes to out_dir as <cand-label>-gate_<stamp>.json
(gate_<stamp>.json with folders).
"""

import argparse
import glob
import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from evals import stats
from evals.settings import load_settings

STAMP = r"\d{8}-\d{6}"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Baseline(_Strict):
    noise_floor: float
    per_set: dict[str, float]
    ttft_margin_s: float


class Retrieval(_Strict):
    sets: list[str]
    top_k: int
    floor: float  # the hard floor; from it up to baseline.noise_floor the differing questions are explained, not failed
    volatile: list[str] = []  # conversation ids whose first-turn retrieval is reported, never counted


class Latency(_Strict):
    sets: list[str]


class Answers(_Strict):
    rerun_runs: int
    regression_at: int
    hang_s: float


class ProbeSet(_Strict):
    file: str
    flaky: list[str] = []
    formats: Literal["v1"] | None = None
    ids: list[
        str
    ] = []  # a panel: only these conversations of the file; it is a view, so the ids stay out of conversations()


class SmokeProbe(_Strict):
    set: str
    id: str
    turns: list[int]
    kind: Literal["lookup", "direct", "too_long", "edit"]


class Smoke(_Strict):
    max_ttft_s: float
    probes: list[SmokeProbe]


class GateConfig(_Strict):
    baseline: Baseline
    retrieval: Retrieval
    latency: Latency
    answers: Answers
    sets: dict[str, ProbeSet]
    core: list[str]  # the sets `evals gate` scores by default
    smoke: Smoke

    @model_validator(mode="after")
    def _core_is_declared(self):
        if unknown := [s for s in self.core if s not in self.sets]:
            raise ValueError("core names sets that gate.yaml lacks: " + ", ".join(unknown))
        return self


def load_config(profile_dir):
    with open(os.path.join(profile_dir, "gate.yaml"), encoding="utf-8") as fh:
        return GateConfig.model_validate(yaml.safe_load(fh))


def conversations(profile_dir, cfg):
    """conversation id -> (set, conversation) over the probe files of gate.yaml; an id in two files is an error."""
    index = {}
    for name, probe_set in cfg.sets.items():
        if probe_set.ids:
            continue  # a panel
        with open(os.path.join(profile_dir, probe_set.file), encoding="utf-8") as fh:
            for conv in json.load(fh)["conversations"]:
                if conv["id"] in index:
                    raise ValueError(
                        "conversation %s is in the sets %s and %s" % (conv["id"], index[conv["id"]][0], name)
                    )
                index[conv["id"]] = (name, conv)
    return index


def set_ids(profile_dir, cfg, name):
    """The conversation ids of a set: a panel's ids, else every id of its file."""
    if cfg.sets[name].ids:
        return list(cfg.sets[name].ids)
    with open(os.path.join(profile_dir, cfg.sets[name].file), encoding="utf-8") as fh:
        return [c["id"] for c in json.load(fh)["conversations"]]


def abort_turns(index):
    """(conversation id, turn) of every client stop: a set-up step, scored as passed."""
    return {
        (cid, n)
        for cid, (_, conv) in index.items()
        for n, turn in enumerate(conv["turns"], 1)
        if turn.get("abort_after_s") is not None
    }


def read_rows(run_dir):
    with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def set_of(rows, index, run_dir, members=None):
    """The run's set. members (set -> ids) lets the folder's name win: <...>-<set>_<stamp> naming a set whose ids cover the rows,
    the longest such name (synthetic_sample before synthetic); otherwise the rows' ids decide."""
    members = members or {}
    base = re.sub(r"_" + STAMP + r"$", "", os.path.basename(run_dir.rstrip("/")))
    for name in sorted(members, key=len, reverse=True):
        if base.endswith("-" + name) and {r["id"] for r in rows} <= set(members[name]):
            return name
    sets = {index[r["id"]][0] for r in rows if r["id"] in index}
    unknown = sorted({r["id"] for r in rows if r["id"] not in index})
    if unknown or len(sets) != 1:
        raise ValueError("%s: not one probe set (sets %s, unknown ids %s)" % (run_dir, sorted(sets), unknown))
    return sets.pop()


def by_set(folders, index, members=None):
    found = {}
    for folder in folders:
        name = set_of(read_rows(folder), index, folder, members)
        if name in found:
            raise ValueError("two runs of the set %s: %s and %s" % (name, found[name], folder))
        found[name] = folder
    return found


def _folders(out_dir, pattern):
    for path in sorted(glob.glob(os.path.join(glob.escape(out_dir), "*"))):
        match = re.fullmatch(pattern, os.path.basename(path))
        if match and os.path.isdir(path):
            yield match, path


def newest_runs(out_dir, label, set_names):
    """set -> the newest <label>-<set>_<stamp> folder; a re-run folder never matches (its middle is rerun<k>-<set>)."""
    found = {}
    for match, path in _folders(out_dir, re.escape(label) + r"-(.+)_(" + STAMP + ")"):
        if match.group(1) in set_names:
            found[match.group(1)] = path  # sorted: the newest stamp comes last
    return found


def rerun_runs(out_dir, label, cand_dirs):
    """set -> the <label>-rerun<k>-<set>_<stamp> folders newer than that set's candidate run, in the order of k."""
    found = {}
    for match, path in _folders(out_dir, re.escape(label) + r"-rerun(\d+)-(.+)_(" + STAMP + ")"):
        name = match.group(2)
        if name in cand_dirs and match.group(3) > cand_dirs[name].rsplit("_", 1)[1]:
            found.setdefault(name, []).append((int(match.group(1)), path))
    return {name: [path for _, path in sorted(paths)] for name, paths in found.items()}


def replay_problem(conv, rows):
    """Why a candidate conversation cannot replay single turns ("" when it can): an abort or replace turn, or rows recorded
    before the runner stored `user` and `turn_id` (a variant or select turn has no id of its own)."""
    if any(t.get("abort_after_s") is not None or t.get("replace_turn") for t in conv["turns"]):
        return "it has an abort or replace turn"
    by_turn = {r["turn"]: r for r in rows}
    for n, turn in enumerate(conv["turns"], 1):
        row = by_turn.get(n)
        if row is None or not row.get("user"):
            return "t%d has no stored row with a user" % n
        if "variant" not in turn and "select" not in turn and not row.get("turn_id"):
            return "t%d has no stored turn_id" % n
    return ""


def _passes(rows):
    return {(r["id"], r["turn"]): bool(r["checks"]["auto_pass"]) for r in rows}


def _verdict(ok):
    return "PASS" if ok else "FAIL"


def _label(version):
    return "v%s" % version


def provenance(cand, ref, reruns):
    """Whether the rows of a pair can be compared: one index generation, and per side (reference; candidate with its re-runs)
    one version of each prompt (a prompt changes between the two runs on purpose; a mix inside one side is a change
    during a run). Rows without `generation` are unrecorded and never void; rows without `prompts`
    (early exits) take no part, and neither does a `null` version (rows recorded by an older backend may carry one: it names
    no text). cand, ref: set -> rows; reruns: set -> [rows, ...]."""
    sides = {
        "reference": [(s, r) for s, rows in ref.items() for r in rows],
        "candidate": [(s, r) for s, rows in cand.items() for r in rows]
        + [(s, r) for s, runs in reruns.items() for rows in runs for r in rows],
    }
    everything = [item for items in sides.values() for item in items]
    generations = Counter(r["generation"] for _, r in everything if r.get("generation") is not None)
    prompts, off = {}, []
    for side, items in sides.items():
        versions = {}
        for s, r in items:
            for name, version in (r.get("prompts") or {}).items():
                if version is None:
                    continue
                versions.setdefault(name, []).append((version, "%s %s t%d" % (s, r["id"], r["turn"])))
        prompts[side] = {}
        for name, seen in versions.items():
            counts = Counter(version for version, _ in seen)
            prompts[side][name] = {_label(v): n for v, n in counts.items()}
            majority = counts.most_common(1)[0][0]  # a tie: the first-seen version
            for version in counts:
                if version != majority:
                    rows = [where for v, where in seen if v == version]
                    more = ", +%d more" % (len(rows) - 10) if len(rows) > 10 else ""
                    off.append(
                        "%s %s %s off %s: %s%s"
                        % (side, name, _label(version), _label(majority), ", ".join(rows[:10]), more)
                    )
    void = len(generations) > 1 or bool(off)
    return {
        "pass": not void,
        "void": void,
        "rows": len(everything),
        "unrecorded": sum(1 for _, r in everything if r.get("generation") is None),
        "generations": {str(g): n for g, n in sorted(generations.items())},
        "prompts": prompts,
        "off": off,
    }


def _provenance_line(p):
    if p["void"]:
        reasons = []
        if len(p["generations"]) > 1:
            reasons.append("generations " + ", ".join("%s (%d rows)" % (g, n) for g, n in p["generations"].items()))
        return "provenance VOID  " + "; ".join(reasons + p["off"])
    side = lambda s: ", ".join("%s %s" % (name, next(iter(v))) for name, v in p["prompts"][s].items()) or "-"  # noqa: E731
    return "provenance OK    generation %s; unrecorded %d of %d rows; prompts reference %s; candidate %s" % (
        next(iter(p["generations"]), "-"),
        p["unrecorded"],
        p["rows"],
        side("reference"),
        side("candidate"),
    )


def evaluate(cand, ref, reruns, cfg, rerun_label, cand_dirs=None, index=None):
    """cand, ref: set -> rows; reruns: set -> [rows, ...] (oldest first); cand_dirs: set -> the candidate's folder and index:
    conversation id -> (set, conversation), which let a pending turn be replayed. Returns (result, report lines,
    re-run commands)."""
    checks, lines, commands = {}, [], []

    checks["provenance"] = provenance(cand, ref, reruns)  # first: a void pair is judged by nothing else
    lines.append(_provenance_line(checks["provenance"]))

    turns = [r for rows in cand.values() for r in rows] + [r for runs in reruns.values() for rows in runs for r in rows]
    erred = ["%s t%d" % (r["id"], r["turn"]) for r in turns if r["checks"]["errors"]]
    hung = ["%s t%d" % (r["id"], r["turn"]) for r in turns if (r.get("total_s") or 0) > cfg.answers.hang_s]
    checks["errors"] = {"pass": not erred and not hung, "errors": erred, "hangs": hung, "turns": len(turns)}
    lines.append(
        "errors     %s  %d candidate turns: %d with an error %s, %d over %g s %s"
        % (
            _verdict(checks["errors"]["pass"]),
            len(turns),
            len(erred),
            erred or "",
            len(hung),
            cfg.answers.hang_s,
            hung or "",
        )
    )

    names = [s for s in cfg.retrieval.sets if s in cand and s in ref]
    volatile = set(cfg.retrieval.volatile)

    def cut(rows, drop=()):
        return [dict(r, sources=(r.get("sources") or [])[: cfg.retrieval.top_k]) for r in rows if r["id"] not in drop]

    full = {s: stats.compare_pair(cut(ref[s]), cut(cand[s])) for s in names}
    pairs = {s: stats.compare_pair(cut(ref[s], volatile), cut(cand[s], volatile)) for s in names}
    pooled = stats.mean([x for p in pairs.values() for x in p["scores"]])
    missing = [m for p in full.values() for m in p["missing"]]
    if missing or (pooled is not None and pooled < cfg.retrieval.floor):
        status = "FAIL"
    elif pooled is not None and pooled < cfg.baseline.noise_floor:
        status = "EXPLAIN"
    else:
        status = "PASS"
    differing = [(s, d) for s, p in pairs.items() for d in p["differing"]]
    checks["retrieval"] = {
        "pass": status != "FAIL",
        "explain": status == "EXPLAIN",
        "pooled": pooled,
        "noise_floor": cfg.baseline.noise_floor,
        "floor": cfg.retrieval.floor,
        "missing": missing,
        "volatile_not_counted": sorted(volatile),
        "sets": {
            s: {
                "agreement": stats.mean(p["scores"]),
                "with_volatile": stats.mean(full[s]["scores"]),
                "baseline": cfg.baseline.per_set.get(s),
                "scored": len(p["scores"]),
                "no_retrieval": len(p["no_retrieval"]),
                "differing": p["differing"],
                "volatile": [d for d in full[s]["differing"] if d["id"] in volatile],
            }
            for s, p in pairs.items()
        },
    }
    lines.append(
        "retrieval  %s  pooled %s (explain below %.4f, fail below %.4f) over %d questions; %s%s"
        % (
            status,
            stats.fmt(pooled),
            cfg.baseline.noise_floor,
            cfg.retrieval.floor,
            sum(len(p["scores"]) for p in pairs.values()),
            "; ".join(
                "%s %s (with volatile %s, baseline %s)"
                % (s, stats.fmt(v["agreement"]), stats.fmt(v["with_volatile"]), stats.fmt(v["baseline"]))
                for s, v in checks["retrieval"]["sets"].items()
            )
            or "no retrieval set in both runs",
            "; missing: " + ", ".join(missing) if missing else "",
        )
    )
    for s, d in differing:
        lines.append(
            "           differing %s %s overlap %.4f%s"
            % (s, d["id"], d["overlap"], " (needs an explanation)" if status == "EXPLAIN" else "")
        )
    for s, v in checks["retrieval"]["sets"].items():
        for d in v["volatile"]:
            lines.append("           volatile %s %s overlap %.4f (reported, not counted)" % (s, d["id"], d["overlap"]))

    answer_ok, per_set = True, {}
    for name in sorted(set(cand) & set(ref)):
        flaky = set(cfg.sets[name].flaky) if name in cfg.sets else set()
        ref_pass, cand_pass = _passes(ref[name]), _passes(cand[name])
        again = [_passes(rows) for rows in reruns.get(name, [])]
        missing_turns = ["%s t%d" % k for k in sorted(set(ref_pass) ^ set(cand_pass))]
        keys = [k for k in sorted(set(ref_pass) & set(cand_pass)) if "%s t%d" % k not in flaky]
        outcomes = {
            k: ([cand_pass[k]] + [run[k] for run in again if k in run])[: cfg.answers.rerun_runs]
            for k in keys
            if ref_pass[k] and not cand_pass[k]
        }
        pending = sorted(k for k, o in outcomes.items() if len(o) < cfg.answers.rerun_runs)
        regressions = sorted(
            k
            for k, o in outcomes.items()
            if len(o) == cfg.answers.rerun_runs and o.count(False) >= cfg.answers.regression_at
        )
        settled = sum(cand_pass[k] if k not in outcomes else (k not in regressions and k not in pending) for k in keys)
        before = sum(ref_pass[k] for k in keys)
        ok = not missing_turns and not regressions and (bool(pending) or settled >= before)
        answer_ok = answer_ok and ok
        per_set[name] = {
            "pass": ok,
            "reference": before,
            "candidate": settled,
            "missing": missing_turns,
            "regressions": ["%s t%d" % k for k in regressions],
            "pending": ["%s t%d" % k for k in pending],
            "reruns": {"%s t%d" % k: o for k, o in outcomes.items()},
            "flaky_excluded": sorted(flaky),
        }
        dirs = cand_dirs or {}

        def replays(cid, name=name):
            conv = (index or {}).get(cid)
            rows = [r for r in cand[name] if r["id"] == cid]
            return bool(name in dirs and conv and not replay_problem(conv[1], rows))

        # the next re-run numbers -> the pending turns that still lack that run, counted per turn; a replayable conversation is
        # counted by its least-run pending turn and lists all its flipped turns (settled too), so every replay starts at its earliest
        wanted = {}
        for key in pending:
            mates = sorted(m for m in outcomes if m[0] == key[0]) if replays(key[0]) else [key]
            first = min(len(outcomes[m]) for m in mates if m in pending)
            for number in range(first, cfg.answers.rerun_runs):
                wanted.setdefault(number, set()).update(mates)

        for k, keys_ in sorted(wanted.items()):
            again_turns = sorted(key for key in keys_ if replays(key[0]))
            whole = sorted({cid for cid, _ in keys_ if not replays(cid)})
            if again_turns:
                commands.append(
                    "evals run %s --replay %s --turns %s --label %s-rerun%d"
                    % (
                        name,
                        os.path.basename(dirs[name].rstrip("/")),
                        ",".join("%s:t%d" % key for key in again_turns),
                        rerun_label,
                        k,
                    )
                )
            if whole:
                commands.append("evals run %s --ids %s --label %s-rerun%d" % (name, ",".join(whole), rerun_label, k))
    checks["answers"] = {"pass": answer_ok, "sets": per_set}
    lines.append(
        "answers    %s  %s"
        % (
            _verdict(answer_ok),
            "; ".join(
                "%s %d -> %d%s%s%s"
                % (
                    s,
                    v["reference"],
                    v["candidate"],
                    " regressions " + ",".join(v["regressions"]) if v["regressions"] else "",
                    " re-runs needed " + ",".join(v["pending"]) if v["pending"] else "",
                    " missing " + ",".join(v["missing"]) if v["missing"] else "",
                )
                for s, v in per_set.items()
            ),
        )
    )
    for s, v in per_set.items():
        for turn, outcome in v["reruns"].items():
            lines.append("           re-run %s %s: %s" % (s, turn, " ".join("ok" if o else "FAIL" for o in outcome)))

    lat = [s for s in cfg.latency.sets if s in cand and s in ref]
    ref_med = stats.median([t for s in lat for t in stats.ttfts(stats.first_turns(ref[s]))])
    cand_med = stats.median([t for s in lat for t in stats.ttfts(stats.first_turns(cand[s]))])
    limit = None if ref_med is None else ref_med + cfg.baseline.ttft_margin_s
    ok = limit is None or cand_med is None or cand_med <= limit
    checks["latency"] = {
        "pass": ok,
        "reference_median": ref_med,
        "candidate_median": cand_med,
        "limit": limit,
        "sets": lat,
    }
    lines.append(
        "latency    %s  TTFT median %s -> %s s, limit %s s (%s)"
        % (
            _verdict(ok),
            stats.fmt(ref_med, "%.2f"),
            stats.fmt(cand_med, "%.2f"),
            stats.fmt(limit, "%.2f"),
            ", ".join(lat) or "no latency set in both runs",
        )
    )

    failed = [name for name, check in checks.items() if not check["pass"]]
    pending_any = any(v["pending"] for v in per_set.values())
    void = checks["provenance"]["void"]
    code = 4 if void else 1 if failed else 2 if pending_any else 3 if checks["retrieval"]["explain"] else 0
    if void:
        commands = []  # re-runs and explanations of a void pair mean nothing: the pair is recorded again
    lines.append(
        "RESULT %s (exit %d)"
        % (
            {
                0: "PASS",
                1: "FAIL: " + ", ".join(failed),
                2: "RE-RUNS NEEDED",
                3: "EXPLAIN: each differing question above needs an explanation",
                4: "VOID: the runs span index generations or a side mixes prompt versions; record the pair again",
            }[code],
            code,
        )
    )
    lines += ["  " + c for c in commands]
    return {"exit": code, "checks": checks, "commands": commands}, lines, commands


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="evals gate", description="The structural gate: candidate runs against the reference"
    )
    ap.add_argument("cand_label", nargs="?", default="")
    ap.add_argument("--against", default="", help="the reference label")
    ap.add_argument("--cand", nargs="+", default=[], metavar="DIR")
    ap.add_argument("--ref", nargs="+", default=[], metavar="DIR")
    ap.add_argument("--reruns", nargs="+", default=[], metavar="DIR")
    ap.add_argument(
        "--sets", default="", help="the sets scored, a comma list (default core; core expands); labels only"
    )
    args = ap.parse_args(argv)
    labels = bool(args.cand_label) and bool(args.against)
    folders = bool(args.cand) and bool(args.ref)
    if labels == folders or (labels and (args.cand or args.ref)) or (folders and (args.cand_label or args.against)):
        ap.error("give <cand-label> --against <ref-label>, or --cand DIR... --ref DIR...")
    settings = load_settings().evals
    cfg = load_config(settings.profile_dir)
    index = conversations(settings.profile_dir, cfg)
    members = {name: set_ids(settings.profile_dir, cfg, name) for name in cfg.sets}
    if folders and args.sets:
        ap.error("--sets belongs to labels; folders declare exactly the sets they are")
    if labels:
        declared = list(
            dict.fromkeys(
                n for part in (args.sets or "core").split(",") for n in (cfg.core if part == "core" else [part]) if n
            )
        )
        if unknown := [n for n in declared if n not in cfg.sets]:
            ap.error("unknown set %s (gate.yaml: %s)" % (", ".join(unknown), ", ".join(cfg.sets)))
        cand_dirs = newest_runs(settings.runs_dir, args.cand_label, declared)
        ref_dirs = newest_runs(settings.runs_dir, args.against, declared)
        for who, found in (("candidate", cand_dirs), ("reference", ref_dirs)):
            if lacking := [n for n in declared if n not in found]:
                ap.error("no %s run for %s" % (who, ", ".join(lacking)))
        rerun_dirs = rerun_runs(settings.runs_dir, args.cand_label, cand_dirs)
    else:
        cand_dirs, ref_dirs, rerun_dirs = by_set(args.cand, index, members), by_set(args.ref, index, members), {}
    # a folder counts once: one found by its label and named again, one named twice, or a candidate named as a re-run
    seen = {os.path.realpath(p) for ps in rerun_dirs.values() for p in ps} | {
        os.path.realpath(p) for p in cand_dirs.values()
    }
    for folder in args.reruns:
        if os.path.realpath(folder) not in seen:
            seen.add(os.path.realpath(folder))
            rerun_dirs.setdefault(set_of(read_rows(folder), index, folder, members), []).append(folder)
    if stray := sorted(set(rerun_dirs) - set(cand_dirs)):
        ap.error("re-runs of %s, which has no candidate run (declare it with --sets)" % ", ".join(stray))
    if not cand_dirs:
        ap.error("no candidate run found")
    if lacking := sorted(set(cand_dirs) - set(ref_dirs)):  # folder mode
        ap.error("no reference run for " + ", ".join(lacking))
    if not set(cand_dirs) & set(cfg.retrieval.sets) or not set(cand_dirs) & set(cfg.latency.sets):
        ap.error(
            "declare at least one of %s (retrieval and latency read their first turns)" % ", ".join(cfg.retrieval.sets)
        )
    cand = {s: stats.load_rows(d) for s, d in cand_dirs.items()}
    ref = {s: stats.load_rows(ref_dirs[s]) for s in cand_dirs}
    reruns = {s: [stats.load_rows(d) for d in ds] for s, ds in rerun_dirs.items()}
    result, lines, _ = evaluate(cand, ref, reruns, cfg, args.cand_label or "gate", cand_dirs, index)
    for s in cand_dirs:
        print("%-10s candidate %s | reference %s | re-runs %s" % (s, cand_dirs[s], ref_dirs[s], rerun_dirs.get(s, [])))
    print("\n".join(lines))
    result["runs"] = {"candidate": cand_dirs, "reference": {s: ref_dirs[s] for s in cand_dirs}, "reruns": rerun_dirs}
    os.makedirs(settings.out_dir, exist_ok=True)
    out = os.path.join(
        settings.out_dir,
        "%s_%s.json"
        % ((args.cand_label + "-gate") if args.cand_label else "gate", datetime.now().strftime("%Y%m%d-%H%M%S-%f")),
    )
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("Report: " + out)
    return result["exit"]
