"""The probe runner of the evals job.

Plays scripted conversations against the assistant's /api/chat over HTTP and records per turn the streamed answer,
the sources of the `docs` frame, error frames, latency and the checks of evals.checks. In the stack it runs in its own
container, one run at a time (Docker refuses a second container of the same name), and the run is copied out after it:

  docker compose --env-file .env -f deploy/compose.yaml --profile evals run --no-deps --name evals-run \\
      evals run <set> [<set> ...] --label <label>
  docker cp evals-run:/app/out/. data/eval_runs/
  docker rm evals-run

Each set writes <out_dir>/<label>-<set>_<YYYYmmdd-HHMMSS>/ (results.jsonl, summary.md); out_dir is the container's /app/out
(data/eval_runs on a development machine). --probes <file> runs any file in the probe format (<label>_<stamp>/); --record <dir> keeps each
turn's bytes."""

import argparse
import codecs
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import httpx

from evals.checks import LINE_METADATA, evaluate_turn, turn_expect
from evals.gate import load_config, read_rows, replay_problem
from evals.settings import load_settings
from evals.sse import parse_sse


def section_id(item, url):
    """A source's section identity: its url, its part_index
    (else its page) and the sha-256 of its text, so two runs on one index generation can be compared section by section."""
    meta = (item or {}).get("metadata") or {}
    position = meta.get("part_index")
    if position is None:
        position = meta.get("page_number", "")
    digest = hashlib.sha256(str((item or {}).get("page_content") or "").encode("utf-8")).hexdigest()
    return "%s|%s|%s" % (url, position, digest)


def sources_from_event(content):
    """Sources of a `docs` event as [{url, title, internal, section}]; internal is the source's audience metadata."""
    sources = []
    for item in content or []:
        meta = (item or {}).get("metadata") or {}
        url = meta.get("download_url") or meta.get("url") or meta.get("url_location") or ""
        if not url:
            continue
        internal = meta.get("audience") == "internal"
        title = meta.get("title") or meta.get("download_title") or meta.get("filename") or ""
        sources.append({"url": url, "title": title, "internal": internal, "section": section_id(item, url)})
    return sources


def source_metadata(citations, docs):
    """{n: metadata} of the documents the citations point at: a citation's url is its document's download_url or url, as
    the docs frame carries them (one entry per number; a citation whose document is not in the frame has none). Only the
    metadata the appended lines depend on (checks.LINE_METADATA) is kept."""
    by_url = {}
    for item in docs or []:
        meta = (item or {}).get("metadata") or {}
        if url := meta.get("download_url") or meta.get("url"):
            by_url.setdefault(url, {k: meta[k] for k in LINE_METADATA if k in meta})
    return {
        c["n"]: by_url[c["url"]]
        for c in citations or []
        if isinstance(c, dict) and isinstance(c.get("n"), int) and c.get("url") in by_url
    }


def build_question(turn):
    """The question text of a turn; `repeat` builds long inputs without storing them."""
    q = turn["q"]
    rep = turn.get("repeat")
    if rep:
        q = (rep["text"] + " ") * int(rep["times"]) + q
    return q


def ask(
    client,
    base_url,
    token,
    question,
    user,
    timeout_s,
    turn_id=None,
    replace_from=None,
    abort_after_s=None,
    kept=None,
    formats=None,
    variant=None,
    variant_of=None,
    variant_select=None,
    meta=None,
    record=None,
):
    """One request; returns (answer, docs_content, errors, first_token_s, total_s, event_types). abort_after_s closes the
    connection after that many seconds (a client stop: the backend sees a disconnect); event_types then ends 'aborted'.
    kept (with replace_from): the ids of the questions the page still shows before the replaced one.
    formats, variant, variant_of, variant_select: response formats; meta (a dict) receives the shape, citations (their
    numbers, and as source_metadata each cited document's line metadata from the docs frame, joined after the stream), variant and status-step frames.
    record: a path stem; the response bytes go to <stem>.raw exactly as received, the
    HTTP status, the response headers and the query parameters (never the token) to <stem>.json. Recorded or not, the
    bytes pass the same incremental UTF-8 decoder, so a recorded run measures like any other."""
    started = time.monotonic()
    first, docs, buf = None, None, ""
    cited_entries = None  # the citations frame's entries; None when there was none
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    parts, errors, types = [], [], []
    params = {"question": question, "conversation_id": user}
    if turn_id:
        params["turn_id"] = turn_id
    if replace_from:
        params["replace_from"] = replace_from
        if kept is not None:
            params["kept"] = ",".join(kept)
    if formats:
        params["formats"] = formats
    if variant:
        params["variant"] = variant
    if variant_select:
        params["variant_select"] = variant_select
    if variant_of:
        params["variant_of"] = variant_of

    def take(events):
        nonlocal first, docs, cited_entries
        for ev in events:
            typ = ev.get("type")
            types.append(typ)
            content = ev.get("content")
            if typ == "chunk":
                if first is None:
                    first = time.monotonic() - started
                parts.append(str(ev.get("content", "")))
            elif typ == "docs":
                docs = content
            elif typ == "error":
                errors.append(str(ev.get("content", "")))
            elif meta is not None and typ == "shape":
                meta["shape"] = content
            elif meta is not None and typ == "citations":
                meta["citations"] = (
                    [c.get("n") for c in content if isinstance(c, dict)] if isinstance(content, list) else []
                )
                cited_entries = content if isinstance(content, list) else []
            elif meta is not None and typ == "variant":
                meta["variant_ok"] = bool((content or {}).get("ok"))
            elif meta is not None and typ == "status":
                meta.setdefault("steps", []).append(content.get("step") if isinstance(content, dict) else None)
            elif meta is not None and typ == "done" and isinstance(content, dict):
                # the turn's index generation and prompt versions (the gate's void rule)
                meta["generation"] = content.get("generation")
                meta["prompts"] = content.get("prompts")
                meta["scope_fallback"] = content.get(
                    "scope_fallback"
                )  # true on an out-of-scope question answered by retrieval

    raw = open(record + ".raw", "wb") if record else None
    status, headers = None, {}
    try:
        with client.stream(
            "GET",
            base_url + "/api/chat",
            params=params,
            headers={"Authorization": "Bearer " + token},
            timeout=timeout_s,
        ) as resp:
            status, headers = resp.status_code, dict(resp.headers)
            if resp.status_code != 200:
                body = resp.read()
                if raw:
                    raw.write(body)
                errors.append("http %s: %s" % (resp.status_code, body.decode("utf-8", errors="replace")[:300]))
            else:
                for piece in resp.iter_bytes():
                    if raw:
                        raw.write(piece)
                    events, buf = parse_sse(buf + decoder.decode(piece))
                    take(events)
                    if abort_after_s is not None and time.monotonic() - started > abort_after_s:
                        types.append("aborted")
                        break
                else:
                    events, buf = parse_sse(buf + decoder.decode(b"", final=True))
                    take(events)
    except Exception as exc:  # network errors, timeouts
        errors.append("runner: %s: %s" % (type(exc).__name__, exc))
    finally:
        if raw and record:
            raw.close()
            with open(record + ".json", "w", encoding="utf-8") as fh:
                json.dump({"status": status, "headers": headers, "params": params}, fh, ensure_ascii=False, indent=1)
    if (
        meta is not None and cited_entries is not None
    ):  # after the stream: the docs frame may follow the citations frame
        meta["source_metadata"] = source_metadata(cited_entries, docs)
    return "".join(parts), docs, errors, first, time.monotonic() - started, types


def run_stamp():
    """Run folder and conversation names carry seconds, so two runs never share them."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_conversation(conv, ask_fn, min_interval, label, stamp, emit, formats=None, record_dir=None, replay=None):
    """Play one conversation's turns in order and return its rows; emit(row) after each turn.
    ask_fn(question, user, opts) returns the tuple of ask(); opts carries the turn's turn_id, replace_from (the id of
    turn `replace_turn`: an edit or a retry), kept (the ids of the turns the page still shows before it, at most five),
    abort_after_s (a client stop), formats and a meta dict for the shape and variant events. A turn {"variant": name} or
    {"select": name} changes the answer of the last question (response formats): it sends that question's id and
    gets no id of its own. record_dir: keep each turn's bytes as <record_dir>/<id>_t<n>.raw and .json.
    replay: {user, start, end, ids, shapes} of the stored candidate rows. Plays only the turns start..end
    as the stored user, with their own numbers. The first question is sent as a retry of its stored id (replace_from) with the
    stored earlier question ids as kept, so the backend answers on the stored history and a second replay sees the same one; a
    variant of a question before `start` names its stored id."""
    rows = []
    user = replay["user"] if replay else "probe-%s-%s-%s" % (label, stamp, conv["id"])
    ids_by_turn = {}
    shown = []  # the turns the page still shows, in order: an edit or a retry cuts at the replaced turn
    last = None  # (turn number, question) of the last question asked
    last_shape = None  # the shape event of the last question: the original's form
    start, stop, cut = 1, len(conv["turns"]), None
    if replay:
        start, stop = replay["start"], replay["end"]
        shown = [n for n in sorted(replay["ids"]) if n < start]
        ids_by_turn = {n: replay["ids"][n] for n in shown}
        if shown:
            last, last_shape = (
                (shown[-1], build_question(conv["turns"][shown[-1] - 1])),
                replay["shapes"].get(shown[-1]),
            )
        cut = True  # the first question played replaces its stored turn
    for t_idx, turn in enumerate(conv["turns"], 1):
        if not start <= t_idx <= stop:
            continue
        k = None
        if "variant" in turn or "select" in turn:
            assert last is not None, "a variant or select turn follows a question"
            q_idx, question = last
            opts = {
                "formats": formats or "v1",
                "variant_of": ids_by_turn[q_idx],
                "meta": {},
                "variant": turn.get("variant"),
                "variant_select": turn.get("select"),
            }
        else:
            question = build_question(turn)
            k = turn.get("replace_turn")
            kept = None
            if k:
                if k in shown:
                    shown = shown[: shown.index(k)]
                kept = [ids_by_turn[n] for n in shown][-5:]
            opts = {
                "turn_id": str(uuid.uuid4()),
                "replace_from": ids_by_turn.get(k) if k else None,
                "abort_after_s": turn.get("abort_after_s"),
                "kept": kept,
                "formats": formats,
                "meta": {},
            }
            if cut and replay:  # a replay's first question: the retry of its stored turn
                opts["replace_from"], opts["kept"], cut = (
                    replay["ids"][t_idx],
                    [ids_by_turn[n] for n in shown][-5:],
                    None,
                )
            ids_by_turn[t_idx] = opts["turn_id"]
            shown.append(t_idx)
            last = (t_idx, question)
        if record_dir:
            opts["record"] = os.path.join(record_dir, "%s_t%d" % (conv["id"], t_idx))
        answer, docs, errors, first, total, types = ask_fn(question, user, opts)
        meta = opts.get("meta") or {}
        if "variant" not in turn and "select" not in turn:
            last_shape = meta.get("shape")
        sources = sources_from_event(docs)
        expect = turn_expect(turn, last_shape)
        checks = evaluate_turn(answer, sources, errors, expect, meta)
        if turn.get("abort_after_s") is not None:
            checks = dict(checks, failed=[], auto_pass=True, manual_review=False)  # a set-up step: stopped on purpose
        row = {
            "id": conv["id"],
            "set": conv.get("set"),
            "turn": t_idx,
            "user": user,
            "turn_id": opts.get("turn_id"),  # null for a variant or select turn; a replay (evals gate) names it
            "question": question if len(question) < 500 else question[:500] + "…",
            "answer": answer,
            "sources": sources,
            "first_token_s": first,
            "total_s": total,
            "event_types": types,
            "checks": checks,
            "expect": expect,
            "aborted": "aborted" in types,
            "replace_from_turn": k or None,
            "shape": meta.get("shape"),
            "variant": turn.get("variant") or turn.get("select"),
            "citations": meta.get("citations"),
            "source_metadata": meta.get("source_metadata"),
            "variant_ok": meta.get("variant_ok"),
            "generation": meta.get("generation"),  # None = unrecorded (an old backend, a client stop, a variant)
            "prompts": meta.get("prompts"),
            "scope_fallback": meta.get("scope_fallback"),  # older rows have no key (readers use .get)
        }
        rows.append(row)
        emit(row)
        time.sleep(max(0.0, min_interval - total))
    return rows


def run_all(convs, concurrency, run_one):
    """run_one(conv) for every conversation, at most `concurrency` at a time (load test);
    the rows come back in the probe file's order."""
    if concurrency <= 1:
        return [row for conv in convs for row in run_one(conv)]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(run_one, convs))
    return [row for rows in results for row in rows]


def md_cell(text, limit=160):
    text = " ".join(str(text).split())
    text = text.replace("|", "\\|")
    return text[:limit] + ("…" if len(text) > limit else "")


def write_summary(path, label, rows):
    turns = len(rows)
    passed = sum(1 for r in rows if r["checks"]["auto_pass"])

    def count(key):
        return sum(1 for r in rows if key in r["checks"]["failed"])

    lat = [r["total_s"] for r in rows if r["total_s"] is not None]
    lines = [
        "# Probe run %s" % label,
        "",
        "Turns: %d | auto-pass: %d | error events: %d | empty: %d | internal sources: %d | "
        "leaks: %d | emoji: %d | informal: %d | mean latency: %.1f s"
        % (
            turns,
            passed,
            count("error_event"),
            count("empty_answer"),
            count("internal_source"),
            count("leak"),
            count("emoji"),
            count("informal_address"),
            (sum(lat) / len(lat)) if lat else 0.0,
        ),
        "",
        "| id | turn | pass | failed | manual | words | sources | internal | s | question | answer |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        c = r["checks"]
        lines.append(
            "| %s | %d | %s | %s | %s | %d | %d | %d | %.0f | %s | %s |"
            % (
                r["id"],
                r["turn"],
                "yes" if c["auto_pass"] else "no",
                ", ".join(c["failed"]),
                "yes" if c["manual_review"] else "",
                c["words"],
                c["n_sources"],
                c["internal_sources"],
                r["total_s"] or 0,
                md_cell(r["question"], 80),
                md_cell(r["answer"]),
            )
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def play(
    probes_path,
    folder_label,
    settings,
    *,
    ids=(),
    groups=(),
    concurrency=1,
    formats=None,
    record_dir=None,
    replay=None,
    client_factory=httpx.Client,
):
    """Play one probe file; returns its run folder <out_dir>/<folder_label>_<stamp>. replay: conversation id -> the replay
    dict of run_conversation."""
    with open(probes_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    convs = [c for c in spec["conversations"] if (not groups or c.get("set") in groups) and (not ids or c["id"] in ids)]
    token = settings.token.get_secret_value()
    stamp = run_stamp()
    run_dir = os.path.join(settings.out_dir, "%s_%s" % (folder_label, stamp))
    os.makedirs(run_dir, exist_ok=False)
    lock = threading.Lock()
    local = threading.local()
    clients = []

    def ask_fn(question, user, opts):
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = client_factory()
            with lock:
                clients.append(client)
        return ask(client, settings.assistant_url, token, question, user, settings.timeout_s, **opts)

    with open(os.path.join(run_dir, "results.jsonl"), "w", encoding="utf-8") as out:

        def emit(row):
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                c = row["checks"]
                print(
                    "%-6s t%d %-4s %5.1fs %s"
                    % (
                        row["id"],
                        row["turn"],
                        "ok" if c["auto_pass"] else "FAIL",
                        row["total_s"],
                        ",".join(c["failed"]),
                    ),
                    flush=True,
                )

        try:
            rows = run_all(
                convs,
                concurrency,
                lambda conv: run_conversation(
                    conv,
                    ask_fn,
                    settings.min_interval_s,
                    folder_label,
                    stamp,
                    emit,
                    formats=formats,
                    record_dir=record_dir,
                    replay=(replay or {}).get(conv["id"]),
                ),
            )
        finally:
            for client in clients:
                client.close()
    write_summary(os.path.join(run_dir, "summary.md"), folder_label, rows)
    print("Results: " + run_dir)
    return run_dir


def replay_plan(ap: argparse.ArgumentParser, settings, probes_path, folder, turns):
    """--replay FOLDER --turns <id>:t<n>,... (also "<id> t<n>"): conversation id -> the replay dict of run_conversation, from
    the candidate's stored rows (user, turn ids, shapes); a conversation that cannot replay is a usage error naming it."""
    wanted = {}
    for item in (t.strip() for t in turns.split(",") if t.strip()):
        match = re.fullmatch(r"(\S+?)[: ]t(\d+)", item)
        if not match:
            ap.error('--turns takes "<id>:t<n>" entries, not "%s"' % item)
        wanted.setdefault(match.group(1), []).append(int(match.group(2)))
    stored = read_rows(os.path.join(settings.runs_dir, folder))
    with open(probes_path, encoding="utf-8") as fh:
        convs = {c["id"]: c for c in json.load(fh)["conversations"]}
    plan = {}
    for cid, ns in wanted.items():
        rows = [r for r in stored if r["id"] == cid]
        if cid not in convs or not rows or not 1 <= min(ns) <= max(ns) <= len(convs[cid]["turns"]):
            ap.error("%s: not in the set, not in %s, or a turn out of range" % (cid, folder))
        if problem := replay_problem(convs[cid], rows):
            ap.error("%s cannot replay: %s (re-run it whole with --ids)" % (cid, problem))
        plan[cid] = {
            "user": rows[0]["user"],
            "start": min(ns),
            "end": max(ns),
            "ids": {r["turn"]: r["turn_id"] for r in rows if r.get("turn_id")},
            "shapes": {r["turn"]: r.get("shape") for r in rows if r.get("turn_id")},
        }
    return plan


def main(argv=None, client_factory=httpx.Client):
    ap = argparse.ArgumentParser(prog="evals run", description="Play probe sets against the assistant")
    ap.add_argument("sets", nargs="*", help="sets of gate.yaml (an unknown one lists them all)")
    ap.add_argument("--probes", default="", help="a file in the probe format instead of named sets")
    ap.add_argument("--label", required=True)
    ap.add_argument("--ids", default="", help="comma list of conversation ids")
    ap.add_argument("--groups", default="", help="comma list of the conversations' inner set field")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="conversations played in parallel (load test); one conversation's turns stay in order",
    )
    ap.add_argument("--formats", default="", choices=["", "v1"], help="send formats=v1 (a set of gate.yaml may set it)")
    ap.add_argument(
        "--record",
        default="",
        help="keep each turn's response bytes and HTTP facts in this folder (in the container: under /app/out)",
    )
    ap.add_argument(
        "--replay",
        default="",
        metavar="FOLDER",
        help="a candidate run folder in runs_dir: re-send only --turns of it, on its stored earlier turns (one set)",
    )
    ap.add_argument("--turns", default="", help="with --replay: comma list of <id>:t<n>, e.g. G28:t2,FV01:t7")
    args = ap.parse_args(argv)
    if bool(args.sets) == bool(args.probes):
        ap.error("name sets of gate.yaml or one --probes file")
    if bool(args.replay) != bool(args.turns) or (args.replay and (len(args.sets) != 1 or args.ids)):
        ap.error("--replay FOLDER and --turns come together, with exactly one named set and no --ids")
    settings = load_settings().evals
    if settings.token is None:
        print("no backend token: mount the compose secret backend_token or set RAG__EVALS__TOKEN", file=sys.stderr)
        return 2
    given = {i for i in args.ids.split(",") if i}
    if args.probes:
        jobs = [(args.label, args.probes, args.formats or None, given)]
    else:
        cfg = load_config(settings.profile_dir)
        unknown = [s for s in args.sets if s not in cfg.sets]
        if unknown:
            ap.error("unknown set %s (gate.yaml: %s)" % (", ".join(unknown), ", ".join(cfg.sets)))
        jobs = []
        for s in args.sets:
            panel = set(cfg.sets[s].ids)
            ids = (
                panel & given if panel and given else panel or given
            )  # a panel plays its ids, within --ids; else --ids or all
            if panel and not ids:
                continue  # --ids names none of this panel's conversations
            jobs.append(
                (
                    "%s-%s" % (args.label, s),
                    os.path.join(settings.profile_dir, cfg.sets[s].file),
                    args.formats or cfg.sets[s].formats,
                    ids,
                )
            )
        if not jobs:
            ap.error("--ids names none of the panel's conversations")
    replay = replay_plan(ap, settings, jobs[0][1], args.replay, args.turns) if args.replay else None
    if replay:
        jobs = [(*jobs[0][:3], set(replay))]
    if args.record:
        os.makedirs(args.record, exist_ok=True)
    for folder_label, path, formats, ids in jobs:
        play(
            path,
            folder_label,
            settings,
            ids=ids,
            replay=replay,
            groups={g for g in args.groups.split(",") if g},
            concurrency=args.concurrency,
            formats=formats,
            record_dir=args.record or None,
            client_factory=client_factory,
        )
    return 0
