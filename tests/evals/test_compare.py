"""evals compare: per-run summaries and the turns fixed and newly failed."""

import json

from evals import compare


def write_run(folder, rows):
    folder.mkdir()
    (folder / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(folder)


def row(cid, failed):
    return {
        "id": cid,
        "turn": 1,
        "set": "gold",
        "answer": "Antwort",
        "total_s": 8.0,
        "first_token_s": 6.0,
        "checks": {"failed": failed, "errors": [], "n_sources": 3},
    }


def test_fixed_and_new_failures_from_the_first_to_the_last_run(tmp_path, capsys):
    a = write_run(tmp_path / "a", [row("G01", ["contains_any"]), row("G02", []), row("G03", ["emoji"])])
    b = write_run(tmp_path / "b", [row("G01", []), row("G02", ["source_url_any"]), row("G03", [])])
    assert compare.changes(compare.load(a), compare.load(b)) == (["G01t1:contains_any"], ["G02t1:source_url_any"])
    assert compare.main([a, b]) == 0
    out = capsys.readouterr().out
    assert (
        "FIXED (first -> last): ['G01t1:contains_any']" in out
        and "NEW FAILURES (first -> last): ['G02t1:source_url_any']" in out
    )
