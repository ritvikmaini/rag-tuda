"""The list safeguard: a link missing from one list keeps its PDF and its place in the build; it is removed only after
grace_runs consecutive misses; a sharply shorter list changes nothing until it persists; the same list served again
counts nothing; a corrupt state file keeps everything. Driven through sync_pdfs on temporary folders."""

import dataclasses
import hashlib
import json

import pytest

from indexer.sync import safeguard
from indexer.sync.pdfs import PdfEntry, sync_pdfs
from indexer.sync.registry import REGISTRY, local_path
from indexer.sync.safeguard import LIST_STATE, Safeguard

SETTINGS = {"safeguard": {"grace_runs": 3, "max_shrink": 0.2}}


def url(i):
    return f"https://docs.example.org/files/doc-{i}.pdf"


@pytest.fixture
def env(tmp_path, pdf_dir, make_ctx):
    crawl = {"n": 0}  # every run is a new list (doc 0's hash changes); same_list=True serves the previous list again

    def pdf(link):
        def download():
            content = ("PDF " + link).encode()
            (ctx.cache_dir / f"{hashlib.sha256(content).hexdigest()}.json").write_text("{}", encoding="utf-8")
            return content

        return download

    ctx = make_ctx({url(i): pdf(url(i)) for i in range(10)}, settings=SETTINGS)

    def run(ids, limit=None, same_list=False, reuse=False):
        if not same_list:
            crawl["n"] += 1
        entries = [PdfEntry(url(i), f"Doc {i}", f"h{i}" if i else f"h0.{crawl['n']}", {}) for i in ids]
        return sync_pdfs(entries, dataclasses.replace(ctx, limit=limit, reuse=reuse), pdf_dir=pdf_dir)

    return {"run": run, "ctx": ctx, "pdf_dir": pdf_dir}


def local(env, i):
    return local_path(url(i), env["pdf_dir"])


def cache(env, i):
    return env["ctx"].cache_dir / (hashlib.sha256(("PDF " + url(i)).encode()).hexdigest() + ".json")


def registry(env):
    return json.loads(env["ctx"].state.path(REGISTRY).read_text(encoding="utf-8"))


def list_state_path(env):
    return env["ctx"].state.path(LIST_STATE)


def list_state(env):
    return json.loads(list_state_path(env).read_text(encoding="utf-8"))


def retained(result):
    return sorted(e.url for e in result.retained)


def test_missing_entry_kept_for_grace_runs(env):
    env["run"](range(10))
    for n in (1, 2):
        result = env["run"](range(9))
        assert retained(result) == [url(9)] and result.removed == 0
        assert result.retained[0] == PdfEntry(url(9), "Doc 9", "h9", {})  # the last list's entry
        assert url(9) in [e.url for e in result.entries]  # still part of the build
        assert local(env, 9).exists() and cache(env, 9).exists() and url(9) in registry(env)
        assert list_state(env)["missing"][url(9)]["runs"] == n
    result = env["run"](range(9))  # the third consecutive miss
    assert result.retained == [] and result.removed == 1
    assert not local(env, 9).exists() and url(9) not in registry(env)
    assert cache(env, 9).exists()  # the extraction cache stays: another link may hold the same file
    assert url(9) not in list_state(env)["missing"] and url(9) not in list_state(env)["known"]


def test_list_shrinking_over_max_ignored(env):
    env["run"](range(10))
    for n in (1, 2):
        result = env["run"](range(5))  # half the list: an unreliable crawl
        assert result.check["shrunk"] is True and result.check["shrunk_runs"] == n
        assert retained(result) == sorted(url(i) for i in range(5, 10)) and result.removed == 0
        assert all(list_state(env)["missing"][url(i)]["runs"] == 0 for i in range(5, 10))
        assert all(local(env, i).exists() for i in range(5, 10))
    result = env["run"](range(5))  # the third short list in a row is taken as real: counting starts
    assert result.check["accepted"] is True and list_state(env)["last_full_count"] == 5
    assert all(list_state(env)["missing"][url(i)]["runs"] == 1 for i in range(5, 10))
    env["run"](range(5))
    result = env["run"](range(5))
    assert result.removed == 5 and not any(local(env, i).exists() for i in range(5, 10))


def test_the_counter_restarts_when_the_link_returns(env):
    env["run"](range(10))
    env["run"](range(9))
    env["run"](range(10))
    assert url(9) not in list_state(env)["missing"]
    env["run"](range(9))
    result = env["run"](range(9))
    assert retained(result) == [url(9)] and local(env, 9).exists()


def test_the_same_list_served_again_counts_no_second_miss(env):
    env["run"](range(10))
    env["run"](range(9))
    for _ in range(3):
        result = env["run"](range(9), same_list=True)
        assert result.check["repeat"] is True and retained(result) == [url(9)] and result.removed == 0
        assert list_state(env)["missing"][url(9)]["runs"] == 1
    env["run"](range(9))
    result = env["run"](range(9))  # the third different list without the link
    assert result.removed == 1 and not local(env, 9).exists()


def test_a_short_list_served_again_is_not_a_second_short_run(env):
    env["run"](range(10))
    env["run"](range(5))
    result = env["run"](range(5), same_list=True)
    assert result.check["shrunk_runs"] == 1 and result.removed == 0
    assert list_state(env)["last_full_count"] == 10


def test_a_limited_run_neither_deletes_nor_changes_the_state(env):
    env["run"](range(10))
    before = list_state_path(env).read_text(encoding="utf-8")
    result = env["run"](range(10), limit=3)
    assert result.retained == [] and result.removed == 0
    assert list_state_path(env).read_text(encoding="utf-8") == before
    assert all(local(env, i).exists() for i in range(10))


def test_a_reuse_run_leaves_the_safeguard_state_untouched(env):
    env["run"](range(10))
    env["run"](range(9))  # link 9 missed once
    before = list_state_path(env).read_bytes()
    result = env["run"](range(9), reuse=True)
    assert retained(result) == [url(9)] and result.check["dry_run"] is True and result.removed == 0
    assert list_state_path(env).read_bytes() == before


def test_links_registered_before_the_state_file_get_a_minimal_entry(env):
    env["run"](range(10))
    list_state_path(env).unlink()  # no list state yet: the first run with the safeguard
    result = env["run"](range(9))
    assert result.retained == [PdfEntry(url(9), "", "h9", {})]


def test_a_minimal_entry_takes_its_title_from_the_extraction_cache(env):
    env["run"](range(10))
    list_state_path(env).unlink()
    cache(env, 9).write_text(json.dumps({"download_title": "Title 9"}), encoding="utf-8")
    assert env["run"](range(9)).retained[0].title == "Title 9"


@pytest.mark.parametrize(
    "content",
    [
        "null",
        "[]",
        '"x"',
        '{"last_full_count": "abc"}',
        '{"missing": {"%s": 5}}' % url(9),
        '{"known": [], "missing": []}',
        '{"known": {"%s": {"url": "%s", "title": "", "sha256": null, "meta": {}, "old_field": 1}}}' % (url(9), url(9)),
        "{broken",
    ],
)
def test_a_malformed_state_file_keeps_everything(env, content):
    env["run"](range(10))
    list_state_path(env).write_text(content, encoding="utf-8")
    result = env["run"](range(9))
    assert retained(result) == [url(9)] and result.removed == 0
    assert local(env, 9).exists() and cache(env, 9).exists() and url(9) in registry(env)
    assert result.check["corrupt"] is True  # reported, the file left as it is
    assert list_state_path(env).read_text(encoding="utf-8") == content


def test_a_failing_safeguard_keeps_every_missing_link(env, monkeypatch):
    env["run"](range(10))

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(safeguard, "save_state", boom)
    result = env["run"](range(8))
    assert retained(result) == sorted([url(8), url(9)]) and result.removed == 0
    assert result.check == {"error": "disk full", "corrupt": False}
    assert local(env, 9).exists() and url(9) in registry(env)


def test_the_settings_name_the_thresholds():
    assert Safeguard.from_settings({}) == Safeguard(grace_runs=3, max_shrink=0.2)
    assert Safeguard.from_settings(SETTINGS) == Safeguard(3, 0.2)
    assert Safeguard.from_settings({"safeguard": {"grace_runs": 0, "max_shrink": 0.5}}) == Safeguard(1, 0.5)
