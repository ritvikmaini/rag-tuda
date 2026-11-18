"""sync_pdfs on temporary folders through a FakeContext whose fetcher serves the PDFs: the list's hash decides whether a
PDF is fetched again, every download goes through the context's allowed-host fetcher, a failed download never records a
hash, and the frozen-input mode fetches and writes nothing."""

import dataclasses
import hashlib
import json

import pytest
from rag_core.domain.sources import CorruptStateError

from indexer.sync.pdfs import PDF_DOWNLOAD_TIMEOUT, PdfEntry, sync_pdfs
from indexer.sync.registry import REGISTRY, file_hash, local_path


def url(i):
    return f"https://docs.example.org/files/doc-{i}.pdf"


def entry(i, sha256=None):
    return PdfEntry(url(i), f"Doc {i}", sha256 if sha256 is not None else f"h{i}", {"n": i})


def body(i):
    return f"PDF {i}".encode()


def served(*ids):
    return {url(i): body(i) for i in ids}


def registry(ctx):
    return json.loads(ctx.state.path(REGISTRY).read_text(encoding="utf-8"))


def test_hash_unchanged_skips_download(make_ctx, pdf_dir):
    ctx = make_ctx(served(1, 2))
    first = sync_pdfs([entry(1), entry(2)], ctx, pdf_dir=pdf_dir)
    assert (first.downloaded, first.unchanged) == (2, 0)
    assert registry(ctx) == {url(1): "h1", url(2): "h2"}  # the list's hash, not the file's
    ctx.fetch.calls.clear()

    again = sync_pdfs([entry(1), entry(2)], ctx, pdf_dir=pdf_dir)
    assert ctx.fetch.calls == [] and (again.downloaded, again.unchanged) == (0, 2)
    assert [r["status"] for r in again.results] == ["cached", "cached"]

    changed = sync_pdfs([entry(1), entry(2, sha256="h2-new")], ctx, pdf_dir=pdf_dir)
    assert [c[1] for c in ctx.fetch.calls] == [url(2)] and changed.downloaded == 1
    assert registry(ctx)[url(2)] == "h2-new"


def test_an_entry_without_a_hash_is_fetched_every_run_and_registered_by_its_file_hash(make_ctx, pdf_dir):
    ctx = make_ctx(served(1))
    plain = PdfEntry(url(1), "Doc 1", None, {})
    sync_pdfs([plain], ctx, pdf_dir=pdf_dir)
    sync_pdfs([plain], ctx, pdf_dir=pdf_dir)
    assert len(ctx.fetch.calls) == 2
    assert registry(ctx) == {url(1): hashlib.sha256(body(1)).hexdigest()}


def test_a_changed_file_drops_its_old_extraction_cache_entry(make_ctx, pdf_dir):
    ctx = make_ctx({url(1): b"version two"})
    path = local_path(url(1), pdf_dir)
    pdf_dir.mkdir()
    path.write_bytes(b"version one")
    old = ctx.cache_dir / f"{file_hash(path)}.json"
    old.write_text("{}", encoding="utf-8")
    sync_pdfs([PdfEntry(url(1), "Doc 1", "new", {})], ctx, pdf_dir=pdf_dir)
    assert path.read_bytes() == b"version two" and not old.exists()


def test_failed_download_keeps_entry_without_hash(make_ctx, pdf_dir):
    ctx = make_ctx(served(2))  # doc 1 answers 404
    result = sync_pdfs([entry(1), entry(2)], ctx, pdf_dir=pdf_dir)
    assert (result.downloaded, result.failed, result.unchanged) == (1, 1, 0)
    assert [(r["url"], r["status"]) for r in result.results] == [(url(1), "failed"), (url(2), "downloaded")]
    assert "404" in result.results[0]["error"]
    assert [e.url for e in result.entries] == [url(1), url(2)]  # the entry stays in the list
    assert url(1) not in registry(ctx) and not local_path(url(1), pdf_dir).exists()
    assert not list(pdf_dir.glob("*.tmp"))
    ctx.fetch.bodies.update(served(1))  # no hash was recorded, so the next run tries again
    retry = sync_pdfs([entry(1), entry(2)], ctx, pdf_dir=pdf_dir)
    assert (retry.downloaded, retry.unchanged) == (1, 1) and registry(ctx)[url(1)] == "h1"


def test_a_failed_download_keeps_the_file_and_hash_it_had(make_ctx, pdf_dir):
    ctx = make_ctx(served(1))
    sync_pdfs([entry(1)], ctx, pdf_dir=pdf_dir)
    ctx.fetch.bodies.clear()  # the PDF vanished upstream and its list hash changed
    result = sync_pdfs([entry(1, sha256="h1-new")], ctx, pdf_dir=pdf_dir)
    assert [r["status"] for r in result.results] == ["retained_on_error"]
    assert (result.failed, result.unchanged, result.downloaded) == (1, 1, 0)
    assert local_path(url(1), pdf_dir).read_bytes() == body(1) and registry(ctx) == {url(1): "h1"}


def test_a_download_goes_through_the_allowlist(make_ctx, pdf_dir):
    ctx = make_ctx({**served(1), "https://other.example/x.pdf": b"x"}, allowed_hosts=["docs.example.org"])
    refused = PdfEntry("https://other.example/x.pdf", "X", None, {})
    result = sync_pdfs([entry(1), refused], ctx, pdf_dir=pdf_dir)
    assert ctx.fetch.calls == [("get", url(1), PDF_DOWNLOAD_TIMEOUT)]  # no request to the refused host
    assert [r["status"] for r in result.results] == ["downloaded", "failed"]
    assert "not an allowed host" in result.results[1]["error"]


def test_a_corrupt_registry_is_refused_before_any_call(make_ctx, pdf_dir):
    ctx = make_ctx(served(1))
    ctx.state.path(REGISTRY).parent.mkdir(parents=True)
    ctx.state.path(REGISTRY).write_text("{broken", encoding="utf-8")
    with pytest.raises(CorruptStateError):
        sync_pdfs([entry(1)], ctx, pdf_dir=pdf_dir)
    assert ctx.fetch.calls == []


def test_an_empty_list_is_refused(make_ctx, pdf_dir):
    with pytest.raises(ValueError, match="no PDF entries"):
        sync_pdfs([], make_ctx(), pdf_dir=pdf_dir)


def test_a_limited_run_syncs_the_first_entries_only(make_ctx, pdf_dir):
    ctx = make_ctx(served(1, 2, 3), limit=2)
    result = sync_pdfs([entry(1), entry(2), entry(3)], ctx, pdf_dir=pdf_dir)
    assert [e.url for e in result.entries] == [url(1), url(2)] and result.check == {}
    assert not ctx.state.path("pdf_list_state.json").exists()


def test_frozen_inputs_fetch_nothing_and_write_nothing(make_ctx, pdf_dir):
    ctx = make_ctx(served(1, 2))
    sync_pdfs([entry(1), entry(2)], ctx, pdf_dir=pdf_dir)
    before = ctx.state.path(REGISTRY).read_bytes()
    local_path(url(2), pdf_dir).unlink()
    ctx.fetch.calls.clear()
    result = sync_pdfs([entry(1), entry(2)], dataclasses.replace(ctx, reuse=True), pdf_dir=pdf_dir)
    assert ctx.fetch.calls == []
    assert [r["status"] for r in result.results] == ["cached", "missing"]
    assert (result.unchanged, result.missing, result.downloaded) == (1, 1, 0)
    assert ctx.state.path(REGISTRY).read_bytes() == before
