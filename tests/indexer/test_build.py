"""The build over the connectors: each source in profile order, its PDFs extracted, every document chunked, then the gates;
the parents are handed back (no hand-off file). Fake sources and an embedding server without /info."""

import json
import logging

import pytest
from rag_core.domain.sources import CorruptStateError, PageItem, PdfItem

from indexer import build, gates


def test_a_failing_stats_write_does_not_fail_the_swapped_build(tmp_path):
    # After a successful swap, a stats write error must not turn the reply into a 500.
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    assert gates.record_build_stats(str(blocker / "state"), {"parents": 1, "children": 2, "documents": 1}) is False
    assert (
        gates.record_build_stats(str(tmp_path / "new" / "state"), {"parents": 1, "children": 2, "documents": 1}) is True
    )


def page(url):
    return PageItem(url, f"Text von {url}", {"url": url})


PDF = PdfItem(
    "/app/data/pdfs/a.pdf",
    "https://docs.example.org/a/?layout=file",
    "h1",
    {"url": "https://docs.example.org/a/", "download_hash": "d1", "download_title": "A"},
)
EXTRACTED = {"processed": 1, "cached": 0, "failed": 0, "skipped_page_limit": 0}


class Parent:
    children = ("c1", "c2")


def stub_pdfs(monkeypatch, documents=True):
    """extract_pdfs records its call; pdf_documents gives one page per item (none with documents=False), one missing."""
    calls = {}

    def extract(*, inventory, extraction, extracted_dir, skipped_log):
        calls.update(inventory=inventory, extraction=extraction)
        return EXTRACTED

    def pdf_documents(items, seen_hashes, extracted_dir, clean):
        calls["clean"] = clean
        docs = [{"page_content": "PDF", "metadata": {"file_hash": i.file_hash, "url": "pdf"}} for i in items]
        return (docs if documents else []), 1

    monkeypatch.setattr(build, "extract_pdfs", extract)
    monkeypatch.setattr(build, "pdf_documents", pdf_documents)
    return calls


def same_build(settings):
    """The profile and source set `three_sources` is built under, as the stats file names them."""
    return {"profile": settings.profile_name, "sources": gates.source_set(None, ("web", "pdfs", "maps"))}


def three_sources(fakes, web_minimum=0, pdf_minimum=0):
    return {
        "web": fakes.Source([page("web1"), page("web2")], {"crawl": "kept", "crawl_pages": 2}, web_minimum),
        "pdfs": fakes.Source([PDF], {"selected": 1}, pdf_minimum),
        "maps": fakes.Source([page("plan1")]),
    }


def test_the_documents_reach_the_chunker_in_profile_order_then_the_gates(settings, fakes, monkeypatch, caplog):
    events, chunked = [], {}
    pdfs = stub_pdfs(monkeypatch)

    def chunk(docs, **kwargs):
        events.append("chunk")
        chunked.update(docs=docs, kwargs=kwargs)
        return [Parent(), Parent()]

    refusal = gates.refusal
    monkeypatch.setattr(build, "chunk_documents", chunk)
    monkeypatch.setattr(gates, "refusal", lambda checks: events.append("gates") or refusal(checks))
    with caplog.at_level(logging.INFO):
        result = build.build_corpus(settings, three_sources(fakes), embedding=fakes.Embedding(None))

    assert events == ["chunk", "gates"]
    # The Parents come back under "units"; "parents" is their count
    assert len(result["units"]) == result["parents"] == 2
    assert [d["metadata"]["url"] for d in chunked["docs"]] == ["web1", "web2", "pdf", "plan1"]
    assert [d["source"] for d in chunked["docs"]] == [
        "example.web",
        "example.web",
        "example.pdfs",
        "example.maps",
    ]  # each document carries its source's connector
    c = settings.profile.ingest.chunking
    assert chunked["kwargs"] == {
        "mode": c.mode,
        "parent_chars": c.parent_chars,
        "children": c.children,
        "header": c.header,
        "parent_overlap": c.parent_overlap,
        "child_chars": c.child_chars,
        "child_overlap": c.child_overlap,
        "scope": settings.profile.ingest.scope,
    }
    assert pdfs["extraction"] is settings.profile.ingest.extraction and pdfs["clean"] is True
    assert pdfs["inventory"] == [
        {
            "filename": "a.pdf",
            "path": "/app/data/pdfs/a.pdf",
            "file_hash": "h1",
            "url": "https://docs.example.org/a/",
            "download_url": "https://docs.example.org/a/?layout=file",
            "download_hash": "d1",
            "download_title": "A",
        }
    ]
    assert result["sources"]["pdfs"] == {
        "selected": 1,
        "documents": 1,
        "with_documents": 1,
        "missing_cache": 1,
        "extract": EXTRACTED,
    }
    assert result["sources"]["web"] == {"crawl": "kept", "crawl_pages": 2, "documents": 2}
    assert list(result["gates"]) == ["web", "pdfs", "maps", "hosts", "size", "pages"]
    assert result["gates"]["pdfs"]["kind"] == "pdfs" and result["gates"]["maps"]["kind"] == "documents"
    assert (result["parents"], result["children"], result["reused"]) == (2, 4, False)
    assert set(result["minutes"]) == {"collect", "extract", "chunk"} and "minutes_total" in result
    finished = [r.message for r in caplog.records if r.message.startswith("[Corpus] build finished: ")]
    [summary] = [json.loads(m.split(": ", 1)[1]) for m in finished]
    assert summary.pop("minutes") >= 0
    assert summary == {"documents": {"web": 2, "pdfs": 1, "maps": 1}, "parents": 2, "children": 4}


def test_a_limit_build_skips_the_gates(settings, fakes, monkeypatch):
    stub_pdfs(monkeypatch, documents=False)
    sources = three_sources(fakes, web_minimum=200, pdf_minimum=1)
    result = build.build_corpus(settings, sources, limit=5, embedding=fakes.Embedding(None))
    assert result["status"] == "success" and len(result["units"]) == result["parents"]
    assert not result["gates"]["web"]["passed"] and not result["gates"]["pdfs"]["passed"]
    assert [s.asked[0].limit for s in sources.values()] == [5, 5, 5]


def test_a_limit_build_does_not_read_the_live_manifest(settings, fakes, monkeypatch, tmp_path):
    stub_pdfs(monkeypatch)
    fakes.stats.write_text(json.dumps({"parents": 100, "generation": 3, **same_build(settings)}), encoding="utf-8")
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "3.json").write_text("{broken", encoding="utf-8")
    assert build.build_corpus(settings, three_sources(fakes), limit=5, embedding=fakes.Embedding(None))["units"]
    with pytest.raises(CorruptStateError, match="3.json"):
        build.build_corpus(settings, three_sources(fakes), embedding=fakes.Embedding(None))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("web", "only 2 web documents (minimum 200); swap refused"),
        ("pdfs", "only 0 pdfs PDFs produced documents (minimum 1); swap refused"),
        ("size", "corpus shrank to 1 parents (last build 100); swap refused"),
    ],
)
def test_each_gate_refusal_says_swap_refused(settings, fakes, monkeypatch, case, message):
    stub_pdfs(monkeypatch, documents=case != "pdfs")
    if case == "size":
        fakes.stats.write_text(json.dumps({"parents": 100, **same_build(settings)}), encoding="utf-8")
        monkeypatch.setattr(build, "chunk_documents", lambda docs, **kwargs: [Parent()])
    sources = three_sources(fakes, web_minimum=200 if case == "web" else 0, pdf_minimum=1)
    with pytest.raises(RuntimeError) as refused:
        build.build_corpus(settings, sources, embedding=fakes.Embedding(None))
    assert str(refused.value) == message


@pytest.mark.parametrize(
    "last",
    [
        {"profile": "tu-informatik"},  # another profile, the same source set
        {"sources": {"set": "full", "names": ["maps", "pdfs", "web"]}},  # the full set after a core build
        {"sources": {"set": "core", "names": ["docs", "maps", "pdfs", "web"]}},  # a registered folder came or went
        {"profile": None, "sources": None},  # stats written before they named either
    ],
)
def test_a_smaller_build_of_another_profile_or_source_set_is_not_size_gated(settings, fakes, monkeypatch, last):
    stub_pdfs(monkeypatch)
    monkeypatch.setattr(build, "chunk_documents", lambda docs, **kwargs: [Parent()])
    fakes.stats.write_text(json.dumps({"parents": 100, **same_build(settings), **last}), encoding="utf-8")
    result = build.build_corpus(settings, three_sources(fakes), embedding=fakes.Embedding(None))
    assert result["gates"]["size"]["passed"] and result["gates"]["size"]["skipped"]
    assert result["profile"] == settings.profile_name
    assert result["source_set"] == {"set": "core", "names": ["maps", "pdfs", "web"]}


def test_the_source_set_asked_for_is_the_one_the_gate_compares(settings, fakes, monkeypatch):
    stub_pdfs(monkeypatch)
    monkeypatch.setattr(build, "chunk_documents", lambda docs, **kwargs: [Parent()])
    fakes.stats.write_text(json.dumps({"parents": 100, **same_build(settings)}), encoding="utf-8")
    result = build.build_corpus(
        settings, three_sources(fakes), options={"sources": "full"}, embedding=fakes.Embedding(None)
    )
    assert result["gates"]["size"]["skipped"] and result["source_set"]["set"] == "full"
