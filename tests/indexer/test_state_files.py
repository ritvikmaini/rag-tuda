"""The safety-net state files are written atomically, and a file that exists but cannot
be read refuses the build instead of being read as empty. The registry's store is tested in tests/indexer/test_sources.py,
the PDF list state with the connectors."""

import dataclasses
import json

import pytest
from rag_core.domain.sources import CorruptStateError

from indexer import build, gates
from indexer.sources import JsonStateStore


def test_missing_stats_mean_no_size_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "STATS_FILE", str(tmp_path / "corpus_build_stats.json"))
    assert gates.previous_stats() == {}


def test_corrupt_stats_refuse(tmp_path, monkeypatch):
    path = tmp_path / "corpus_build_stats.json"
    monkeypatch.setattr(gates, "STATS_FILE", str(path))
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(CorruptStateError, match="corpus_build_stats.json"):
        gates.previous_stats()
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(CorruptStateError, match="JSON object"):
        gates.previous_stats()


def test_stats_are_written_atomically(tmp_path, monkeypatch):
    path = tmp_path / "corpus_build_stats.json"
    monkeypatch.setattr(gates, "STATS_FILE", str(path))
    assert gates.record_build_stats({"parents": 3, "children": 9, "documents": {"web_pages": 1}}) is True
    stats = json.loads(path.read_text(encoding="utf-8"))
    assert stats["parents"] == 3 and stats["children"] == 9 and "built_at" in stats
    assert sorted(p.name for p in tmp_path.iterdir()) == ["corpus_build_stats.json"]


def test_build_corpus_refuses_before_any_download_when_the_stats_file_is_corrupt(settings, fakes):
    fakes.stats.write_text("{broken", encoding="utf-8")
    source, embedding = fakes.Source(), fakes.Embedding(None)
    with pytest.raises(CorruptStateError, match="corpus_build_stats.json"):
        build.build_corpus(settings, {"web": source}, embedding=embedding)
    assert source.asked == [] and embedding.calls == 0


def test_build_corpus_refuses_before_any_download_when_the_registry_is_corrupt(settings, fakes, tmp_path, monkeypatch):
    state = tmp_path / "vector_database_contents"
    state.mkdir()
    (state / "pdf_downloaded_urls.json").write_text("{broken", encoding="utf-8")
    context = build.context
    monkeypatch.setattr(
        build, "context", lambda *args: dataclasses.replace(context(*args), state=JsonStateStore(state))
    )
    web, pdfs = fakes.Source(), fakes.Source(state_files=["pdf_downloaded_urls.json"])
    with pytest.raises(
        CorruptStateError,
        match=r"pdf_downloaded_urls.json exists but cannot be read .*fix or remove it before the next build",
    ):
        build.build_corpus(settings, {"web": web, "pdfs": pdfs}, embedding=fakes.Embedding(None))
    assert web.asked == [] and pdfs.asked == []
