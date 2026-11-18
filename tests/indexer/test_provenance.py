"""Provenance: /info is read once at build start; a model or dimension
the profile does not name refuses the build before any download or table; no /info records null and builds."""

import logging

import pytest
from rag_core.domain.sources import PageItem

from indexer import build


def run(settings, info, monkeypatch, tmp_path):
    monkeypatch.setattr(build.gates, "STATS_FILE", str(tmp_path / "stats.json"))
    asked = []

    class Source:
        name, state_files = "example.web", ()

        def collect(self, ctx):
            asked.append(ctx)
            raise AssertionError("collect must not run")

    embedding = type("E", (), {"info": lambda self: info})()
    with pytest.raises(Exception) as caught:
        build.build_corpus(settings, {"web": Source()}, embedding=embedding)
    return caught.value, asked


def test_a_model_or_dimension_mismatch_refuses_the_build_before_any_download_or_table(settings, monkeypatch, tmp_path):
    for info, needle in (
        ({"model": "other/model", "revision": "a" * 40, "dimension": 768}, "model 'other/model'"),
        (
            {"model": "Snowflake/snowflake-arctic-embed-m-v2.0", "revision": "a" * 40, "dimension": 1024},
            "dimension 1024",
        ),
    ):
        error, asked = run(settings, info, monkeypatch, tmp_path)
        assert isinstance(error, build.EmbeddingMismatch) and needle in str(error) and "build refused" in str(error)
        assert asked == []


def page(n):
    return PageItem(f"https://www.example.org/{n}", f"Seite {n}", {"url": f"https://www.example.org/{n}"})


def two_sources(fakes):
    return {"web": fakes.Source([page(1)]), "maps": fakes.Source([page(2)])}


def test_an_embedding_server_without_info_records_null_and_builds(settings, fakes, caplog):
    with caplog.at_level(logging.WARNING):
        result = build.build_corpus(settings, two_sources(fakes), embedding=fakes.Embedding(None))
    assert result["status"] == "success" and result["embedding"] is None
    assert "the embedding server has no /info" in caplog.text
    assert build.provenance(settings, None)["embedding_model"] is None


def test_the_generation_row_records_the_info_model_and_revision(settings, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "0123abc")
    info = {"model": "Snowflake/snowflake-arctic-embed-m-v2.0", "revision": "b" * 40, "dimension": 768}
    assert build.provenance(settings, info) == {
        "git_commit": "0123abc",
        "config_hash": settings.config_hash(),
        "embedding_model": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "embedding_revision": "b" * 40,
    }


def test_info_is_read_once_per_build(settings, fakes):
    info = {"model": "Snowflake/snowflake-arctic-embed-m-v2.0", "revision": "c" * 40, "dimension": 768}
    embedding = fakes.Embedding(info)
    result = build.build_corpus(settings, two_sources(fakes), embedding=embedding)
    assert embedding.calls == 1 and result["embedding"] == info
