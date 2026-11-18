"""Shared fixtures of the indexer tests: the example profile with the harness overlay, and fakes of a
connector and of the embedding server, so a build runs without fetching, embedding or touching a GPU."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from rag_core.domain.sources import SourceResult

from indexer import build
from indexer.settings import load_settings

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "profiles" / "example" / "profile.yaml"
CI = str(ROOT / "deploy" / "env" / "ci-example.yaml")
# the sources the build tests feed with fakes, listed after the example profile's own: pages, PDFs and a small map source
SOURCES = {
    "web": {"connector": "example.web", "audience": "public", "min_documents": 200},
    "pdfs": {"connector": "example.pdfs", "audience": "internal", "min_doc_ratio": 0.9},
    "maps": {"connector": "example.maps", "audience": "public", "min_documents": 1},
}


@pytest.fixture
def profile(tmp_path):
    """The example profile with SOURCES added to its ingest.sources, written to tmp_path."""
    spec = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    spec["ingest"]["sources"] |= SOURCES
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    return str(path)


@pytest.fixture
def settings(tmp_path, profile):
    return load_settings({"RAG_PROFILE": profile, "RAG_OVERLAY": CI}, secrets_dir=str(tmp_path / "secrets"))


@pytest.fixture
def db_settings(tmp_path, profile):
    """The same settings with the database secrets set."""
    environ = {
        "RAG_PROFILE": profile,
        "RAG_OVERLAY": CI,
        "RAG__DATABASE__NAME": "vector_db",
        "RAG__DATABASE__USER": "rag_indexer",
        "RAG__DATABASE__PASSWORD": "pw",
    }
    return load_settings(environ, secrets_dir=str(tmp_path / "secrets"))


class FakeSource:
    """Returns its items; records every context it is asked with."""

    def __init__(self, items=(), report=None, minimum=0, state_files=()):
        self.items, self.report, self.minimum, self.state_files = list(items), report or {}, minimum, tuple(state_files)
        self.asked = []

    def collect(self, ctx):
        self.asked.append(ctx)
        return SourceResult(self.items, self.report, self.minimum)


class FakeEmbedding:
    """The embedding server's GET /info; counts the calls."""

    def __init__(self, info=None):
        self._info, self.calls = info, 0

    def info(self):
        self.calls += 1
        return self._info


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """The build's stats file moved to tmp_path: absent until a test writes it."""
    stats = tmp_path / "corpus_build_stats.json"
    monkeypatch.setattr(build.gates, "STATS_FILE", str(stats))
    return SimpleNamespace(Source=FakeSource, Embedding=FakeEmbedding, stats=stats)
