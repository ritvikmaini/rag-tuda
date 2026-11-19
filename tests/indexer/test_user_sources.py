"""The folders `rag add` registers (.rag/sources.json) are sources beside the profile's: a folder source each, with the
own-documents origin."""

import json

import pytest
from rag_core.settings import SettingsError

from indexer import build
from indexer.connectors.folder import Folder
from indexer.settings import load_settings

from .conftest import CI


@pytest.fixture
def load(tmp_path, profile, moved):
    def load(content=None):
        listing = tmp_path / "sources.json"
        if content is not None:
            listing.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        environ = {"RAG_PROFILE": profile, "RAG_OVERLAY": CI, "RAG__PATHS__USER_SOURCES": str(listing)}
        environ |= {f"RAG__PATHS__{k.upper()}": v for k, v in moved.items()}
        return load_settings(environ, secrets_dir=str(tmp_path / "secrets"))

    return load


def listing(*folders):
    return {"version": 1, "sources": [{"name": name, "target": str(path)} for name, path in folders]}


def test_a_registered_folder_is_a_folder_source_after_the_profiles(load, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    settings = load(listing(("house", docs)))
    names = list(settings.profile.ingest.sources)
    assert names[-1] == "house" and {"gesetze", "web"} <= set(names)
    entry = settings.profile.ingest.sources["house"]
    assert (entry.connector, entry.audience) == ("folder", "public") and "base_url" not in (entry.model_extra or {})
    assert settings.endpoints["house"].dir == str(docs)


def test_the_next_build_indexes_an_added_folder(load, tmp_path, fakes):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "hausordnung.md").write_text("# Hausordnung\n\nRuhezeiten gelten ab 22 Uhr.\n", encoding="utf-8")
    settings = load(listing(("house", docs)))
    sources = {"house": Folder()}
    result = build.build_corpus(settings, sources, embedding=fakes.Embedding(None))
    assert result["sources"]["house"]["documents"] == 1
    (parent,) = result["units"]
    assert parent.metadata["url"] == "folder:house/hausordnung.md" and parent.metadata["origin"] == "documents"


def test_the_same_file_name_in_two_folders_stays_two_documents(load, tmp_path, fakes):
    folders = []
    for name, text in (("a", "# Erstes\n\nText der ersten Mappe.\n"), ("b", "# Zweites\n\nText der zweiten Mappe.\n")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "README.md").write_text(text, encoding="utf-8")
        folders.append((name, folder))
    settings = load(listing(*folders))
    sources = {"a": Folder(), "b": Folder()}
    result = build.build_corpus(settings, sources, embedding=fakes.Embedding(None))
    urls = sorted(p.metadata["url"] for p in result["units"])
    assert urls == ["folder:a/README.md", "folder:b/README.md"] and result["parents"] == 2


@pytest.mark.parametrize("content", [None, "", {"version": 1, "sources": []}])
def test_a_missing_or_empty_listing_changes_nothing(load, content):
    base = load()
    again = load(content)
    assert again.profile == base.profile and again.endpoints == base.endpoints
    assert again.config_hash() == base.config_hash()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "sources.json"),
        ({"version": 1, "sources": [{"name": "web", "target": "/x"}]}, "web"),
        ({"version": 1, "sources": [{"name": "a"}]}, r"sources\[0\].*\"target\""),
        ({"version": 1, "sources": [{"name": "ok", "target": "/x"}, {"target": "/y"}]}, r"sources\[1\].*\"name\""),
        ({"version": 1, "sources": [{"name": "a", "target": 3}]}, r"sources\[0\].*\"target\""),
        ({"version": 1, "sources": ["a"]}, r"sources\[0\]"),
        ({"version": 1}, "sources"),
    ],
)
def test_a_listing_that_cannot_be_used_is_refused_naming_the_file(load, content, message):
    with pytest.raises(SettingsError, match=message):
        load(content)
