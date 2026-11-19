import json
from pathlib import Path

import pytest

from rag_tuda import sources

SAMPLE = Path(__file__).parent / "fixtures" / "sample_docs"


def raw(root) -> str:
    return (root / ".rag" / "sources.json").read_text()


def test_fixture_folder_has_two_markdown_files():
    assert sorted(p.name for p in SAMPLE.glob("*.md")) == ["hausordnung.md", "waschkeller.md"]


def test_folder(tmp_path):
    s = sources.register(tmp_path, SAMPLE)
    assert (s.name, s.target) == ("sample_docs", str(SAMPLE.resolve()))
    assert sources.load(tmp_path) == [s]


def test_relative_folder_resolves_against_cwd(tmp_path, monkeypatch):
    (tmp_path / "notes").mkdir()
    monkeypatch.chdir(tmp_path)
    assert sources.register(tmp_path, "notes").target == str((tmp_path / "notes").resolve())


@pytest.mark.parametrize("target", ["", "ftp://example.org/x", "https://example.org/", "prices.csv"])
def test_rejected_targets(tmp_path, target):
    with pytest.raises(ValueError):
        sources.register(tmp_path, target)
    assert not (tmp_path / ".rag" / "sources.json").exists()


def test_rejects_missing_and_unsupported_paths(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "prices.csv").write_text("a,b\n1,2\n")  # a file is not a folder
    for bad in (tmp_path / "nope", tmp_path / "a.txt", tmp_path / "prices.csv"):
        with pytest.raises(ValueError):
            sources.register(tmp_path, bad)


def test_readd_is_idempotent_and_keeps_order(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    sources.register(tmp_path, SAMPLE)
    sources.register(tmp_path, other)
    before = raw(tmp_path)
    sources.register(tmp_path, SAMPLE)
    sources.register(tmp_path, str(SAMPLE) + "/")
    assert raw(tmp_path) == before
    assert [s.name for s in sources.load(tmp_path)] == ["sample_docs", "other"]


def test_default_name_collision_gets_a_suffix(tmp_path):
    a, b, c = (tmp_path / x / "docs" for x in "abc")
    for d in (a, b, c):
        d.mkdir(parents=True)
    names = [sources.register(tmp_path, d).name for d in (a, b, c)]
    assert names == ["docs", "docs-2", "docs-3"]
    assert sources.register(tmp_path, b).name == "docs-2"  # re-adding keeps its name


def test_explicit_name_collision_with_other_target_fails(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    sources.register(tmp_path, SAMPLE, name="handbook")
    with pytest.raises(ValueError, match="handbook"):
        sources.register(tmp_path, other, name="handbook")
    assert len(sources.load(tmp_path)) == 1


def test_same_target_new_name_renames(tmp_path):
    sources.register(tmp_path, SAMPLE)
    sources.register(tmp_path, SAMPLE, name="house")
    assert [s.name for s in sources.load(tmp_path)] == ["house"]
    sources.register(tmp_path, SAMPLE)  # no name given: keeps the chosen one
    assert [s.name for s in sources.load(tmp_path)] == ["house"]


@pytest.mark.parametrize("name", ["", "has space", "a/b", "x" * 65])
def test_bad_names(tmp_path, name):
    with pytest.raises(ValueError):
        sources.register(tmp_path, SAMPLE, name=name)


def test_load_without_file_is_empty(tmp_path):
    assert sources.load(tmp_path) == []


def test_file_is_stable_json(tmp_path):
    sources.register(tmp_path, SAMPLE)
    text = raw(tmp_path)
    assert text.endswith("\n") and json.loads(text) == {
        "version": 1,
        "sources": [{"name": "sample_docs", "target": str(SAMPLE.resolve())}],
    }


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '{"version": 1}',
        '{"sources": ["a"]}',
        '{"sources": [{"name": "a"}]}',
        '{"sources": [{"name": 1, "target": "/x"}]}',
    ],
)
def test_a_hand_edited_file_is_a_clear_error_not_a_key_error(tmp_path, content):
    (tmp_path / ".rag").mkdir()
    (tmp_path / ".rag" / "sources.json").write_text(content)
    with pytest.raises(ValueError, match=r"sources\.json.*(sources\[\d\]|not a sources file)"):
        sources.load(tmp_path)
    with pytest.raises(ValueError, match=r"sources\.json"):
        sources.register(tmp_path, SAMPLE)  # an add does not overwrite a file it cannot read
