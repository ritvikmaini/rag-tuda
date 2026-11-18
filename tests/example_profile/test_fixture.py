"""The example profile's connector: one document per Markdown file in the overlay's directory, titled by its first heading, its
url from the profile's base_url and the file name; no year, no date."""

from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace

import pytest
from example_profile.fixture import Fixture
from rag_core.domain.sources import PageItem

BASE = "https://www.gesetze-im-internet.de/"


def ctx(folder: Path | None, limit: int | None = None, **settings):
    return SimpleNamespace(data_dir=folder, limit=limit, settings={"min_documents": 2, "base_url": BASE, **settings})


def write(folder: Path, name: str, text: str) -> None:
    (folder / name).write_text(text, encoding="utf-8")


def test_each_markdown_file_is_one_undated_page_in_name_order(tmp_path: Path) -> None:
    write(tmp_path, "zweites.md", "# Zweites Gesetz\n\n## § 1 Zweck\n\nText zwei.\n")
    write(tmp_path, "erstes.md", "# Erstes Gesetz\n\n## § 1 Geltung\n\nText eins.\n")
    write(tmp_path, "notiz.txt", "not a document")
    result = Fixture().collect(ctx(tmp_path))
    assert [type(i) for i in result.items] == [PageItem, PageItem]
    first = result.items[0]
    assert first.url == BASE + "erstes/" and first.text.startswith("# Erstes Gesetz")
    assert dict(first.metadata) == {
        "title": "Erstes Gesetz",
        "url": BASE + "erstes/",
        "source_type": "page",
        "doc_type": "Gesetz",
        "origin": "website",
        "section": "gesetze",
        "doc_year": None,
        "doc_date": None,
        "page_number": -1,
        "doc_key": BASE + "erstes/",
    }
    assert result.expected_minimum == 2 and result.report == {"files": 2}


def test_a_limited_build_reads_the_first_files_only(tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        write(tmp_path, name + ".md", f"# {name.upper()}\n\nText.\n")
    assert [i.url for i in Fixture().collect(ctx(tmp_path, limit=2)).items] == [BASE + "a/", BASE + "b/"]


def test_without_a_directory_or_a_base_url_the_source_refuses(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"connectors\.<source>\.dir"):
        Fixture().collect(ctx(None))
    with pytest.raises(RuntimeError, match="base_url"):
        Fixture().collect(SimpleNamespace(data_dir=tmp_path, limit=None, settings={"min_documents": 0}))


def test_a_file_without_a_title_line_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "ohne.md", "Kein Titel.\n")
    with pytest.raises(ValueError, match="ohne.md"):
        Fixture().collect(ctx(tmp_path))


def test_the_connector_is_found_by_its_entry_point() -> None:
    found = {ep.name: ep for ep in entry_points(group="rag_core.connectors")}
    assert found["example.fixture"].load() is Fixture
