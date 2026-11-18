"""The v2 writer's pure part: the sections and chunks rows equal the hand-off's,
the typed columns follow the column rules, and the embedding batch size stays 128."""

import datetime
import hashlib
import json
import re

import pytest

from indexer.chunking.units import Child, Parent
from indexer.index import cache, ddl, writer

KEY = ("example", "m", "r", "")


def connector_of(meta):
    """The connector the original writer derived from the metadata (the build now tags each document with it)."""
    if meta.get("source_type") == "map":
        return "example.maps"
    return "example.pdfs" if meta.get("origin") == "pdf" else "example.web"


def parent(name, texts, prefix="T: ", **meta):
    base = {
        "section_id": name,
        "part_index": 0,
        "filename": "Doc",
        "title": "Doc",
        "origin": "website",
        "source_type": "page",
        "url": "https://docs.example.org/doc",
        "doc_key": "https://docs.example.org/doc",
    }
    base.update(meta)
    return Parent(f"text of {name}", base, tuple(Child(prefix + t) for t in texts), prefix, connector_of(base))


def sections_of(sections):
    return [dict(zip(ddl.SECTION_COLUMNS, row, strict=True)) for row in sections]


def chunks_of(chunks):
    return [dict(zip(ddl.CHUNK_COLUMNS, chunk.values("v"), strict=True)) for chunk in chunks]


def names(sections):
    """{section_id: the name parent() gave it}: parent(<name>, …) writes 'text of <name>' (the ids are hashes)."""
    return {s["section_id"]: s["content"].removeprefix("text of ") for s in sections_of(sections)}


def d20(*parts):  # the test's own oracle of the id rule
    return hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()[:32]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_ids_are_d20s_hashes_and_the_metadata_names_the_section(settings):
    sections, chunks = writer.rows([parent("p1", ["x", "y"], part_index=2)], settings.profile)
    (s,) = sections_of(sections)
    document = d20("example.web", "https://docs.example.org/doc")
    assert s["document_id"] == document == writer.document_id("example.web", "https://docs.example.org/doc")
    assert s["section_id"] == d20(document, 2, sha("text of p1"))
    assert json.loads(s["metadata"])["section_id"] == s["section_id"]
    rows = chunks_of(chunks)
    assert [c["chunk_id"] for c in rows] == [d20(s["section_id"], 0, sha("T: x")), d20(s["section_id"], 1, sha("T: y"))]
    assert all(re.fullmatch("[0-9a-f]{32}", i) for i in (document, s["section_id"], *[c["chunk_id"] for c in rows]))


def test_the_ids_ignore_the_run_and_the_input_ids(settings):
    def ids(**meta):
        sections, chunks = writer.rows([parent("p1", ["x", "y"], **meta)], settings.profile)
        return [s["section_id"] for s in sections_of(sections)], [c["chunk_id"] for c in chunks_of(chunks)]

    assert ids(scrapedAt="eins") == ids(scrapedAt="zwei", section_id="another-random-id")


def test_a_changed_text_changes_only_the_ids_that_contain_it(settings):
    def ids(first_child, second_text):
        second = parent("p2", ["z"], part_index=1)
        parents = [
            parent("p1", [first_child, "y"]),
            Parent(second_text, second.metadata, second.children, second.prefix, second.source),
        ]
        sections, chunks = writer.rows(parents, settings.profile)
        return [s["section_id"] for s in sections_of(sections)], [c["chunk_id"] for c in chunks_of(chunks)]

    (s1, c1), (s2, c2) = ids("x", "text of p2"), ids("x2", "text of p2, changed")
    assert s1[0] == s2[0] and c1[0] != c2[0] and c1[1] == c2[1]  # a child's text: that chunk only
    assert s1[1] != s2[1] and c1[2] != c2[2]  # a section's text: the section and its chunks


def test_the_same_key_in_two_sources_gives_two_documents(settings):
    web = parent("w", ["x"])
    plan = parent("s", ["x"], prefix="", source_type="map")  # the same doc_key as the web page
    sections, _ = writer.rows([web, plan], settings.profile)
    assert len({s["document_id"] for s in sections_of(sections)}) == 2


def test_rows_follow_the_handoff_order(settings):
    parents = [
        parent("a1", ["x", "y"], filename="A"),
        parent("b1", ["z"], filename="B"),
        parent("a2", ["w"], filename="A"),
    ]
    sections, chunks = writer.rows(parents, settings.profile)
    assert isinstance(sections, list) and isinstance(chunks, list)
    assert list(names(sections).values()) == ["a1", "a2", "b1"]  # grouped by filename, first seen
    assert [(names(sections)[c["section_id"]], c["content"]) for c in chunks_of(chunks)] == [
        ("a1", "T: x"),
        ("a1", "T: y"),
        ("a2", "T: w"),
        ("b1", "T: z"),
    ]


def test_a_parent_without_filename_falls_in_the_unknown_group(settings):
    parents = [parent("a1", ["x"], filename="A"), parent("u1", ["y"]), parent("a2", ["z"], filename="A")]
    del parents[1].metadata["filename"]
    sections, _ = writer.rows(parents, settings.profile)
    rows = sections_of(sections)
    assert list(names(sections).values()) == ["a1", "a2", "u1"]
    assert json.loads(rows[2]["metadata"])["source"] == "unknown"


def test_children_dedup_per_section_after_numbering(settings):
    parents = [parent("p1", ["x", "x\x00", "y"]), parent("p2", ["x"])]
    sections, chunks = writer.rows(parents, settings.profile)
    rows = chunks_of(chunks)
    assert [(names(sections)[c["section_id"]], c["content"]) for c in rows] == [
        ("p1", "T: x"),
        ("p1", "T: y"),
        ("p2", "T: x"),
    ]


def test_a_duplicate_unique_id_fails(settings):
    with pytest.raises(ValueError, match="Duplicate unique_id values found: \\['[0-9a-f]{32}_0'\\]"):
        writer.rows([parent("p1", ["x"]), parent("p1", ["y"])], settings.profile)


def test_unique_ids_are_numbered_before_the_dedup(settings):
    # the second p1's "x" is dropped as a duplicate but still counted, so its "z" is p1_1 like the first p1's "y"
    # (numbering after the dedup would give p1_0)
    with pytest.raises(ValueError, match="Duplicate unique_id values found: \\['[0-9a-f]{32}_1'\\]"):
        writer.rows([parent("p1", ["x", "y"]), parent("p1", ["x", "z"])], settings.profile)


def test_section_metadata_is_the_handoff_round_trip(settings):
    first = parent(
        "p1", ["x"], filename="Grp", folder_iid="F\x00", tags=("a", "b\x00"), nested={1: "eins"}, note="ä\x00ö"
    )
    second = parent("p2", ["y"], filename="Grp", folder_iid="other")
    second = Parent("Inhalt\x00 mit Umlaut ü", second.metadata, second.children, second.prefix, second.source)
    sections, _ = writer.rows([first, second], settings.profile)
    rows = sections_of(sections)
    expected_iid = rows[0]["section_id"]
    expected = {
        "scraper": {"runDuration": "", "errors": ""},
        **first.metadata,
        "section_id": expected_iid,
        "tags": ["a", "b"],
        "nested": {"1": "eins"},
        "note": "äö",
        "folder_iid": "F",
        "source": "Grp",
        "file-folder": "F",
    }
    assert json.loads(rows[0]["metadata"]) == expected
    assert json.loads(rows[1]["metadata"])["file-folder"] == "F"  # the group's first parent's folder_iid
    assert rows[1]["content"] == "Inhalt mit Umlaut ü"
    assert rows[0]["metadata"] == json.dumps(expected)  # the legacy writer's json.dumps text, key order included


def test_the_typed_columns_follow_the_interfaces_rule(settings):
    pdf = parent(
        "pdf",
        ["x"],
        prefix="Dienstanweisung · Stand 2023: ",
        origin="pdf",
        source_type="pdf",
        doc_key="https://docs.example.org/d.pdf",
        url="https://docs.example.org/page",
        page_number=3,
        doc_year=2023,
        doc_date=None,
        title="Dienstanweisung",
        doc_type="Dienstanweisung",
        part_index=4,
    )
    web = parent("web", ["x"], page_number=-1, doc_year=None, doc_type="Webseite", doc_key="")
    news = parent("news", ["x"], source_type="news", doc_date="2024-05-03", doc_year=2024, page_number=-1)
    plan = parent(
        "plan",
        ["x"],
        prefix="",
        source_type="map",
        doc_key="https://maps.example.org/#uid=1",
        url="https://maps.example.org/#uid=1",
        title="Rathaus",
        doc_type="Karte",
    )
    year = parent("year", ["x"], doc_year="2024", doc_date="03.05.2024", page_number="2")
    sections, chunks = writer.rows([pdf, web, news, plan, year], settings.profile)
    derived = (
        "document_id",
        "collection",
        "source",
        "audience",
        "doc_type",
        "year",
        "date",
        "url",
        "page",
        "title",
        "header",
        "part_index",
        "extra",
    )
    got = [tuple(s[k] for k in derived) for s in sections_of(sections)]
    page, doc = "https://docs.example.org/doc", "Doc"
    web_id = writer.document_id("example.web", page)
    assert got == [
        (
            writer.document_id("example.pdfs", "https://docs.example.org/d.pdf"),
            "rag",
            "example.pdfs",
            "internal",
            "Dienstanweisung",
            2023,
            None,
            "https://docs.example.org/page",
            3,
            "Dienstanweisung",
            "Dienstanweisung · Stand 2023: ",
            4,
            "{}",
        ),
        (web_id, "rag", "example.web", "public", "Webseite", None, None, page, None, doc, "T: ", 0, "{}"),
        (
            web_id,
            "rag",
            "example.web",
            "public",
            None,
            2024,
            datetime.date(2024, 5, 3),
            page,
            None,
            doc,
            "T: ",
            0,
            "{}",
        ),
        (
            writer.document_id("example.maps", "https://maps.example.org/#uid=1"),
            "rag",
            "example.maps",
            "public",
            "Karte",
            None,
            None,
            "https://maps.example.org/#uid=1",
            None,
            "Rathaus",
            None,
            0,
            "{}",
        ),
        (web_id, "rag", "example.web", "public", None, 2024, None, page, 2, doc, "T: ", 0, "{}"),
    ]
    by_section = {s["section_id"]: s for s in sections_of(sections)}
    copied = ddl.CHUNK_COLUMNS[2:13]  # document_id .. header: copied from the section
    for chunk in chunks_of(chunks):
        assert {k: chunk[k] for k in copied} == {k: by_section[chunk["section_id"]][k] for k in copied}
        assert chunk["extra"] == "{}"


class FakeConn:
    """Records the statements and the COPY rows of a direct psycopg connection."""

    def __init__(self):
        self.log = []

    def execute(self, statement):
        self.log.append(("execute", statement))

    def cursor(self):
        return self

    def copy(self, statement):
        self.log.append(("copy", statement))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write_row(self, row):
        self.log.append(("row", row))


def test_write_v2_takes_its_vectors_from_the_cache_and_writes_vector_text(settings, monkeypatch):
    filled, taken = [], []

    def fill(conn, cache_key, generation, embed, texts):
        filled.append((cache_key, generation, embed, list(texts)))
        return {"cached": 297, "computed": 3}

    def vector_of(text):  # the cache's vector of a text: it depends on the text alone
        return f"[{int(cache.sha(text)[:6], 16)}.0, 0.25]"

    def take(conn, cache_key, generation, texts):
        taken.append(list(texts))
        return {cache.sha(t): vector_of(t) for t in texts}

    monkeypatch.setattr(writer.cache, "fill", fill)
    monkeypatch.setattr(writer.cache, "take", take)
    parents = [parent(f"p{i:03d}", [f"c{j}" for j in range(3)]) for i in range(100)]
    conn = FakeConn()
    tables = {"children": "example_g0009_chunks", "parents": "example_g0009_sections"}
    written = writer.write_v2(conn, "embed", tables, parents, settings.profile, generation=9, cache_key=KEY)
    assert written == {"children": 300, "parents": 100, "embeddings": {"cached": 297, "computed": 3}}
    assert filled == [(KEY, 9, "embed", [f"T: c{j}" for _ in range(100) for j in range(3)])]
    assert [len(t) for t in taken] == [128, 128, 44]
    kinds = [kind for kind, _ in conn.log]
    assert kinds[:2] == ["execute", "execute"] and kinds[-4:] == ["execute"] * 4
    assert [s for k, s in conn.log if k == "execute"] == [
        *ddl.create_tables(tables["parents"], tables["children"], "german", 768),
        *ddl.create_indexes(tables["parents"], tables["children"]),
    ]
    copies = [s for k, s in conn.log if k == "copy"]
    assert copies[0] == f'COPY "example_g0009_sections" ({", ".join(ddl.SECTION_COLUMNS)}) FROM STDIN'
    assert copies[1:] == [f'COPY "example_g0009_chunks" ({", ".join(ddl.CHUNK_COLUMNS)}) FROM STDIN'] * 3
    chunk_rows, into = [], None
    for kind, item in conn.log:
        into = item if kind == "copy" else into
        if kind == "row" and "_chunks" in into:
            chunk_rows.append(item)
    assert len(chunk_rows) == 300
    content, embedding = ddl.CHUNK_COLUMNS.index("content"), ddl.CHUNK_COLUMNS.index("embedding")
    assert all(r[embedding] == vector_of(r[content]) for r in chunk_rows)  # each chunk holds its own text's vector
    assert len({r[embedding] for r in chunk_rows}) == 3  # three distinct texts, three distinct vectors


@pytest.mark.parametrize(
    "connector, audience",
    [("example.web", "public"), ("example.pdfs", "internal"), ("example.maps", "public")],
)
def test_a_section_takes_the_source_and_audience_of_its_connector(settings, connector, audience):
    meta = {
        "url": "https://x.example/",
        "title": "X",
        "source_type": "pdf",
        "origin": "pdf",
    }  # the metadata does not decide
    parent = Parent(
        "Text.",
        {**meta, "section_id": "p1", "part_index": 0, "filename": "X"},
        (Child("X: Text."),),
        "X: ",
        connector,
    )
    sections, _ = writer.rows([parent], settings.profile)
    row = dict(zip(ddl.SECTION_COLUMNS, sections[0]))
    assert (row["source"], row["audience"]) == (connector, audience)


def test_a_section_without_a_listed_connector_is_refused(settings):
    parent = Parent(
        "Text.",
        {"url": "https://x.example/", "section_id": "p1", "part_index": 0, "filename": "X"},
        (Child("Text."),),
        "",
        "example.unlisted",
    )
    with pytest.raises(ValueError, match="example.unlisted"):
        writer.rows([parent], settings.profile)
