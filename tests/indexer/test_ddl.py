"""Schema v2 DDL: pure strings, checked without a database."""

import pytest

from indexer.index import ddl
from indexer.index import generations as gen


def test_v2_names_follow_the_base_name():
    assert ddl.v2_tables("example_g0012") == {
        "children": "example_g0012_chunks",
        "parents": "example_g0012_sections",
    }


def test_the_typed_views_list_their_columns():
    chunks, sections = ddl.typed_views("example", "example_g0012_chunks", "example_g0012_sections")
    assert chunks.startswith('CREATE OR REPLACE VIEW "example_chunks" AS SELECT chunk_id, section_id, document_id')
    assert sections.startswith('CREATE OR REPLACE VIEW "example_sections" AS SELECT section_id, document_id')
    assert "*" not in chunks + sections


def _table_columns(create_table: str) -> list[str]:
    body = create_table[create_table.index("(") + 1 : create_table.rindex(")")]
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(body):  # split at the commas outside parentheses: text[], vector(768), CHECK (...)
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:])
    return [p.split()[0] for p in parts]


def test_the_tables_and_the_views_have_exactly_the_expected_columns():
    sections, chunks = ddl.create_tables("a_sections", "a_chunks", 768)
    assert _table_columns(sections) == list(ddl.TYPED_SECTIONS)
    assert _table_columns(chunks) == list(ddl.TYPED_CHUNKS)
    chunk_view, section_view = ddl.typed_views("example", "a_chunks", "a_sections")
    assert chunk_view.split(" AS SELECT ")[1].split(" FROM ")[0].split(", ") == list(ddl.TYPED_CHUNKS)
    assert section_view.split(" AS SELECT ")[1].split(" FROM ")[0].split(", ") == list(ddl.TYPED_SECTIONS)


def test_the_document_metadata_is_typed_on_both_tables_and_both_views():
    sections, chunks = ddl.create_tables("a_sections", "a_chunks", 768)
    for column in (
        "regulation text",
        "degree text[]",
        "programme text[]",
        "po_version integer[]",
        "archived boolean NOT NULL DEFAULT false",
        "valid_from date",
        "lang text NOT NULL",
        "twin_key text",
    ):
        assert column in sections and column in chunks, column
    assert "CHECK (audience = 'public')" in sections
    views = " ".join(ddl.typed_views("example", "a_chunks", "a_sections"))
    assert all(c in views for c in ddl.DOCUMENT_COLUMNS) and "fts_de, fts_en" in views


def test_the_full_text_columns_are_generated_and_a_chunk_is_written_without_them():
    chunks = ddl.create_tables("a_sections", "a_chunks", 768)[1]
    assert "fts_de tsvector NOT NULL GENERATED ALWAYS AS (to_tsvector('german'::regconfig, content)) STORED" in chunks
    assert "CASE WHEN lang = 'en' THEN to_tsvector('english'::regconfig, content) END" in chunks
    assert not set(ddl.CHUNK_COLUMNS) & {"fts_de", "fts_en", "fts"}
    assert ddl.CHUNK_COLUMNS == tuple(c for c in ddl.TYPED_CHUNKS if c not in ("fts_de", "fts_en"))
    with pytest.raises(ValueError):
        ddl.create_tables("a_sections", "a_chunks", "768")


def test_the_indexes_are_the_full_text_ones_and_there_is_no_vector_index():
    statements = " ".join(ddl.create_indexes("a_sections", "a_chunks"))
    assert "USING gin (fts_de)" in statements and "USING gin (fts_en)" in statements
    assert "(section_id)" in statements and "(document_id)" in statements
    assert "hnsw" not in statements.lower() and "ivfflat" not in statements.lower()


def test_required_indexes_name_their_definitions():
    names = {name for name, _, _ in ddl.REQUIRED_INDEXES["children"]}
    assert names == {"gin on fts_de", "gin on fts_en", "primary chunk_id", "btree on section_id"}


def test_the_chunk_table_ends_with_the_extra_column():
    chunks = ddl.create_tables("a_sections", "a_chunks", 768)[1]
    assert chunks.endswith("extra jsonb NOT NULL DEFAULT '{}'::jsonb)")


def test_the_cache_and_the_registry_are_created_with_the_generation_tables():
    assert gen.DDL[-2:] == (ddl.EMBEDDING_CACHE, ddl.INDEX_DOCUMENTS)
    assert "PRIMARY KEY (collection, model, revision, passage_prefix, dimensions, text_sha256)" in ddl.EMBEDDING_CACHE
    assert "dimensions integer NOT NULL" in ddl.EMBEDDING_CACHE
    assert (
        "embedding vector NOT NULL" in ddl.EMBEDDING_CACHE and "last_generation integer NOT NULL" in ddl.EMBEDDING_CACHE
    )
    assert "PRIMARY KEY (collection, document_id)" in ddl.INDEX_DOCUMENTS  # generation numbers count per collection
    assert "content_sha256 text NOT NULL" in ddl.INDEX_DOCUMENTS


def test_the_reader_grants_name_a_collections_two_views_and_the_registry():
    assert ddl.READER == "rag_app"
    assert ddl.view_grant("example") == 'GRANT SELECT ON "example_chunks", "example_sections" TO "rag_app"'
    assert ddl.REGISTRY_GRANT == 'GRANT SELECT ON "index_generations" TO "rag_app"'
    with pytest.raises(ValueError):
        ddl.view_grant("Robert'); DROP TABLE x;--")
