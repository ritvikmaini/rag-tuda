"""Schema v2 DDL: pure strings, checked without a database."""

import pytest

from indexer.index import ddl
from indexer.index import generations as gen


def test_v2_names_follow_the_base_name():
    assert ddl.v2_tables("example_g0012") == {
        "children": "example_g0012_chunks",
        "parents": "example_g0012_sections",
    }


def test_the_typed_views_list_their_columns_and_no_legacy_column():
    chunks, sections = ddl.typed_views("example", "example_g0012_chunks", "example_g0012_sections")
    assert chunks.startswith('CREATE OR REPLACE VIEW "example_chunks" AS SELECT chunk_id, section_id, document_id')
    assert sections.startswith('CREATE OR REPLACE VIEW "example_sections" AS SELECT section_id, document_id')
    assert "langchain" not in chunks + sections and "*" not in chunks + sections


def test_the_fts_language_comes_from_the_profile_and_is_a_plain_word():
    # the 'german' check by behaviour, not by reading source: the DDL uses the language it is given
    chunks = ddl.create_tables("a_sections", "a_chunks", "simple", 768)[1]
    assert "to_tsvector('simple'::regconfig, content)" in chunks and "german" not in chunks
    assert "to_tsvector('german'::regconfig, content)" in ddl.create_tables("a_sections", "a_chunks", "german", 768)[1]
    with pytest.raises(ValueError):
        ddl.create_tables("a_sections", "a_chunks", "german'); DROP TABLE x; --", 768)


def test_indexes_match_the_hnsw_parameters():
    statements = " ".join(ddl.create_indexes("a_sections", "a_chunks"))
    assert "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128)" in statements
    assert "USING gin (fts)" in statements and "(section_id)" in statements and "(document_id)" in statements


def test_required_indexes_name_their_definitions():
    names = {name for name, _, _ in ddl.REQUIRED_INDEXES["children"]}
    assert names == {
        "hnsw on embedding",
        "gin on fts",
        "primary chunk_id",
        "btree on section_id",
    }


def test_new_chunk_tables_carry_no_legacy_column():  # new generations have no langchain_* columns
    chunks = ddl.create_tables("a_sections", "a_chunks", "german", 768)[1]
    assert "langchain" not in chunks and chunks.endswith("extra jsonb NOT NULL DEFAULT '{}'::jsonb)")
    assert ddl.CHUNK_COLUMNS == tuple(c for c in ddl.TYPED_CHUNKS if c != "fts")
    assert not [n for n, _, _ in ddl.REQUIRED_INDEXES["children"] if "langchain" in n]


def test_the_cache_and_the_registry_are_created_with_the_generation_tables():
    assert gen.DDL[-2:] == (ddl.EMBEDDING_CACHE, ddl.INDEX_DOCUMENTS)
    assert "PRIMARY KEY (collection, model, revision, passage_prefix, text_sha256)" in ddl.EMBEDDING_CACHE
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
