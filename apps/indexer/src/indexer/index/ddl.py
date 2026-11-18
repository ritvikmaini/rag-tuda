"""Schema v2: the generation tables, their indexes, the two typed
views and the required indexes. Only strings; generations.py runs them."""

import re

_NAME = re.compile(r"[a-z][a-z0-9_]{0,62}")


def ident(name):
    """A table or view name as a quoted SQL identifier; only lower-case names of letters, digits and _."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"not a valid relation name: {name!r}")
    return '"' + name + '"'


TYPED_CHUNKS = (
    "chunk_id",
    "section_id",
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
    "content",
    "embedding",
    "fts",
    "extra",
)
TYPED_SECTIONS = (
    "section_id",
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
    "content",
    "metadata",
    "extra",
)
SECTION_COLUMNS = TYPED_SECTIONS  # COPY order of the sections table
CHUNK_COLUMNS = tuple(c for c in TYPED_CHUNKS if c != "fts")  # COPY order of the chunks table: fts is generated
HNSW = "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128)"  # langchain's HNSWIndex(16, 128)
REQUIRED_INDEXES = {  # (name, needle in pg_get_indexdef, must be unique), per kind
    "children": (
        ("hnsw on embedding", "USING hnsw (embedding", False),
        ("gin on fts", "USING gin (fts)", False),
        ("primary chunk_id", "(chunk_id)", True),
        ("btree on section_id", "USING btree (section_id)", False),
    ),
    "parents": (
        ("primary section_id", "(section_id)", True),
        ("btree on document_id", "USING btree (document_id)", False),
    ),
}
_LANGUAGE = re.compile(r"^[a-z_]+$")


def v2_tables(base):
    return {"children": base + "_chunks", "parents": base + "_sections"}


def create_tables(sections, chunks, language, dims):
    if not _LANGUAGE.fullmatch(language) or not isinstance(dims, int):
        raise ValueError(f"not a text search configuration or dimension: {language!r}, {dims!r}")
    return (
        f"CREATE TABLE {ident(sections)} (section_id text PRIMARY KEY, document_id text NOT NULL, collection text NOT NULL,"
        " source text NOT NULL, audience text NOT NULL CHECK (audience IN ('internal', 'public')), doc_type text,"
        " year integer, date date, url text, page integer, title text, header text, part_index integer NOT NULL,"
        " content text NOT NULL, metadata jsonb NOT NULL, extra jsonb NOT NULL DEFAULT '{}'::jsonb)",
        f"CREATE TABLE {ident(chunks)} (chunk_id text PRIMARY KEY, section_id text NOT NULL, document_id text NOT NULL,"
        " collection text NOT NULL, source text NOT NULL, audience text NOT NULL, doc_type text, year integer, date date,"
        f" url text, page integer, title text, header text, content text NOT NULL, embedding vector({dims}) NOT NULL,"
        f" fts tsvector NOT NULL GENERATED ALWAYS AS (to_tsvector('{language}'::regconfig, content)) STORED,"
        " extra jsonb NOT NULL DEFAULT '{}'::jsonb)",
    )


def create_indexes(sections, chunks):
    return (
        f"CREATE INDEX {ident(chunks + '_hnsw')} ON {ident(chunks)} {HNSW}",
        f"CREATE INDEX {ident(chunks + '_fts')} ON {ident(chunks)} USING gin (fts)",
        f"CREATE INDEX {ident(chunks + '_section')} ON {ident(chunks)} USING btree (section_id)",
        f"CREATE INDEX {ident(sections + '_document')} ON {ident(sections)} USING btree (document_id)",
    )


def typed_views(collection, chunks, sections):
    return (
        f"CREATE OR REPLACE VIEW {ident(collection + '_chunks')} AS SELECT {', '.join(TYPED_CHUNKS)} FROM {ident(chunks)}",
        f"CREATE OR REPLACE VIEW {ident(collection + '_sections')} AS SELECT {', '.join(TYPED_SECTIONS)} "
        f"FROM {ident(sections)}",
    )


# the embedding cache and the document registry, created with the generation registry
EMBEDDING_CACHE = (
    "CREATE TABLE IF NOT EXISTS embedding_cache (collection text NOT NULL, model text NOT NULL, revision text NOT NULL,"
    " passage_prefix text NOT NULL, text_sha256 text NOT NULL, embedding vector NOT NULL, last_generation integer NOT NULL,"
    " PRIMARY KEY (collection, model, revision, passage_prefix, text_sha256))"
)
INDEX_DOCUMENTS = (
    "CREATE TABLE IF NOT EXISTS index_documents (collection text NOT NULL, document_id text NOT NULL,"
    " source text NOT NULL, url text, doc_key text NOT NULL, first_generation integer NOT NULL,"
    " last_generation integer NOT NULL, content_sha256 text NOT NULL, PRIMARY KEY (collection, document_id))"
)


# The assistant's role: it reads each collection's two views and the generation registry and nothing else (a fresh cluster
# gives it no default privilege). Per-deployment role names would replace this one name.
READER = "rag_app"
REGISTRY_GRANT = f"GRANT SELECT ON {ident('index_generations')} TO {ident(READER)}"


def view_grant(collection):
    """SELECT on the collection's two views; privileges survive CREATE OR REPLACE VIEW, so this is a no-op after the first."""
    return f"GRANT SELECT ON {ident(collection + '_chunks')}, {ident(collection + '_sections')} TO {ident(READER)}"
