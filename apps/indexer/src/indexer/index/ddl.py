"""Schema v2: the generation tables, their indexes, the two typed
views and the required indexes. Only strings; generations.py runs them."""

import re

_NAME = re.compile(r"[a-z][a-z0-9_]{0,62}")


def ident(name):
    """A table or view name as a quoted SQL identifier; only lower-case names of letters, digits and _."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"not a valid relation name: {name!r}")
    return '"' + name + '"'


# the document metadata the crawler derives, stored on every section and every chunk of the document
DOCUMENT_COLUMNS = ("regulation", "degree", "programme", "po_version", "archived", "valid_from", "lang", "twin_key")
GENERATED_COLUMNS = ("fts_de", "fts_en")  # computed from content and lang by the chunks table
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
    *DOCUMENT_COLUMNS,
    "content",
    "embedding",
    *GENERATED_COLUMNS,
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
    *DOCUMENT_COLUMNS,
    "part_index",
    "content",
    "metadata",
    "extra",
)
SECTION_COLUMNS = TYPED_SECTIONS  # COPY order of the sections table
CHUNK_COLUMNS = tuple(c for c in TYPED_CHUNKS if c not in GENERATED_COLUMNS)  # COPY order of the chunks table
# no vector index: the corpus is small enough for an exact scan, which always returns k rows under a filter
REQUIRED_INDEXES = {  # (name, needle in pg_get_indexdef, must be unique), per kind
    "children": (
        ("gin on fts_de", "USING gin (fts_de)", False),
        ("gin on fts_en", "USING gin (fts_en)", False),
        ("primary chunk_id", "(chunk_id)", True),
        ("btree on section_id", "USING btree (section_id)", False),
    ),
    "parents": (
        ("primary section_id", "(section_id)", True),
        ("btree on document_id", "USING btree (document_id)", False),
    ),
}


def v2_tables(base):
    return {"children": base + "_chunks", "parents": base + "_sections"}


# the document metadata columns (DOCUMENT_COLUMNS); degree, programme and po_version are NULL where a document applies to all
_DOCUMENT_DDL = (
    "regulation text, degree text[], programme text[], po_version integer[], archived boolean NOT NULL DEFAULT false,"
    " valid_from date, lang text NOT NULL, twin_key text"
)


def create_tables(sections, chunks, dims):
    """The two tables of a generation. fts_de covers every chunk; fts_en only the English ones (NULL for the rest)."""
    if not isinstance(dims, int):
        raise ValueError(f"not a vector dimension: {dims!r}")
    return (
        f"CREATE TABLE {ident(sections)} (section_id text PRIMARY KEY, document_id text NOT NULL, collection text NOT NULL,"
        " source text NOT NULL, audience text NOT NULL CHECK (audience = 'public'), doc_type text,"
        f" year integer, date date, url text, page integer, title text, header text, {_DOCUMENT_DDL},"
        " part_index integer NOT NULL, content text NOT NULL, metadata jsonb NOT NULL,"
        " extra jsonb NOT NULL DEFAULT '{}'::jsonb)",
        f"CREATE TABLE {ident(chunks)} (chunk_id text PRIMARY KEY, section_id text NOT NULL, document_id text NOT NULL,"
        " collection text NOT NULL, source text NOT NULL, audience text NOT NULL, doc_type text, year integer, date date,"
        f" url text, page integer, title text, header text, {_DOCUMENT_DDL}, content text NOT NULL,"
        f" embedding vector({dims}) NOT NULL,"
        " fts_de tsvector NOT NULL GENERATED ALWAYS AS (to_tsvector('german'::regconfig, content)) STORED,"
        " fts_en tsvector GENERATED ALWAYS AS (CASE WHEN lang = 'en' THEN to_tsvector('english'::regconfig, content) END)"
        " STORED, extra jsonb NOT NULL DEFAULT '{}'::jsonb)",
    )


def create_indexes(sections, chunks):
    return (
        f"CREATE INDEX {ident(chunks + '_fts_de')} ON {ident(chunks)} USING gin (fts_de)",
        f"CREATE INDEX {ident(chunks + '_fts_en')} ON {ident(chunks)} USING gin (fts_en)",
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
    " passage_prefix text NOT NULL, dimensions integer NOT NULL, text_sha256 text NOT NULL, embedding vector NOT NULL,"
    " last_generation integer NOT NULL,"
    " PRIMARY KEY (collection, model, revision, passage_prefix, dimensions, text_sha256))"
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
