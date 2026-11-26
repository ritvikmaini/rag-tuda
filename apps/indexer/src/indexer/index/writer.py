"""Writes one build into the generation's tables: rows for sections and chunks (schema v2). Ids are content hashes."""

import json
import datetime
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import logging
from collections import Counter
from typing import cast


from indexer.chunking.units import Parent
from indexer.index import cache, ddl


def sanitize_text(text: str) -> str:
    """
    Removes null bytes from a given string.
    """
    return text.replace("\x00", "")


def sanitize_metadata(metadata: dict | list | str) -> dict | list | str:
    """
    Recursively sanitizes metadata by removing null characters.
    """
    if isinstance(metadata, dict):
        return {k: sanitize_metadata(v) for k, v in metadata.items()}
    elif isinstance(metadata, list):
        return [sanitize_metadata(item) for item in metadata]
    elif isinstance(metadata, str):
        return metadata.replace("\u0000", "")
    return metadata


BATCH_SIZE = 128  # chunks per embedding batch
_DIGITS = re.compile(r"[0-9]+")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _id(*parts) -> str:
    """The first 32 hex characters of the SHA-256 of the parts joined by NUL."""
    return hashlib.sha256("\0".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:32]


def document_id(source: str, doc_key: str) -> str:
    return _id(source, doc_key)


def section_id(document: str, part_index: int, content: str) -> str:
    return _id(document, part_index, cache.sha(content))


def chunk_id(section: str, chunk_index: int, text: str) -> str:
    return _id(section, chunk_index, cache.sha(text))


@dataclass(frozen=True)
class ChunkRow:
    """One chunks row by column (ddl.CHUNK_COLUMNS) except the embedding, which write_v2 adds after its batch's call."""

    columns: dict[str, Any]

    @property
    def content(self) -> str:
        return self.columns["content"]

    def values(self, vector_text: str) -> tuple:
        return tuple(vector_text if c == "embedding" else self.columns[c] for c in ddl.CHUNK_COLUMNS)


def _digits(value):
    """The value as an int when its JSON text is digits only, else None."""
    text = str(value) if isinstance(value, int) and not isinstance(value, bool) else value
    return int(text) if isinstance(text, str) and _DIGITS.fullmatch(text) else None


def _date(value):
    """An ISO day (YYYY-MM-DD) as a date, else None."""
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _array(value, cast):
    """A list of the value's items as `cast` makes them, or None for an absent or empty value: NULL, never an empty array.
    A bare string is one item, not its characters."""
    if isinstance(value, str):
        value = [value] if value else None
    return [cast(v) for v in value] if value else None


def _flag(value):
    """A flag as a bool: a bool, or the text "true" or "false" in any case; bool("false") would be true."""
    if isinstance(value, str):
        if value.strip().lower() not in ("true", "false"):
            raise ValueError(f"archived: {value!r} is neither true nor false")
        return value.strip().lower() == "true"
    return bool(value)


def _json_default(value):
    """json.dumps for the metadata: a date is its ISO day."""
    if isinstance(value, datetime.date):
        return value.isoformat()
    raise TypeError(f"metadata value {value!r} is not JSON serializable")


def _typed(meta, prefix, profile, source):
    """The typed columns every section and its chunks carry. The source is the connector that produced the document.
    The document metadata (ddl.DOCUMENT_COLUMNS) comes from the metadata keys of the same names: degree, programme and
    po_version as lists, valid_from as an ISO day; a document that states no language has the profile's."""
    audiences = {s.connector: s.audience for s in profile.ingest.sources.values()}
    if source not in audiences:
        raise ValueError(
            f"a section of {meta.get('url')!r} names connector {source!r}, which the profile does not list"
        )
    page = _digits(meta.get("page_number"))
    return {
        "document_id": document_id(source, meta.get("doc_key") or meta.get("url") or ""),
        "collection": profile.collections[0].name,
        "source": source,
        "audience": audiences[source],
        "doc_type": meta.get("doc_type"),
        "year": _digits(meta.get("doc_year")),
        "date": _date(meta.get("doc_date")),
        "url": meta.get("url"),
        "page": page if page else None,
        "title": meta.get("title"),
        "header": sanitize_text(prefix) or None,
        "regulation": meta.get("regulation") or None,
        "degree": _array(meta.get("degree"), str),
        "programme": _array(meta.get("programme"), str),
        "po_version": _array(meta.get("po_version"), int),
        "archived": _flag(meta.get("archived")),
        "valid_from": _date(meta.get("valid_from")),
        "lang": meta.get("lang") or profile.identity.language,
        "twin_key": meta.get("twin_key") or None,
    }


def rows(parents: list[Parent], profile) -> tuple[list[tuple], list[ChunkRow]]:
    """The sections (tuples in ddl.SECTION_COLUMNS order) and chunks of one build: parents
    grouped by filename in first-seen order; the parent metadata after the JSON round trip, its section_id the section
    id; children numbered before the deduplication per (section, sanitized content)."""
    groups = defaultdict(list)
    for p in parents:
        groups[p.metadata.get("filename", "unknown")].append(p)
    sections, chunks, seen, unique_ids = [], [], set(), []
    for name, group in groups.items():
        folder = group[0].metadata.get("folder_iid", "")
        for p in group:
            combined = {
                "scraper": {"runDuration": "", "errors": ""},
                **p.metadata,
                "section_id": None,  # set below, once the section id is known; the key keeps its place
                "source": name,
                "file-folder": folder,
            }
            meta = cast(
                dict, sanitize_metadata(json.loads(json.dumps(combined, default=_json_default)))
            )  # a dict comes back as a dict
            typed = _typed(meta, p.prefix, profile, p.source)
            content = sanitize_text(p.text)
            sid = section_id(typed["document_id"], meta["part_index"], content)
            meta["section_id"] = sid  # the assistant reads the section id from the metadata too (citations)
            sections.append(
                tuple(
                    {
                        **typed,
                        "section_id": sid,
                        "part_index": meta["part_index"],
                        "content": content,
                        "metadata": json.dumps(meta),
                        "extra": "{}",
                    }[c]
                    for c in ddl.SECTION_COLUMNS
                )
            )
            for i, child in enumerate(p.children):
                text = sanitize_text(child.text)
                if (sid, text) in seen:
                    continue
                seen.add((sid, text))
                unique_ids.append(f"{sid}_{i}")
                chunks.append(
                    ChunkRow(
                        {**typed, "chunk_id": chunk_id(sid, i, text), "section_id": sid, "content": text, "extra": "{}"}
                    )
                )
    duplicate_ids = [item for item, count in Counter(unique_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate unique_id values found: {duplicate_ids}")
    return sections, chunks


def write_v2(conn, embed, tables, parents, profile, *, generation, cache_key) -> dict:
    """Writes one v2 generation: sections, then the embedding cache's misses (cache.fill), then
    chunks in batches of 128 with their cached vectors (cache.take), then the indexes. conn: a direct psycopg connection
    (autocommit); embed: the async EmbeddingClient.embed_passages; tables: the row's names; generation: the row's number;
    cache_key: cache.key(...). Vectors go in as text, never through pgvector's binary adapter.
    Returns {"children", "parents"} for verify and {"embeddings": {"cached", "computed"}} for the reply."""
    sections, chunks = rows(parents, profile)
    for statement in ddl.create_tables(tables["parents"], tables["children"], profile.retrieval.embedding.dimensions):
        conn.execute(statement)
    with conn.cursor().copy(
        f"COPY {ddl.ident(tables['parents'])} ({', '.join(ddl.SECTION_COLUMNS)}) FROM STDIN"
    ) as copy:
        for row in sections:
            copy.write_row(row)
    embeddings = cache.fill(conn, cache_key, generation, embed, [c.content for c in chunks])
    batches = -(-len(chunks) // BATCH_SIZE)
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors = cache.take(conn, cache_key, generation, [c.content for c in batch])
        with conn.cursor().copy(
            f"COPY {ddl.ident(tables['children'])} ({', '.join(ddl.CHUNK_COLUMNS)}) FROM STDIN"
        ) as copy:
            for chunk in batch:
                copy.write_row(chunk.values(vectors[cache.sha(chunk.content)]))
        logging.getLogger(__name__).info(f"[Writer] batch {start // BATCH_SIZE + 1}/{batches}")
    for statement in ddl.create_indexes(tables["parents"], tables["children"]):
        conn.execute(statement)
    return {"children": len(chunks), "parents": len(sections), "embeddings": embeddings}
