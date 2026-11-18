"""The embedding cache: one table, keyed per collection by the
embedding model, its revision, the passage prefix and the SHA-256 of the exact text the server embeds after the prefix. A
build looks every chunk text up first and sends only the misses to the embedding server, in batches of 128; every entry a
build uses gets last_generation = its generation; housekeeping deletes what neither the live nor the previous generation
used. The first build on an empty cache copies the live generation's vectors when its provenance names the same model and
revision. Every write runs inside the build lock."""

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from indexer.index.ddl import ident

BATCH_SIZE = 128  # texts per request to the embedding server
COLUMNS = "collection, model, revision, passage_prefix, text_sha256, embedding, last_generation"
_KEY = "collection = %s AND model = %s AND revision = %s AND passage_prefix = %s"
log = logging.getLogger(__name__)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def key(collection: str, embedding: Any, served_revision: str | None) -> tuple[str, str, str, str]:
    """(collection, model, revision, passage prefix). The revision is the one the server reports in /info (the build's
    provenance); the profile's only when the server has no /info (an older server image, the harness's fake)."""
    return (collection, embedding.model, served_revision or embedding.revision, embedding.passage_prefix)


def seed(conn, cache_key: tuple[str, str, str, str]) -> int:
    """Fills an empty cache of this collection from the live generation's chunks when that generation's provenance names
    the key's model and revision: per distinct text the vector of its chunk with the lowest chunk_id. Returns the entries
    written (0: the cache has entries, or no such generation)."""
    collection, model, revision, _ = cache_key
    if conn.execute("SELECT EXISTS (SELECT 1 FROM embedding_cache WHERE collection = %s)", (collection,)).fetchone()[0]:
        return 0
    row = conn.execute(
        "SELECT generation, tables ->> 'children' FROM index_generations WHERE collection = %s AND status = 'live' "
        "AND embedding_model = %s AND embedding_revision = %s",
        (collection, model, revision),
    ).fetchone()
    if row is None:
        log.info("[Embeddings] the cache is empty and no live generation of this model and revision can seed it")
        return 0
    generation, chunks = row
    seeded = conn.execute(
        f"INSERT INTO embedding_cache ({COLUMNS}) SELECT DISTINCT ON (h) %s::text, %s::text, %s::text, %s::text, h, embedding, "
        f"%s::integer FROM "
        f"(SELECT encode(sha256(convert_to(content, 'UTF8')), 'hex') AS h, chunk_id, embedding FROM {ident(chunks)}) t "
        "ORDER BY h, chunk_id ON CONFLICT DO NOTHING",
        (*cache_key, generation),
    ).rowcount
    log.info(f"[Embeddings] seeded {seeded} entries from generation {generation}")
    return seeded


def fill(
    conn,
    cache_key,
    generation: int,
    embed: Callable[[list[str]], Coroutine[Any, Any, list[list[float]]]],
    texts: Sequence[str],
) -> dict[str, int]:
    """Embeds the texts the cache lacks (each distinct text once, first-seen order, batches of 128) and stores them.
    Returns {"cached": texts an existing entry serves, "computed": distinct texts embedded now}."""
    started = time.monotonic()
    have = {r[0] for r in conn.execute(f"SELECT text_sha256 FROM embedding_cache WHERE {_KEY}", cache_key).fetchall()}
    hashes = [sha(t) for t in texts]
    misses = list({h: t for h, t in zip(hashes, texts, strict=True) if h not in have}.items())
    batches = -(-len(misses) // BATCH_SIZE)
    for start in range(0, len(misses), BATCH_SIZE):
        batch = misses[start : start + BATCH_SIZE]
        vectors = asyncio.run(embed([t for _, t in batch]))  # one event loop per batch
        with conn.cursor().copy(f"COPY embedding_cache ({COLUMNS}) FROM STDIN") as copy:
            for (h, _), vector in zip(batch, vectors, strict=True):
                copy.write_row((*cache_key, h, str([float(d) for d in vector]), generation))
        log.info(f"[Embeddings] batch {start // BATCH_SIZE + 1}/{batches}")
    counts = {"cached": sum(1 for h in hashes if h in have), "computed": len(misses)}
    seconds = time.monotonic() - started
    log.info(f"[Embeddings] {counts['cached']} cached, {counts['computed']} computed in {seconds:.0f} s")
    return counts


def take(conn, cache_key, generation: int, texts: Sequence[str]) -> dict[str, str]:
    """{text sha: the vector as text} for texts that fill() stored; marks every entry as used by this generation."""
    wanted = list({sha(t) for t in texts})
    got = dict(
        conn.execute(
            f"UPDATE embedding_cache SET last_generation = %s WHERE {_KEY} AND text_sha256 = ANY(%s) "
            "RETURNING text_sha256, embedding::text",
            (generation, *cache_key, wanted),
        ).fetchall()
    )
    if len(got) != len(wanted):
        raise RuntimeError(f"[Embeddings] {len(wanted) - len(got)} texts have no cached vector")
    return got


def prune(conn, collection: str, below: int) -> int:
    """Deletes this collection's entries that no generation from `below` on has used (housekeeping)."""
    return conn.execute(
        "DELETE FROM embedding_cache WHERE collection = %s AND last_generation < %s", (collection, below)
    ).rowcount
