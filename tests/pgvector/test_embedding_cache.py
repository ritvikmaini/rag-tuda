"""The embedding cache and the stable ids on a real cluster: a second write
embeds nothing and stores the same vectors, an edited text is the only one embedded, seeding copies the live generation's
vectors only for the same model and revision, housekeeping keeps what the live and the previous generation use (per
collection), a stable chunk id reads as a uuid (contract_reverse.sql's cast), the document registry follows every publish
and the change report reads across the switch to stable ids."""

import os
import random
from pathlib import Path

import pytest

from indexer.chunking.units import Child, Parent
from indexer.index import cache, writer
from indexer.index import generations as gen
from indexer.settings import load_settings

pytestmark = pytest.mark.skipif(os.environ.get("RUN_DOCKER_TESTS") != "1", reason="needs Docker (RUN_DOCKER_TESTS=1)")
ROOT = Path(__file__).resolve().parents[2]


def vector_of(text):
    """The vector the embedding server gives a text: it depends on the text alone (the float4-hard values stay in front)."""
    rng = random.Random(text)
    return [1 / 3, 0.1, 2.0000001] + [rng.uniform(-1, 1) for _ in range(765)]


@pytest.fixture
def profile(tmp_path):
    return load_settings({"RAG_PROFILE": str(ROOT / "profiles" / "example" / "profile.yaml"),
                          "RAG_OVERLAY": str(ROOT / "deploy" / "env" / "ci-example.yaml")},
                         secrets_dir=str(tmp_path / "secrets")).profile  # fmt: skip


class Embedder:
    """The embedding server: vector_of(text) per text; records every text it was sent."""

    def __init__(self):
        self.texts = []

    async def __call__(self, texts):
        self.texts += texts
        return [vector_of(t) for t in texts]


def items(extra=""):
    """Two sections, four chunks, three distinct chunk texts (one shared across the sections), non-ASCII included."""
    web = {"part_index": 0, "filename": "Rathaus", "origin": "website", "source_type": "page",
           "url": "https://example.org/r", "doc_key": "https://example.org/r", "page_number": -1}  # fmt: skip
    pdf = {"part_index": 0, "filename": "DA", "origin": "archive", "source_type": "pdf",
           "doc_key": "https://docs.example.org/da.pdf", "page_number": 2}  # fmt: skip
    return [
        Parent(
            "Öffnungszeiten im Rathaus",
            web,
            (Child("Rathaus: Montag"), Child("Rathaus: Dienstag" + extra)),
            "Rathaus: ",
            "example.fixture",
        ),
        Parent(
            "§ 1 Geltung",
            pdf,
            (Child("DA · § 1: Geltung"), Child("Rathaus: Montag")),
            "DA · § 1: ",
            "example.fixture",
        ),
    ]


def chunks_hold_their_texts_vectors(conn, target, n):
    """Every chunk of generation n holds the vector of its own text, joined on content: just embedded, served by the cache or
    seeded. The cache's text round trip must store what vector_in gave on the embedder's float list (a wrong text-to-vector
    mapping, a vector of another text, fails here)."""
    chunks = target.tables(n)["children"]
    texts = [r[0] for r in conn.execute(f'SELECT DISTINCT content FROM "{chunks}"').fetchall()]
    assert texts
    for text in texts:
        total, own = conn.execute(
            f'SELECT count(*), count(*) FILTER (WHERE embedding = %s::vector) FROM "{chunks}" WHERE content = %s',
            (str(vector_of(text)), text),
        ).fetchone()
        assert total and total == own, f"{own} of {total} chunks of {text!r} hold its own vector"


def write(conn, target, profile, embed, parents, prov=None, publish=True):
    """One build as write_and_publish runs it: begin, seed, write, verify, publish; every chunk is checked against its own
    text's vector. Returns (n, written, seeded)."""
    prov = prov or {"config_hash": "x"}
    n, _ = gen.begin(conn, target, prov)
    key = cache.key(target.collection, profile.retrieval.embedding, prov.get("embedding_revision"))
    seeded = cache.seed(conn, key)
    written = writer.write_v2(conn, embed, target.tables(n), parents, profile, generation=n, cache_key=key)
    gen.verify(conn, target, n, written)
    chunks_hold_their_texts_vectors(conn, target, n)
    if publish:
        gen.publish(conn, target, n)
    return n, written, seeded


def ids(conn, target, n):
    tables = target.tables(n)
    return (sorted(r[0] for r in conn.execute(f'SELECT section_id FROM "{tables["parents"]}"').fetchall()),
            sorted(r[0] for r in conn.execute(f'SELECT chunk_id FROM "{tables["children"]}"').fetchall()))  # fmt: skip


def test_a_second_write_embeds_nothing_and_stores_the_same_vectors(connect, profile):
    conn, target, embed = connect(), gen.Target.for_collection("e1"), Embedder()
    n1, w1, _ = write(conn, target, profile, embed, items())
    assert w1["embeddings"] == {"cached": 0, "computed": 3} and len(embed.texts) == 3
    n2, w2, _ = write(conn, target, profile, embed, items())
    assert w2["embeddings"] == {"cached": 4, "computed": 0} and len(embed.texts) == 3
    assert ids(conn, target, n1) == ids(conn, target, n2)
    for n in (n1, n2):  # the fresh and the cache-served write both hold each text's own vector (write() checked it too)
        chunks_hold_their_texts_vectors(conn, target, n)
    assert conn.execute("SELECT count(*), min(last_generation), max(last_generation) FROM embedding_cache "
                        "WHERE collection = 'e1'").fetchone() == (3, n2, n2)  # fmt: skip


def test_an_edited_text_is_the_only_one_embedded(connect, profile):
    conn, target, embed = connect(), gen.Target.for_collection("e2"), Embedder()
    write(conn, target, profile, embed, items())
    embed.texts.clear()
    _, written, _ = write(conn, target, profile, embed, items(extra=" und Mittwoch"))
    assert embed.texts == ["Rathaus: Dienstag und Mittwoch"] and written["embeddings"] == {"cached": 3, "computed": 1}


def test_seeding_copies_the_live_vectors_only_for_the_same_model_and_revision(connect, profile):
    conn, target, embed = connect(), gen.Target.for_collection("e3"), Embedder()
    e = profile.retrieval.embedding
    prov = {"config_hash": "x", "embedding_model": e.model, "embedding_revision": e.revision}
    write(conn, target, profile, embed, items(), prov=prov)
    conn.execute("DELETE FROM embedding_cache WHERE collection = 'e3'")  # a live generation with no cache
    embed.texts.clear()
    _, written, seeded = write(conn, target, profile, embed, items(), prov=prov)
    assert (
        seeded == 3 and embed.texts == [] and written["embeddings"] == {"cached": 4, "computed": 0}
    )  # SQL sha == Python sha
    assert cache.seed(conn, cache.key("e3", e, None)) == 0  # never twice: the collection's cache is not empty
    conn.execute("DELETE FROM embedding_cache WHERE collection = 'e3'")
    other = {**prov, "embedding_revision": "f" * 40}
    _, written, seeded = write(conn, target, profile, embed, items(), prov=other)
    assert seeded == 0 and written["embeddings"]["computed"] == 3


def test_housekeeping_keeps_what_live_and_previous_use_per_collection(connect, profile):
    conn, embed = connect(), Embedder()
    target, other = gen.Target.for_collection("e4"), gen.Target.for_collection("e4b")
    write(conn, other, profile, embed, items())  # generation 1 of another collection
    write(conn, target, profile, embed, items())
    write(conn, target, profile, embed, items(extra=" A"))
    write(conn, target, profile, embed, items(extra=" B"))
    result = gen.housekeep(conn, target)  # keeps 3 (live) and 2 (previous): "Rathaus: Dienstag" was used by 1 only
    assert result["cache_pruned"] == 1
    kept = {r[0] for r in conn.execute("SELECT text_sha256 FROM embedding_cache WHERE collection = 'e4'").fetchall()}
    assert kept == {
        cache.sha(t) for t in ("Rathaus: Montag", "DA · § 1: Geltung", "Rathaus: Dienstag A", "Rathaus: Dienstag B")
    }
    assert conn.execute("SELECT count(*) FROM embedding_cache WHERE collection = 'e4b'").fetchone()[0] == 3
    assert gen.rollback(conn, target)["live"] == 2
    assert gen.housekeep(conn, target)["cache_pruned"] == 0  # live 2, previous 3: both still served


def test_a_stable_chunk_id_reads_as_a_uuid(connect, profile):  # contract_reverse.sql's chunk_id::uuid
    conn, target = connect(), gen.Target.for_collection("e5")
    n, _, _ = write(conn, target, profile, Embedder(), items())
    chunks = target.tables(n)["children"]
    rows = conn.execute(f'SELECT chunk_id, chunk_id::uuid::text FROM "{chunks}"').fetchall()
    assert len(rows) == 4 and all(u.replace("-", "") == c for c, u in rows)


def registry(conn, collection):
    return {r[0]: r[1:] for r in conn.execute(
        "SELECT doc_key, first_generation, last_generation, content_sha256 FROM index_documents WHERE collection = %s",
        (collection,)).fetchall()}  # fmt: skip


def test_publish_records_the_documents_and_the_report_counts_them(connect, profile):
    conn, target, embed = connect(), gen.Target.for_collection("e6"), Embedder()
    n1, _, _ = write(conn, target, profile, embed, items())
    first = registry(conn, "e6")
    assert {k: v[:2] for k, v in first.items()} == {"https://example.org/r": (n1, n1),
                                                     "https://docs.example.org/da.pdf": (n1, n1)}  # fmt: skip
    n2, _, _ = write(conn, target, profile, embed, items(extra=" X"), publish=False)
    assert gen.document_changes(conn, target, n2) == {
        "added": 0,
        "changed": 0,
        "removed": 0,
    }  # a child text: no section text
    edited = items()
    edited[1] = Parent("§ 1 Geltung, neu", edited[1].metadata, edited[1].children, edited[1].prefix, edited[1].source)
    moved = [edited[0], Parent("Neu", {**edited[0].metadata, "doc_key": "https://example.org/neu", "url": "https://example.org/neu",
                                       "filename": "Neu"}, (Child("Neu: Text"),), "Neu: ", edited[0].source)]  # fmt: skip
    n3, _, _ = write(conn, target, profile, embed, edited, publish=False)
    assert gen.document_changes(conn, target, n3) == {"added": 0, "changed": 1, "removed": 0}
    n4, _, _ = write(conn, target, profile, embed, moved, publish=False)
    assert gen.document_changes(conn, target, n4) == {"added": 1, "changed": 0, "removed": 1}
    gen.publish(conn, target, n3)
    after = registry(conn, "e6")
    assert after["https://docs.example.org/da.pdf"][:2] == (n1, n3)
    assert after["https://docs.example.org/da.pdf"][2] != first["https://docs.example.org/da.pdf"][2]
    gen.rollback(conn, target)  # back to n1: first and last generation never move backwards
    assert registry(conn, "e6")["https://docs.example.org/da.pdf"][:2] == (n1, n3)


def test_the_change_report_reads_across_the_id_switch(connect, profile):
    conn, target, embed = connect(), gen.Target.for_collection("e7"), Embedder()
    n1, _, _ = write(conn, target, profile, embed, items())
    sections = target.tables(n1)[
        "parents"
    ]  # the live generation as the earlier writer wrote it: random ids, doc_key ids
    conn.execute(f"UPDATE \"{sections}\" SET document_id = metadata ->> 'doc_key', section_id = md5(random()::text)")
    n2, _, _ = write(conn, target, profile, embed, items(), publish=False)
    assert gen.document_changes(conn, target, n2) == {"added": 0, "changed": 0, "removed": 0}
