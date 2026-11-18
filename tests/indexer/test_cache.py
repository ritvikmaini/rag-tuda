"""cache.fill without a database: only the distinct texts the cache lacks go
to the embedding server, in first-seen order and batches of 128; the counts name the texts an entry serves and the texts
embedded now; every new entry is written with the key, the text's hash, the vector as text and the generation."""

from indexer.index import cache

KEY = ("example", "m", "r", "")


class FakeConn:
    """Answers the cache lookup with the given hashes and records the COPY rows."""

    def __init__(self, have=()):
        self.have, self.rows = [(h,) for h in have], []

    def execute(self, statement, params=None):
        return self

    def fetchall(self):
        return self.have

    def cursor(self):
        return self

    def copy(self, statement):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write_row(self, row):
        self.rows.append(row)


def vector_of(text):
    """The vector the embedding server gives a text: it depends on the text alone."""
    return [int(cache.sha(text)[:6], 16) / 1e6, 0.25]


def test_fill_embeds_each_distinct_miss_once_in_batches_of_128():
    calls = []

    async def embed(texts):
        calls.append(list(texts))
        return [vector_of(t) for t in texts]

    texts = [f"t{i}" for i in range(310)] * 2  # every text twice
    conn = FakeConn(have=[cache.sha(f"t{i}") for i in range(10)])  # t0 .. t9 are cached
    assert cache.fill(conn, KEY, 9, embed, texts) == {"cached": 20, "computed": 300}
    assert [len(c) for c in calls] == [128, 128, 44]
    misses = [f"t{i}" for i in range(10, 310)]
    assert sum(calls, []) == misses
    # every entry holds its own text's vector (the key, the text's hash, the vector as text, the generation), across all batches
    assert conn.rows == [(*KEY, cache.sha(t), str(vector_of(t)), 9) for t in misses]
