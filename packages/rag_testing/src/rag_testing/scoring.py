"""The fake embedder's vectors, the fake reranker's scores and its /v2/rerank scores (aligned to the documents).

vector(): a normalised 768-dimensional vector from the SHA-256 of the text, the `query: ` prefix stripped first, so a
query and an equal passage map alike. rerank(): the share of a query's words found in a text, the maximum over the
queries, one entry per distinct text, sorted like the real service (score, then text, descending), so the assistant's 0.50
threshold and its three-document minimum act as in production. Deterministic; standard library only."""

import hashlib
import math
import re
from collections.abc import Sequence

DIMENSION = 768
QUERY_PREFIX = "query: "
WORD = re.compile(r"\w+")


def vector(text: str) -> list[float]:
    seed = hashlib.sha256(text.removeprefix(QUERY_PREFIX).encode("utf-8")).digest()
    raw = b"".join(hashlib.sha256(seed + i.to_bytes(2, "big")).digest() for i in range(DIMENSION // 32))
    values = [byte - 127.5 for byte in raw]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def score(query: str, text: str) -> float:
    asked = words(query)
    return len(asked & words(text)) / len(asked) if asked else 0.0


def scores(queries: Sequence[str], documents: Sequence[str]) -> list[float]:
    """/v2/rerank: one score per document, in input order (duplicates too), the maximum over the queries."""
    return [max((score(q, d) for q in queries), default=0.0) for d in documents]
