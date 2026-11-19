"""Fusion of the two legs of one query and the cut over all queries. Pure; standard library only."""

from collections.abc import Sequence

from rag_core.domain.retrieval import Hit

DISTANCE = "child_chunk_semantic_search_score"  # the semantic leg's score: a distance, the nearest is the best
RANK = "fts_rank"  # the keyword leg's score: the higher the better
FUSION_SCORE = "fusion_score"


def _normalised(values: Sequence[float]) -> list[float]:
    """The values scaled to [0, 1] by min-max; a leg whose values do not differ (one hit, or all equal) scores 1.0 each."""
    low, high = min(values, default=0.0), max(values, default=0.0)
    if high == low:
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def _by_text(hits: Sequence[Hit], key: str, sign: float) -> dict[str, tuple[Hit, float]]:
    """Each distinct text's first hit with its normalised score under `key`; sign -1 for a distance. A hit without the
    score counts as the leg's worst."""
    first: dict[str, Hit] = {}
    for hit in hits:
        first.setdefault(hit.page_content, hit)
    given = [None if (v := hit.metadata.get(key)) is None else sign * float(v) for hit in first.values()]
    worst = min((v for v in given if v is not None), default=0.0)
    normalised = _normalised([worst if v is None else v for v in given])
    return {text: (hit, score) for (text, hit), score in zip(first.items(), normalised, strict=True)}


def fuse(sem: list[Hit], lex: list[Hit], w: float) -> list[Hit]:
    """One query's hits from both legs, best first: each leg min-max normalised, then `w * semantic + (1 - w) * lexical`
    (a hit one leg did not find scores 0 there). Each hit is a copy whose metadata merges both legs' and carries
    `fusion_score`; equal scores keep the semantic leg's order, then the keyword leg's."""
    semantic, lexical = _by_text(sem, DISTANCE, -1.0), _by_text(lex, RANK, 1.0)
    fused: list[Hit] = []
    for text in dict.fromkeys([*semantic, *lexical]):
        s_hit, s_score = semantic.get(text, (None, 0.0))
        l_hit, l_score = lexical.get(text, (None, 0.0))
        metadata = {**(l_hit.metadata if l_hit else {}), **(s_hit.metadata if s_hit else {})}
        fused.append(Hit(text, {**metadata, FUSION_SCORE: w * s_score + (1 - w) * l_score}))
    return sorted(fused, key=lambda h: h.metadata[FUSION_SCORE], reverse=True)


def cut_across_queries(per_query: list[list[Hit]], k: int) -> list[Hit]:
    """The k best distinct texts over all queries' fused hits, by the best `fusion_score` each reached, best first (equal
    scores keep the queries' order). The cut is total: k hits for the question, not k per query."""
    best: dict[str, Hit] = {}
    for hits in per_query:
        for hit in hits:
            text = hit.page_content
            if text and (text not in best or hit.metadata[FUSION_SCORE] > best[text].metadata[FUSION_SCORE]):
                best[text] = hit
    return sorted(best.values(), key=lambda h: h.metadata[FUSION_SCORE], reverse=True)[: max(k, 0)]
