"""The retrieval sequence over ports. Per query the semantic and full-text children (each query bounded), fused into one list
by the profile's weight; the best pre_rerank_k over all queries (switches.fusion_cut off: every child, deduplicated);
reranked against the first rerank_queries queries (off: all), children with their parent sections' texts (whole, as the model
server's length setting decides; switches.whole_parents off: cut to 512 tokens, estimated with the card's chars_per_token); the parents of the
reranked children; twins collapsed to one language; sorted, ranked (rank()) and cut to final_k. Children and sections are the domain's Hit and Section; the service reads and
annotates hits by attribute."""

import asyncio
import logging
import time
from collections.abc import Mapping
from datetime import date
from typing import Any

from rag_core.domain.access import ANONYMOUS, AllowAll, Principal, allowed
from rag_core.domain.doc_scores import mark_unscored, sort_score
from rag_core.domain.filters import Filter
from rag_core.domain.fusion import cut_across_queries, fuse
from rag_core.domain.language import detect
from rag_core.domain.parent_rerank import parent_texts, rerank_texts, score_children
from rag_core.domain.ranking import RankingPolicy, rank
from rag_core.domain.resilience import gather_bounded
from rag_core.domain.retrieval import Hit, Section, deduplicate, parents_of
from rag_core.ports.access import AccessFilter
from rag_core.ports.retrieval import FullTextIndex, Reranker, RerankError, SectionStore, VectorStore
from rag_core.settings.profile import Ranking, Retrieval
from rag_core.settings.sections import Timeouts

logger = logging.getLogger(__name__)


def _metadata(doc: Any) -> dict[str, Any]:
    """The document's metadata dict, {} when it has none."""
    meta: dict[str, Any] = doc.metadata if doc.metadata else {}
    return meta


# the reranker length the parent texts are cut to, in tokens, with switches.whole_parents off (the former server limit)
CUT_PARENT_TOKENS = 512

# the documents whose German text binds: a twin of one of them is never shown in place of the German section
BINDING_GERMAN = frozenset({"po", "mhb", "spp", "apb", "eso", "promotion"})


def collapse_twins(hits: list[Hit], language: str) -> list[Hit]:
    """The hits without the twins of the wrong language, not one hit per key. Sections that share a twin_key are the same text
    in two languages (a page section and its counterpart; every section of a PDF and of its counterpart). A key is binding
    when any hit of it carries a regulation in BINDING_GERMAN (the twins' labels are derived apart and may differ): the German
    sections stay for a binding key, else those in the question's language. A key whose hits are all in the other language
    stays whole, so a regulation found only in English is shown in English; so does a hit without a twin_key."""
    languages: dict[str, set[Any]] = {}
    binding: set[str] = set()
    for h in hits:
        if key := h.metadata.get("twin_key"):
            languages.setdefault(key, set()).add(h.metadata.get("lang"))
            if h.metadata.get("regulation") in BINDING_GERMAN:
                binding.add(key)

    def kept(h: Hit) -> bool:
        key = h.metadata.get("twin_key")
        if not key:
            return True
        wanted = "de" if key in binding else language
        return wanted not in languages[key] or h.metadata.get("lang") == wanted

    return [h for h in hits if kept(h)]


def without_denied(
    children: list[Any], found: Mapping[str, Section], principal: Principal, access: AccessFilter
) -> tuple[list[Any], dict[str, Section]]:
    """The sections this principal may read, and the children without those of a denied section: a denied section's text is
    never scored and its children never answer through the no-parents fallback. Allow-all: every section and every child,
    unchanged."""
    permitted = allowed(found, principal, access)
    denied = found.keys() - permitted.keys()
    return [c for c in children if _metadata(c).get("section_id") not in denied], permitted


class RetrievalService:
    def __init__(
        self,
        embedder: Any,
        vector: VectorStore,
        full_text: FullTextIndex,
        sections: SectionStore,
        reranker: Reranker,
        retrieval: Retrieval,
        ranking: Ranking,
        timeouts: Timeouts,
        final_k: int,
        pre_rerank_k: int,
        rerank_queries: int,
        access: AccessFilter | None = None,
        *,
        chars_per_token: float,
    ) -> None:
        self.embedder, self.vector, self.full_text, self.sections = embedder, vector, full_text, sections
        self.reranker, self.retrieval, self.ranking, self.timeouts = reranker, retrieval, ranking, timeouts
        # the model card's budget
        self.final_k, self.pre_rerank_k, self.rerank_queries = final_k, pre_rerank_k, rerank_queries
        self.access = access or AllowAll()
        self.chars_per_token = chars_per_token  # the card's: sizes the whole_parents-off cut

    async def _search(self, query: str, filter: Filter | None) -> list[Hit]:
        """One query's children: the semantic search (its embedding inside the same time bound) and the keyword search,
        each bounded and both under the filter; one failing does not sink the other. Fused best first; with the fusion cut
        off, the two legs' hits one after the other."""
        r = self.retrieval

        async def semantic() -> list[Hit]:
            vector = await self.embedder.aembed_query(query)
            return await self.vector.search(vector, r.semantic_k, max_distance=r.semantic_max_distance, filter=filter)

        results, failures = await gather_bounded(
            [semantic(), self.full_text.search(query, r.keyword_k, filter)], self.timeouts.retrieval_s
        )
        for index, reason in failures:
            kind = "semantic" if index == 0 else "keyword"
            logger.error(f"[HybridRetriever] {kind} search failed: {reason}")
        if len(failures) == 2:
            raise RuntimeError("every search failed")
        if r.switches.fusion_cut:
            return fuse(results[0] or [], results[1] or [], r.fusion_weight)
        return [hit for hits in results if hits for hit in hits]

    async def retrieve(
        self,
        queries: list[str],
        filter: Filter | None = None,
        *,
        question: str = "",
        date_from: str | None = None,
        date_to: str | None = None,
        principal: Principal = ANONYMOUS,
        today: date | None = None,
    ) -> list[Hit]:
        """The ranked hits for the queries. The filter (None: none) narrows both legs and both section lookups; a filter
        that keeps nothing gives no hits, the search is never repeated without it."""
        logger.info(f"[DocumentRetriever] using {len(queries)} retrieval queries")
        tasks = [self._search(query, filter) for query in queries]
        # each query bounded; failed queries are skipped, and if every query failed the request ends with the German
        # error message instead of "nothing found"
        all_docs_nested, failures = await gather_bounded(tasks, self.timeouts.retrieval_s + 2)
        for index, reason in failures:
            logger.error(f"[DocumentRetriever] retrieval query {index + 1}/{len(tasks)} failed: {reason}")
        if tasks and len(failures) == len(tasks):
            raise RuntimeError("retrieval unavailable: every query failed")
        fusion_cut = self.retrieval.switches.fusion_cut
        if fusion_cut:  # after both legs of every query, before the parents are read and the reranker scores
            unique_docs = cut_across_queries([docs for docs in all_docs_nested if docs], self.pre_rerank_k)
        else:
            unique_docs = deduplicate([doc for docs in all_docs_nested if docs for doc in docs])
        logger.info(f"[DocumentRetriever] Found {len(unique_docs)} unique docs to rerank.")

        # the reranker also scores each child's parent text; a child keeps the higher score.
        # When the parent lookup fails, the children are scored alone.
        parent_text_by_section: dict[str, str] = {}
        if self.retrieval.rerank.parent_text:
            section_ids = list(
                {section_id for d in unique_docs if (section_id := _metadata(d).get("section_id")) is not None}
            )
            try:
                found = await asyncio.wait_for(
                    self.sections.get_many(section_ids, filter), timeout=self.timeouts.parent_lookup_s
                )
                # a denied section's children leave before the reranker, so its text is never scored
                unique_docs, found = without_denied(unique_docs, found, principal, self.access)
                parent_text_by_section = parent_texts(
                    {sid: {"text": s.text, "metadata": dict(s.metadata)} for sid, s in found.items()}
                )
                if not self.retrieval.switches.whole_parents:
                    cut = int(CUT_PARENT_TOKENS * self.chars_per_token)
                    parent_text_by_section = {s: text[:cut] for s, text in parent_text_by_section.items()}
            except Exception as e:
                logger.error(f"[DocumentRetriever] parent texts for reranking unavailable: {type(e).__name__} {e}")
            logger.info(
                f"[DocumentRetriever] reranking {len(unique_docs)} children with {len(parent_text_by_section)} parent texts"
            )

        reranked_docs = await self._rerank(
            queries[: self.rerank_queries] if fusion_cut else queries, unique_docs, parent_text_by_section
        )
        logger.info(f"[DocumentRetriever] {len(reranked_docs)} docs after reranking.")

        try:  # bounded
            logger.info(f"[DB] Looking up parents for {len(reranked_docs)} child chunks")
            section_ids = list(dict.fromkeys(i for i in (_metadata(d).get("section_id") for d in reranked_docs) if i))
            if reranked_docs and not section_ids:
                logger.warning("[DB] No section_ids found in child chunks")
            found = await asyncio.wait_for(
                self.sections.get_many(section_ids, filter), timeout=self.timeouts.parent_lookup_s
            )
            if section_ids:
                logger.info(f"[DB] Retrieved {len(found)} parent documents from database")
            reranked_docs, permitted = without_denied(reranked_docs, found, principal, self.access)
            parent_docs = parents_of(reranked_docs, permitted)
            if section_ids:
                logger.info(f"[DB] LEN PARENTS: {len(parent_docs)}")
        except asyncio.TimeoutError:
            logger.error("[DocumentRetriever] parent lookup timed out")
            parent_docs = []
        if not parent_docs and reranked_docs:
            # The parent lookup failed or found nothing: the matched chunks are the context
            logger.warning("[DocumentRetriever] no parents for the matched chunks; answering from the chunks")
            parent_docs = reranked_docs

        final_docs = collapse_twins(deduplicate(parent_docs), detect(question or " ".join(queries)))
        # Always sort by score before the cut: otherwise the list is in database order and the cut could drop the best.
        final_docs.sort(key=lambda d: sort_score(d.metadata), reverse=True)

        r = self.ranking
        # today pins the ranking's clock (the offline harness); production passes none: the day of the request
        policy = RankingPolicy(
            relevance_floor=r.relevance_floor,
            source_min_score=r.source_min_score,
            max_per_document=r.max_per_document,
            recency_types=tuple(r.recency_types),
            recency_bonus_per_year=r.recency_bonus_per_year,
            recency_max_bonus=r.recency_max_bonus,
            date_from=date_from,
            date_to=date_to,
            today=today,
            in_range_bonus=r.in_range_bonus,
            in_range_by_year=r.in_range_by_year,
            question_years=r.question_years,
            no_recency_in_range=r.no_recency_in_range,
        )
        metas = [dict(d.metadata or {}) for d in final_docs]
        kept, adjusted, report = rank(metas, question, policy)
        logger.info(f"[Ranking] {report}")
        for i, doc in enumerate(final_docs):
            doc.metadata["rank_score"] = round(adjusted.get(i, 0.0), 4)
        final_docs = [final_docs[i] for i in kept]
        result_docs = final_docs[: self.final_k]
        for i, doc in enumerate(result_docs):
            meta = _metadata(doc)
            score = meta.get("cross_encoder_score", 0) or 0
            url = meta.get("url", "unknown")
            filename = (meta.get("filename") or "unknown")[:60]
            rank_score = meta.get("rank_score", score) or 0
            logger.info(
                f"[DocumentRetriever] Final doc #{i + 1}: score={score:.3f} rank={rank_score:.3f}, "
                f"file={filename}, url={url}"
            )
        return result_docs

    async def _rerank(
        self, queries: list[str], documents: list[Any], parent_text_by_section: dict[str, str]
    ) -> list[Any]:
        """Max-score fusion over the queries; the threshold, then at least min_docs documents, none of them below the
        ranking's relevance_floor."""
        if not documents or not queries:
            return []
        doc_map = {doc.page_content: doc for doc in documents}
        page_contents = (
            rerank_texts(documents, parent_text_by_section)
            if parent_text_by_section
            else [doc.page_content for doc in documents]
        )
        started = time.monotonic()
        try:
            reranked_results = await self.reranker.rerank(queries, page_contents, self.timeouts.reranker_s)
        except RerankError as e:
            logger.error(f"Error calling reranker multi-query service: {e}. Falling back to original documents.")
            return mark_unscored(documents)
        logger.info(
            f"[DocumentRetriever] rerank call {time.monotonic() - started:.2f}s for {len(page_contents)} texts "
            f"x {len(queries)} queries"
        )
        all_scored_docs: list[Any] = []
        if parent_text_by_section:  # a child keeps the higher of its own and its parent's score
            all_scored_docs = score_children(
                documents, {content: score for score, content in reranked_results}, parent_text_by_section
            )
        for score, content in [] if parent_text_by_section else reranked_results:
            original_doc = doc_map.get(content)
            if original_doc:
                original_doc.metadata["cross_encoder_score"] = score
                all_scored_docs.append(original_doc)
        threshold = self.retrieval.rerank.threshold
        reranked_docs = [doc for doc in all_scored_docs if doc.metadata["cross_encoder_score"] >= threshold]
        min_docs = self.retrieval.rerank.min_docs
        floor = self.ranking.relevance_floor
        if len(reranked_docs) < min_docs and len(all_scored_docs) > len(reranked_docs):
            topped_up = [d for d in all_scored_docs if floor <= d.metadata["cross_encoder_score"] < threshold]
            reranked_docs = [*reranked_docs, *topped_up][:min_docs]
            logger.info(
                f"[DocumentRetriever] Min-doc guarantee: kept {len(reranked_docs)} docs "
                f"(threshold would have left {sum(1 for d in all_scored_docs if d.metadata['cross_encoder_score'] >= threshold)})"
            )
        return reranked_docs
