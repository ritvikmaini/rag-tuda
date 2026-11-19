"""The iterative round: one extra search after the first, for a question that needs sources of more than one document. The
analysis model, shown the kept sections' headers and matched children, names up to two further queries; one search over the
first queries and the new ones together (fused, reranked and cut to final_k like any search, the reference hop included)
replaces the first result. A failing call or search keeps the first result."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from rag_core.domain.access import ANONYMOUS
from rag_core.domain.follow_up import DIGEST_TOKENS, sections_digest
from rag_core.services.follow_up import follow_up_queries

from assistant_api.graph.state import AgentState, GraphDeps

logger = logging.getLogger(__name__)


def make_follow_up(deps: GraphDeps):
    digest_chars = int(DIGEST_TOKENS * deps.card.chars_per_token)
    model = deps.follow_up

    async def follow_up_search(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        if model is None:  # the router sends no turn here without the client
            return {"follow_up_queries": []}
        question = state.get("standalone_question") or state["messages"][-1].content
        searched = state.get("retrieval_queries") or [question]
        queries = await follow_up_queries(
            question=question,
            searched=searched,
            sections=sections_digest(state["raw_docs"], digest_chars),
            identity=deps.profile.identity,
            prompts=deps.prompts,
            model=model,
            timeout_s=deps.card.timeouts.analysis_s,
        )
        if not queries:
            return {"follow_up_queries": []}
        principal = (config.get("configurable") or {}).get("principal", ANONYMOUS)
        try:
            # the first queries stay first: the reranker scores against the leading ones, so round one's scores stand
            hits = await deps.retrieval.retrieve(
                [*searched, *queries],
                state.get("filter"),
                question=question,
                date_from=state.get("date_from"),
                date_to=state.get("date_to"),
                principal=principal,
            )
        except Exception as e:
            logger.error(f"[FollowUp] extra search failed, keeping the first search: {type(e).__name__}")
            return {"follow_up_queries": []}
        if not hits:
            return {"follow_up_queries": []}
        return {
            "raw_docs": [Document(page_content=h.page_content, metadata=h.metadata) for h in hits],
            "follow_up_queries": queries,
        }

    return follow_up_search
