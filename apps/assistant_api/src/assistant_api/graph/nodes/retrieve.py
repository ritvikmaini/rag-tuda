"""The document search node: retrieves the documents for the analysed question; returns a partial update."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from rag_core.domain.access import ANONYMOUS
from rag_core.domain.doc_scores import sort_score

from assistant_api.graph.state import AgentState, GraphDeps

logger = logging.getLogger(__name__)


def make_retrieve(deps: GraphDeps):
    # the router sends an out-of-scope turn here only when the value is set
    threshold = deps.profile.retrieval.rerank.scope_fallback_min or 0.0

    async def get_docs(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        question = state["messages"][-1].content
        # The query analysis already folded the user's text into the queries, so the raw follow-up is not searched again.
        query_list = state.get("retrieval_queries") or [question]
        principal = (config.get("configurable") or {}).get("principal", ANONYMOUS)
        hits = await deps.retrieval.retrieve(
            query_list,
            state.get("filter"),
            question=state.get("standalone_question") or question,
            date_from=state.get("date_from"),
            date_to=state.get("date_to"),
            principal=principal,
        )
        # LangChain Documents inside the graph
        docs = [Document(page_content=h.page_content, metadata=h.metadata) for h in hits]
        if state.get("intent") != "out_of_scope":
            return {"raw_docs": docs}
        # an out-of-scope question was searched only to see whether the corpus holds a strong answer; a weak best passage
        # discards the hits and the question is declined.
        top1 = max((sort_score(h.metadata) for h in hits), default=0.0)  # a parent may carry None: it counts as 0.0
        answered = bool(hits) and top1 >= threshold
        logger.info(f"[ScopeFallback] top1={top1:.3f} threshold={threshold:.3f} answered={'yes' if answered else 'no'}")
        return {"raw_docs": docs if answered else [], "scope_fallback": answered}

    return get_docs
