"""The graph's routers. Their run names (IntentRouter, CheckDocsFoundRouter) appear in traces and the event stream."""

from typing import Any

from rag_core.domain.answer_shapes import VARIANTS
from rag_core.domain.query_analysis import RETRIEVAL_INTENTS


def entry_route(state: Any) -> str:
    """The graph's entry: a variant goes straight to generate, everything else to the query analysis."""
    return "variant" if state.get("variant") in VARIANTS else "question"


async def intent_route(state: Any) -> str:
    """Document intents go to retrieval; all others straight to the reply."""
    intent = state.get("intent") or "lookup"
    return "retrieve" if intent in RETRIEVAL_INTENTS else "direct"


def make_intent_route(scope_fallback_min: float | None):
    """The intent router with the out-of-scope fallback: set, an out-of-scope question goes to retrieval too (the retrieve node decides whether
    its best passage is strong enough to answer); null is `intent_route` exactly."""
    if scope_fallback_min is None:
        return intent_route

    async def scoped_intent_route(state: Any) -> str:
        return "retrieve" if state.get("intent") == "out_of_scope" else await intent_route(state)

    scoped_intent_route.__name__ = "IntentRouter"
    return scoped_intent_route


async def docs_found_route(state: Any) -> str:
    return "no_relevant_docs_found" if not state.get("raw_docs") else "relevant_docs_found"


def make_docs_route(iterative: bool):
    """The router after the search. With the iterative round on, a searching intent whose analysis needs more documents and
    that found some goes to the follow-up search first; everything else (and every turn with the round off) is
    `docs_found_route` exactly."""
    if not iterative:
        return docs_found_route

    async def iterative_docs_route(state: Any) -> str:
        if state.get("needs_more") and state.get("raw_docs") and state.get("intent") in RETRIEVAL_INTENTS:
            return "follow_up"
        return await docs_found_route(state)

    iterative_docs_route.__name__ = "CheckDocsFoundRouter"
    return iterative_docs_route


intent_route.__name__ = "IntentRouter"
docs_found_route.__name__ = "CheckDocsFoundRouter"
