"""The query analysis node: rag_core.services.analysis.analyse over the turn's history and question."""

from datetime import datetime, timezone
from typing import Any

from rag_core.domain.answer_shapes import FORMATS_VERSION
from rag_core.domain.history import context_turns, history_text
from rag_core.services.analysis import analyse

from assistant_api.graph.state import AgentState, GraphDeps


def make_analyse(deps: GraphDeps):
    history = deps.profile.history

    async def rewrite_follow_up_question(state: AgentState) -> dict[str, Any]:
        prior = context_turns(state["messages"], history.turns, history.max_age_hours, datetime.now(timezone.utc))
        question = state["messages"][-1].content
        analysis, queries = await analyse(
            question=question,
            history=history_text(prior),
            now=datetime.now(),
            identity=deps.profile.identity,
            prompts=deps.prompts,
            model=deps.analysis,
            timeout_s=deps.timeouts.analysis_s,
            max_queries=deps.profile.retrieval.max_queries,
            first_turn=not prior,
        )
        # Every run starts from the turn store's input, so these per-turn fields are empty already; the reset guarantees that
        # a direct reply never sees another turn's sources.
        return {
            "language": analysis.language,
            "intent": analysis.intent,
            "standalone_question": analysis.standalone_question,
            "retrieval_queries": queries,
            "date_from": analysis.date_from,
            "date_to": analysis.date_to,
            "raw_docs": [],
            "grouped_raw_docs": [],
            "formatted_docs": "",
            "context_sources": [],
            # the shape only for a page that renders it; a new question has no versions yet
            "answer_shape": analysis.answer_shape if state.get("formats") == FORMATS_VERSION else "text",
            "answer_versions": {},
            "scope_fallback": False,
        }

    return rewrite_follow_up_question
