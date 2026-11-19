"""The assistant's graph: one flat graph, the conditional START (a variant goes straight to generate), nodes closing over their
dependencies. The only module that adds nodes. No configuration at import, no checkpointer: the turn store builds each run's
input."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from assistant_api.graph.nodes.analyse import make_analyse
from assistant_api.graph.nodes.follow_up import make_follow_up
from assistant_api.graph.nodes.format_docs import make_format_docs
from assistant_api.graph.nodes.generate import make_generate
from assistant_api.graph.nodes.retrieve import make_retrieve
from assistant_api.graph.routes import entry_route, make_docs_route, make_intent_route
from assistant_api.graph.state import AgentState, GraphDeps

NODES = ("rewrite_follow_up_question", "get_docs", "follow_up_search", "format_docs", "generate")


def build_graph(deps: GraphDeps) -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("rewrite_follow_up_question", make_analyse(deps))
    workflow.add_node("get_docs", make_retrieve(deps))
    workflow.add_node("follow_up_search", make_follow_up(deps))
    workflow.add_node("format_docs", make_format_docs(deps))
    workflow.add_node("generate", make_generate(deps))
    workflow.add_conditional_edges(
        START, entry_route, {"variant": "generate", "question": "rewrite_follow_up_question"}
    )
    workflow.add_conditional_edges(
        "rewrite_follow_up_question",
        make_intent_route(deps.profile.retrieval.rerank.scope_fallback_min),
        {"retrieve": "get_docs", "direct": "generate"},
    )
    # the iterative round: the card gates it, the profile's switch allows it and the profile ships the follow-up client
    iterative = (
        deps.card.gated.iterative_retrieval
        and deps.profile.retrieval.switches.iterative_round
        and deps.follow_up is not None
    )
    workflow.add_conditional_edges(
        "get_docs",
        make_docs_route(iterative, deps.card.timeouts.analysis_s),
        {
            "relevant_docs_found": "format_docs",
            "no_relevant_docs_found": "generate",
            "follow_up": "follow_up_search",
        },
    )
    workflow.add_edge("follow_up_search", "format_docs")
    workflow.add_edge("format_docs", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
