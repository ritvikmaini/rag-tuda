"""The source formatting node: sorts the retrieved documents and formats them as the prompt's sources; returns a partial
update. The origin labels come from the profile's messages."""

import json
from collections import Counter
from typing import Any

from rag_core.domain.context_budget import format_source, join_sources

from assistant_api.graph.state import AgentState, GraphDeps


def make_format_docs(deps: GraphDeps):
    labels = deps.messages.labels.source_labels()

    async def format_docs(state: AgentState) -> dict[str, Any]:
        docs: Any = state["raw_docs"]
        if isinstance(docs, str):
            docs = json.loads(docs)

        # Sort docs by frequency and filename
        if docs:
            doc_frequency = Counter(doc.metadata.get("filename", "") for doc in docs)
            docs = sorted(
                docs,
                key=lambda doc: (
                    -doc_frequency[doc.metadata.get("filename", "")],
                    doc.metadata.get("filename", ""),
                    doc.metadata.get("page_number", -1),
                ),
            )

        # a partial update: only the two keys this node makes
        return {
            "grouped_raw_docs": docs,
            "formatted_docs": join_sources([format_source(doc, labels=labels) for doc in docs]),
        }

    return format_docs
