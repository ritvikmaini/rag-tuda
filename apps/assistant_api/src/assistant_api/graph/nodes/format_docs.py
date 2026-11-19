"""The source formatting node: sorts the retrieved documents and formats them as the prompt's sources; returns a partial
update. The origin labels and the words that frame the sources come from the profile's messages, in the answer's language."""

import json
from collections import Counter
from typing import Any

from rag_core.domain.answer_shapes import answer_language
from rag_core.domain.context_budget import format_source, join_sources

from assistant_api.graph.state import AgentState, GraphDeps, messages_for


def make_format_docs(deps: GraphDeps):
    async def format_docs(state: AgentState) -> dict[str, Any]:
        labels = messages_for(deps, answer_language(state)).labels.source_labels()
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
            "formatted_docs": join_sources([format_source(doc, labels=labels) for doc in docs], labels),
        }

    return format_docs
