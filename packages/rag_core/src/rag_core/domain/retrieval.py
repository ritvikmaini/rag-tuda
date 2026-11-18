"""Pure steps of the retrieval sequence: deduplication, section identity, parent sections."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from rag_core.domain.types import Doc


def deduplicate(documents: Sequence[Any]) -> list[Any]:
    """The first document of each distinct text, in order: a dict's "text", else the document's page_content."""
    seen: set[str] = set()
    unique: list[Any] = []
    for doc in documents:
        text: Any = (
            cast("dict[str, Any]", doc).get("text") if isinstance(doc, dict) else getattr(doc, "page_content", None)
        )
        if text and text not in seen:
            unique.append(doc)
            seen.add(text)
    return unique


def section_identity(text: str, metadata: Mapping[str, Any]) -> str:
    """A section's identity by evals.runner.section_id's rule: its url, its part_index (else its page) and the sha-256 of
    its text, comparable with probe runs. turn_context keeps it beside the section id."""
    url = metadata.get("download_url") or metadata.get("url") or metadata.get("url_location") or ""
    position = metadata.get("part_index")
    if position is None:
        position = metadata.get("page_number", "")
    return "%s|%s|%s" % (url, position, hashlib.sha256(str(text or "").encode("utf-8")).hexdigest())


@dataclass
class Hit:
    """A child or a section as the retrieval sequence passes it on. Mutable: the service writes the reranker's and the
    ranking's scores into its metadata; the attribute names are the `Doc` protocol's. The graph turns hits into LangChain
    Documents."""

    page_content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Section:
    section_id: str
    text: str
    metadata: dict[str, Any]


def parents_of(children: Sequence[Doc], sections: Mapping[str, Section]) -> list[Hit]:
    """The parent section of each child, in the children's order (after a reranker failure every score is 0.0, so this order is
    all there is): the section's metadata with the best child's (highest cross_encoder_score) scores, retriever, id and text
    merged in; a child's filename or page number wins when it has one; rerank_failed is carried."""
    children_by_section: dict[str, list[Doc]] = {}
    for child in children:
        section_id = (child.metadata or {}).get("section_id")
        if section_id:
            children_by_section.setdefault(section_id, []).append(child)
    parents: list[Hit] = []
    for section_id, kids in children_by_section.items():
        section = sections.get(section_id)
        if section is None:
            continue
        best = max(kids, key=lambda c: c.metadata.get("cross_encoder_score") or 0)
        updates = {
            "cross_encoder_score": best.metadata.get("cross_encoder_score"),
            "rrf_fusion_score": best.metadata.get("rrf_fusion_score"),
            "retriever_type": best.metadata.get("retriever"),
            "section_id": section_id,
            "matched_child_text": best.page_content,
        }
        if best.metadata.get("rerank_failed"):
            updates["rerank_failed"] = True
        for key in ("filename", "page_number"):
            if best.metadata.get(key) is not None:
                updates[key] = best.metadata[key]
        parents.append(Hit(page_content=section.text, metadata={**section.metadata, **updates}))
    return parents
