# pyright: standard
"""
Reranker input "child or parent": the cross-encoder scores every candidate child and, once per section, the child's
parent text ('<title>: ' + the parent without page marks); a child keeps the higher of the two scores.
Measured offline on the structure-chunked index: L Hit@3 0.921 → 0.937, with the threshold
calibrated to 0.50 (keeps the share of candidates 0.15 keeps with children alone).
Pure helpers; services.retrieval wires them when the profile enables parent-text scoring.
"""

from typing import Any, Dict, List, Optional

from rag_core.domain.source_meta import strip_source_text


def parent_texts(parents: Optional[Dict[str, dict[str, Any]]]) -> Dict[str, str]:
    """{section_id: '<title>: ' + parent text without page marks} from a parent lookup result
    ({section_id: {'text': ..., 'metadata': ...}}); parents without text are left out."""
    out: Dict[str, str] = {}
    for section_id, parent in (parents or {}).items():
        text = strip_source_text((parent or {}).get("text") or "")
        if not text:
            continue
        title = str(((parent or {}).get("metadata") or {}).get("title") or "")
        out[section_id] = f"{title}: {text}" if title else text
    return out


def rerank_texts(documents: List[Any], parent_text_by_section: Dict[str, str]) -> List[str]:
    """The texts sent to the reranker: every child's text in order, then each distinct parent text not already among them."""
    texts = [d.page_content for d in documents]
    seen = set(texts)
    for d in documents:
        text = parent_text_by_section.get((d.metadata or {}).get("section_id"))  # pyright: ignore[reportArgumentType]
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def score_children(documents: List[Any], scores: Dict[str, float], parent_text_by_section: Dict[str, str]) -> List[Any]:
    """The children the reranker scored, each with cross_encoder_score = max(own score, its parent's score) and child_score =
    its own score, sorted by (cross_encoder_score, child_score) so that among children tied through their parent the best
    own passage leads (it becomes matched_child_text: the citation quote and the neighbour anchor); a child without a score
    is left out."""
    scored = []
    for d in documents:
        own = scores.get(d.page_content)
        if own is None:
            continue
        parent_score = scores.get(
            parent_text_by_section.get((d.metadata or {}).get("section_id")) or ""  # pyright: ignore[reportArgumentType]
        )
        d.metadata["cross_encoder_score"] = float(own if parent_score is None else max(own, parent_score))
        d.metadata["child_score"] = float(own)
        scored.append(d)
    return sorted(scored, key=lambda d: (d.metadata["cross_encoder_score"], d.metadata["child_score"]), reverse=True)
