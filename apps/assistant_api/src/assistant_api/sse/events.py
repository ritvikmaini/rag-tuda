"""Frames the pipeline sends: shape, citations and docs. They are models (`assistant_api.sse.frames`); the guard encodes
them."""

import logging
from typing import Any, cast

from rag_core.domain.answer_shapes import shape_of
from rag_core.domain.citations import entries

from assistant_api.sse.frames import Citation, Citations, Docs, Shape, ShapeFrame

logger = logging.getLogger(__name__)


def shape_frame(shape: Any) -> ShapeFrame:
    return ShapeFrame(content=cast(Shape, shape_of(shape)))


def citations_frame(sources: Any, docs: Any) -> Citations | None:
    # The citation links must never break an answer: any unexpected input gives no frame.
    try:
        found = entries(sources, docs)
        if not found:
            return None
        return Citations(content=[Citation(**entry) for entry in found])
    except Exception as exc:
        logger.warning("[Citations] no citations frame: %s: %s", type(exc).__name__, exc)
        return None


def serialize_documents(documents: Any, none_text: str) -> list[dict[str, Any]]:
    """The docs payload: every document's text and metadata, or the profile's note that no sources were used."""
    if not documents:
        return [{"page_content": none_text, "metadata": {}}]
    return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]


def docs_frame(documents: Any, none_text: str) -> Docs:
    return Docs(content=serialize_documents(documents, none_text))
