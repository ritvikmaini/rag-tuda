"""What the domain reads from the assistant's documents and messages: attributes, never LangChain's classes. LangChain's
Document and messages fit these protocols; the domain's own Hit and Section are in domain.retrieval."""

from typing import Any, Protocol


class Doc(Protocol):
    page_content: str
    metadata: dict[str, Any]


class Message(Protocol):
    @property
    def type(
        self,
    ) -> str: ...  # "human", "ai", "system" (LangChain's message types); read-only so LangChain's messages match

    content: Any
    id: str | None
    additional_kwargs: dict[str, Any]
