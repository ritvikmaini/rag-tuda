"""What the graph's nodes see: the state and the dependencies the nodes close over."""

from dataclasses import dataclass
from typing import Annotated, Any, Optional, Protocol

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from rag_core.domain.access import ANONYMOUS, Principal
from rag_core.ports.chat import ChatModel, TokenCounter
from rag_core.services.prompts import PromptService
from rag_core.settings.card import Card
from rag_core.settings.profile import Profile
from rag_core.settings.sections import Timeouts

from assistant_api.settings import Messages


class AgentState(TypedDict):
    """
    Represents the state of an agent in a conversational AI system.
    """

    raw_docs: list[dict]
    grouped_raw_docs: list[dict]
    formatted_docs: list[str]
    generation: str
    messages: Annotated[list, add_messages]
    language: str
    standalone_question: str
    intent: str
    retrieval_queries: list[str]
    date_from: Optional[str]
    date_to: Optional[str]
    context_sources: list[dict]
    formats: str  # "v1" when the page renders response formats v1, else ""
    answer_shape: str  # text | table | checklist | steps | mindmap: the shape of the answer on screen
    variant: Optional[str]  # set only while a variant of the last answer runs
    answer_versions: dict  # versions of the last answer: name -> {"text", "shape"}
    scope_fallback: bool  # an out-of-scope question whose best passage was strong enough, answered from the documents


class Retriever(Protocol):
    """What the retrieve node needs of rag_core's RetrievalService (a test passes a fake)."""

    async def retrieve(
        self,
        queries: list[str],
        *,
        question: str = "",
        date_from: str | None = None,
        date_to: str | None = None,
        principal: Principal = ANONYMOUS,
    ) -> list[Any]: ...


@dataclass(frozen=True)
class GraphDeps:
    profile: Profile
    card: Card
    messages: Messages
    prompts: PromptService
    analysis: ChatModel
    answer_llm: Any  # ChatOpenAI: the generate node awaits it, so LangChain's callbacks stream the answer
    counter: TokenCounter
    retrieval: Retriever  # a Protocol, not the concrete RetrievalService
    timeouts: Timeouts
