"""Langfuse tracing of a graph run: the SDK v4 LangChain handler and the run metadata."""

from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler


def make_handler(authenticated: bool) -> Any | None:
    """The callback handler when the client authenticated; None runs untraced."""
    return CallbackHandler() if authenticated else None


def trace_id(handler: object | None, turn_id: str, variant: str | None) -> str | None:
    """The id the run's Langfuse trace carries (meta.trace_id): None when the run is untraced (Runtime.handler is None, also when
    the client failed to authenticate); for a question the turn id's seed, so a trace can be found from the turn; a variant reruns
    the same turn, so it gets its own random id (two runs never share a trace)."""
    if handler is None:
        return None
    return Langfuse.create_trace_id() if variant else Langfuse.create_trace_id(seed=turn_id)


def run_config(handler: Any | None, thread_id: str, collection: str, trace_id: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if handler:
        if trace_id:  # one handler per request: the run's trace takes meta's trace id
            handler = CallbackHandler(trace_context={"trace_id": trace_id})
        config.update(
            {
                "callbacks": [handler],
                "metadata": {
                    "langfuse_session_id": thread_id,
                    "langfuse_user_id": thread_id,
                    "langfuse_tags": [f"{collection}_assistant"],
                },
            }
        )
    return config
