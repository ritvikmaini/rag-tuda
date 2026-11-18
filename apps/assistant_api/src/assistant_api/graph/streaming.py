"""Streaming of a graph run: `run_stream` turns LangGraph v2 events into frames, and the status frames drive the page's
thinking display.

status_frame(event) turns one LangGraph v2 stream event of a pipeline node run into the status frame
(`Status(content={...})`) and returns None for every other event. The page words
the steps in German; the backend only sends the step and its facts. The facts are the ones the pipeline
already logs (the [QueryAnalysis] line) and traces in Langfuse, so every step a user sees can be traced there.

Steps: analysing (query analysis starts), understood (standalone question and intent), searching (the
document search starts, with its search terms; only document intents reach it), found (documents in the
answer context), writing (the answer starts)."""

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from rag_core.domain.answer_shapes import shape_log_line
from rag_core.domain.language import strip_foreign
from rag_core.domain.source_meta import select_sources_for_display

from assistant_api.chat.guard import PROGRESS
from assistant_api.graph.build import NODES
from assistant_api.graph.nodes.generate import holds_answer
from assistant_api.sse import frames as fr
from assistant_api.sse.events import citations_frame, docs_frame, shape_frame

logger = logging.getLogger(__name__)

REWRITE_NODE, SEARCH_NODE, FORMAT_NODE, GENERATE_NODE = NODES


def _source_url(doc):
    meta = doc.get("metadata") if isinstance(doc, dict) else getattr(doc, "metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    return meta.get("download_url") or meta.get("url") or ""


def _status(event):
    kind = event.get("event")
    name = event.get("name")
    data = event.get("data") or {}
    if name == REWRITE_NODE and kind == "on_chain_start":
        return {"step": "analysing"}
    if name == REWRITE_NODE and kind == "on_chain_end":
        output = data.get("output")
        output = output if isinstance(output, dict) else {}
        return {
            "step": "understood",
            "question": str(output.get("standalone_question") or "").strip(),
            "intent": str(output.get("intent") or ""),
        }
    if name == SEARCH_NODE and kind == "on_chain_start":
        state = data.get("input")
        state = state if isinstance(state, dict) else {}
        if state.get("intent") == "out_of_scope":
            return None  # a check search; a declined question shows no search step, an answered one shows "found"
        question = str(state.get("standalone_question") or "").strip()
        queries = []
        for query in state.get("retrieval_queries") or []:
            query = str(query).strip()
            if query and query != question and query not in queries:
                queries.append(query)
        return {"step": "searching", "queries": queries}
    if name == FORMAT_NODE and kind == "on_chain_end":
        state = data.get("input")
        raw_docs = (state.get("raw_docs") if isinstance(state, dict) else None) or []
        urls = {url for url in (_source_url(doc) for doc in raw_docs) if url}
        return {"step": "found", "documents": len(urls), "passages": len(raw_docs)}
    if name == GENERATE_NODE and kind == "on_chain_start":
        return {"step": "writing"}
    return None


def status_frame(event):
    # The status display must never break an answer: any unexpected event shape gives no frame.
    try:
        content = _status(event)
    except Exception as exc:
        logger.warning(
            "[StreamStatus] no status frame for %s/%s: %s: %s",
            event.get("event") if isinstance(event, dict) else "?",
            event.get("name") if isinstance(event, dict) else "?",
            type(exc).__name__,
            exc,
        )
        return None
    return fr.Status(content=content) if content else None


@dataclass
class StreamResult:
    """What the chat service needs after the stream: the turn's sources and the shape asked for."""

    retrieved_docs: list[Any] = field(default_factory=list)
    requested_shape: str = "text"
    analysis: dict[str, Any] = field(default_factory=dict)  # the query analysis (turn row and context)
    candidates: list[Any] = field(default_factory=list)  # the sources, grouped (turn context)
    sources: list[Any] = field(default_factory=list)  # the generate node's context_sources (turn context)
    answer: str | None = None  # the model's answer text; None for a fallback text
    first_chunk: float | None = None  # time.monotonic() at the first chunk sent (for ttft)
    scope_fallback: bool = False  # a strong hit answered an out-of-scope question
    usage: dict[str, int] = field(default_factory=dict)  # the answer call's prompt_tokens and completion_tokens


async def run_stream(
    chain: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
    *,
    formats_on: bool,
    variant: str | None,
    sources_mode: str,
    none_text: str,
    result: StreamResult,
    user: str,
    strip_cjk: bool,
) -> AsyncIterator[Any]:
    """LangGraph's v2 events as frame models; PROGRESS for every other step (the guard's activity signal)."""
    streamed_any = False
    streamed = ""  # the text sent as chunks so far
    held = False
    dropped = 0  # foreign-script characters left out of the answer (the model's bias misses byte-piece tokens)
    async for event in chain.astream_events(inputs, config=config, version="v2"):
        status = status_frame(event)
        if status:
            yield status
        node = event.get("metadata", {}).get("langgraph_node")
        if event["event"] == "on_chain_start" and event.get("name") == GENERATE_NODE:
            try:  # a held answer is sent whole when generate ends
                held = holds_answer(event["data"]["input"])
            except Exception:  # an unexpected input shape streams
                held = False
        if event["event"] == "on_chat_model_start" and node == "generate":
            logging.info("Stream started...")
            yield PROGRESS
        elif event["event"] == "on_chat_model_stream" and node == "generate":
            chunk_content = event["data"][
                "chunk"
            ].text  # the text of the chunk: str content as it is, content blocks joined
            clean = strip_foreign(chunk_content, strip_cjk)
            dropped += len(chunk_content) - len(clean)
            chunk_content = clean
            if held or not chunk_content:
                # any chunk of the model is activity for the guard, a reasoning chunk without text included (nothing is shown
                # or stored of it); a held answer, perhaps replaced, follows at generate's end
                yield PROGRESS
            else:
                streamed_any = True
                streamed += chunk_content
                if result.first_chunk is None:
                    result.first_chunk = time.monotonic()
                yield fr.Chunk(content=chunk_content)
        elif event["event"] == "on_chain_end" and node == "format_docs":
            result.retrieved_docs = event["data"]["input"]["raw_docs"]
            result.candidates = list(
                (event["data"].get("output") or {}).get("grouped_raw_docs") or result.retrieved_docs
            )
            if sources_mode == "before":
                yield docs_frame(result.retrieved_docs, none_text)
        elif event["event"] == "on_chain_end" and event.get("name") == "rewrite_follow_up_question":
            result.analysis = dict(event["data"].get("output") or {})
            result.requested_shape = (event["data"].get("output") or {}).get("answer_shape") or "text"
            if formats_on:
                yield shape_frame(result.requested_shape)
            else:
                yield PROGRESS
        elif event["event"] == "on_chain_end" and event.get("name") == SEARCH_NODE:
            result.scope_fallback = bool((event["data"].get("output") or {}).get("scope_fallback"))
            yield PROGRESS
        elif event["event"] == "on_chain_end" and event.get("name") == "generate":
            output = event["data"].get("output") or {}
            generation = output.get("generation") or ""
            result.sources = list(output.get("context_sources") or [])
            result.usage = dict(output.get("usage") or {})
            produced = output.get("messages") or []
            text = str(produced[-1].content) if produced else ""
            # the answer kept and the answer sent are the same text; the generation is the chunks, the text the stored one
            kept_generation, kept_text = strip_foreign(str(generation), strip_cjk), strip_foreign(text, strip_cjk)
            dropped = max(dropped, len(str(generation)) - len(kept_generation), len(text) - len(kept_text))
            generation, text = kept_generation, kept_text
            if dropped:
                logger.warning(f"[ForeignScript] dropped={dropped} (user: {user})")
            result.answer = text if text.strip() else None  # a fallback text (messages: []) is no model answer
            if formats_on:
                logging.info(f"{shape_log_line(result.requested_shape, generation, variant)} (user: {user})")
            if generation and not streamed_any:
                # Replies that did not come through the model stream (the guard's fallback texts) must still reach the page.
                streamed_any = True
                if result.first_chunk is None:
                    result.first_chunk = time.monotonic()
                yield fr.Chunk(content=generation)
            elif generation.startswith(streamed) and len(generation) > len(streamed):
                # text the node appended after the model's stream (a closing note) still reaches the page
                yield fr.Chunk(content=generation[len(streamed) :])
            elif streamed_any and not produced and not generation.startswith(streamed):
                # the model failed after part of its answer was sent: the fallback text follows it
                yield fr.Chunk(content="\n\n" + generation)
            citations = citations_frame(output.get("context_sources"), result.retrieved_docs)
            if citations:
                yield citations
            if sources_mode in ("seen", "cited") and result.retrieved_docs:
                shown = select_sources_for_display(output.get("context_sources") or [], generation, sources_mode)
                by_section = {(d.metadata or {}).get("section_id"): d for d in result.retrieved_docs}
                docs = [by_section[s["section_id"]] for s in shown if s.get("section_id") in by_section]
                if docs:
                    yield docs_frame(docs, none_text)
        else:
            # every other pipeline step counts as activity for the guard
            yield PROGRESS
