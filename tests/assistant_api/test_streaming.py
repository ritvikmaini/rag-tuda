"""Status frames for the page's thinking display."""

import asyncio
from types import SimpleNamespace

from assistant_api.graph import streaming as ss  # noqa: E402
from assistant_api.sse import frames as fr
from langchain_core.messages import AIMessage, AIMessageChunk


def content(frame):
    obj = fr.wire(frame)
    assert obj["type"] == "status"
    return obj["content"]


class Doc:
    def __init__(self, metadata):
        self.metadata = metadata
        self.page_content = "text"


def test_analysing_when_the_query_analysis_starts():
    event = {"event": "on_chain_start", "name": "rewrite_follow_up_question", "data": {"input": {}}}
    assert content(ss.status_frame(event)) == {"step": "analysing"}


def test_understood_carries_only_the_question_and_intent():
    event = {
        "event": "on_chain_end",
        "name": "rewrite_follow_up_question",
        "data": {
            "output": {
                "standalone_question": "Wie beantrage ich Urlaub?",
                "intent": "lookup",
                "retrieval_queries": ["Wie beantrage ich Urlaub?", "Urlaubsantrag Workflow"],
            }
        },
    }
    assert content(ss.status_frame(event)) == {
        "step": "understood",
        "question": "Wie beantrage ich Urlaub?",
        "intent": "lookup",
    }


def test_understood_names_the_applied_filter_when_the_run_has_one():
    from rag_core.domain.filters import Filter

    applied = Filter(("bachelor",), ("bsc-informatik",), (2015,), None, None, True, "chip")
    event = {
        "event": "on_chain_end",
        "name": "rewrite_follow_up_question",
        "data": {"output": {"standalone_question": "Frage?", "intent": "lookup", "filter": applied}},
    }
    assert content(ss.status_frame(event)) == {
        "step": "understood",
        "question": "Frage?",
        "intent": "lookup",
        "applied_filter": applied.as_json(),
    }
    off = {"standalone_question": "Frage?", "intent": "lookup", "filter": None}  # filtering off: no key
    event["data"] = {"output": off}
    assert "applied_filter" not in content(ss.status_frame(event))


def test_a_direct_reply_on_the_first_turn_reports_no_search():
    # The first turn adds the typed text to retrieval_queries for every intent;
    # a direct reply never searches, so no search terms may reach the page.
    event = {
        "event": "on_chain_end",
        "name": "rewrite_follow_up_question",
        "data": {
            "output": {
                "standalone_question": "Wer bist du?",
                "intent": "about_bot",
                "retrieval_queries": ["Wer bist du?", "wer bist du eigentlich"],
            }
        },
    }
    assert content(ss.status_frame(event)) == {"step": "understood", "question": "Wer bist du?", "intent": "about_bot"}


def test_understood_tolerates_a_missing_or_odd_output():
    for data in ({}, {"output": None}, {"output": "text"}):
        event = {"event": "on_chain_end", "name": "rewrite_follow_up_question", "data": data}
        assert content(ss.status_frame(event)) == {"step": "understood", "question": "", "intent": ""}


def test_searching_when_the_document_search_starts_with_its_terms_without_the_question():
    event = {
        "event": "on_chain_start",
        "name": "get_docs",
        "data": {
            "input": {
                "standalone_question": "Wie beantrage ich Urlaub?",
                "retrieval_queries": [
                    "Wie beantrage ich Urlaub?",
                    "Urlaubsantrag Workflow",
                    " Urlaub Formular ",
                    "Urlaubsantrag Workflow",
                ],
            }
        },
    }
    assert content(ss.status_frame(event)) == {
        "step": "searching",
        "queries": ["Urlaubsantrag Workflow", "Urlaub Formular"],
    }


def test_the_check_search_of_an_out_of_scope_question_reports_no_step():
    event = {
        "event": "on_chain_start",
        "name": "get_docs",
        "data": {"input": {"intent": "out_of_scope", "standalone_question": "Witz?", "retrieval_queries": ["Witz?"]}},
    }
    assert ss.status_frame(event) is None  # a declined question gives no status frame


def test_searching_with_only_the_question_has_no_extra_terms():
    event = {
        "event": "on_chain_start",
        "name": "get_docs",
        "data": {"input": {"standalone_question": "Was gilt?", "retrieval_queries": ["Was gilt?"]}},
    }
    assert content(ss.status_frame(event)) == {"step": "searching", "queries": []}


def test_an_unexpected_event_shape_never_raises():
    # The status display must never break an answer: odd shapes give no frame.
    for event in [
        {"event": "on_chain_start", "name": "get_docs", "data": {"input": {"retrieval_queries": 5}}},
        {"event": "on_chain_end", "name": "format_docs", "data": {"input": {"raw_docs": 7}}},
        {"event": "on_chain_end", "name": "rewrite_follow_up_question", "data": 3},
    ]:
        assert ss.status_frame(event) is None


def test_found_counts_unique_documents_and_all_passages():
    raw = [
        Doc({"download_url": "https://docs.example.org/files/a/?layout=file"}),
        Doc(
            {
                "download_url": "https://docs.example.org/files/a/?layout=file",
                "url": "https://docs.example.org/files/",
            }
        ),
        {"metadata": {"url": "https://example.web/page"}},
        Doc({}),
        Doc(None),
    ]
    event = {"event": "on_chain_end", "name": "format_docs", "data": {"input": {"raw_docs": raw}, "output": {}}}
    assert content(ss.status_frame(event)) == {"step": "found", "documents": 2, "passages": 5}


def test_found_with_no_documents():
    event = {"event": "on_chain_end", "name": "format_docs", "data": {"input": {"raw_docs": []}}}
    assert content(ss.status_frame(event)) == {"step": "found", "documents": 0, "passages": 0}
    event = {"event": "on_chain_end", "name": "format_docs", "data": {}}
    assert content(ss.status_frame(event)) == {"step": "found", "documents": 0, "passages": 0}


def test_writing_when_the_answer_node_starts():
    event = {"event": "on_chain_start", "name": "generate", "data": {}}
    assert content(ss.status_frame(event)) == {"step": "writing"}


def test_other_events_give_no_frame():
    for event in [
        {"event": "on_chat_model_stream", "name": "ChatOpenAI", "metadata": {"langgraph_node": "generate"}, "data": {}},
        {"event": "on_chain_end", "name": "generate", "data": {"output": {}}},
        {"event": "on_chain_start", "name": "format_docs", "data": {}},
        {"event": "on_chain_end", "name": "get_docs", "data": {}},
        {
            "event": "on_chain_start",
            "name": "RunnableSequence",
            "metadata": {"langgraph_node": "rewrite_follow_up_question"},
            "data": {},
        },
        {},
    ]:
        assert ss.status_frame(event) is None


def test_frames_are_ascii_json_like_the_other_frames():
    event = {
        "event": "on_chain_end",
        "name": "rewrite_follow_up_question",
        "data": {"output": {"standalone_question": "Gilt das für Beschäftigte?", "intent": "lookup"}},
    }
    frame = ss.status_frame(event)
    assert fr.encode(frame).isascii()
    assert content(frame)["question"] == "Gilt das für Beschäftigte?"


def chunk_frames(content, strip_cjk=True):
    """The frames run_stream makes of one streamed chunk of the generate node."""
    event = {"event": "on_chat_model_stream", "name": "x", "metadata": {"langgraph_node": "generate"}}
    event["data"] = {"chunk": AIMessageChunk(content=content)}

    class Chain:
        async def astream_events(self, *args, **kwargs):
            yield event

    async def run():
        stream = ss.run_stream(
            Chain(),
            {},
            {},
            formats_on=False,
            variant=None,
            sources_mode="before",
            none_text=lambda language: "",
            result=ss.StreamResult(),
            user="u",
            strip_cjk=strip_cjk,
        )
        return [fr.encode(f) async for f in stream if not isinstance(f, str)]

    return asyncio.run(run())


def test_a_text_chunk_is_sent_as_it_is():
    assert chunk_frames("Die Antwort") == [fr.encode(fr.Chunk(content="Die Antwort"))]


def test_foreign_script_stays_unless_the_card_strips_it():
    assert chunk_frames("Der Urlaub 中文", strip_cjk=False) == [fr.encode(fr.Chunk(content="Der Urlaub 中文"))]
    assert chunk_frames("Der Urlaub 中文") == [fr.encode(fr.Chunk(content="Der Urlaub "))]


def test_a_chunk_of_content_blocks_is_sent_as_its_text():
    blocks = [
        {"type": "text", "text": "Die "},
        {"type": "reasoning", "reasoning": "hm"},
        {"type": "text", "text": "Antwort"},
    ]
    assert chunk_frames(blocks) == [fr.encode(fr.Chunk(content="Die Antwort"))]
    assert chunk_frames([{"type": "reasoning", "reasoning": "nur Denken"}]) == []  # no text: no frame


def test_the_no_sources_note_is_in_the_language_of_the_question():
    def note(state):
        event = {
            "event": "on_chain_end",
            "name": "format_docs",
            "metadata": {"langgraph_node": "format_docs"},
            "data": {"input": {"raw_docs": [], **state}, "output": {"grouped_raw_docs": []}},
        }

        class Chain:
            async def astream_events(self, *args, **kwargs):
                yield event

        async def collect():
            stream = ss.run_stream(
                Chain(),
                {},
                {},
                formats_on=False,
                variant=None,
                sources_mode="before",
                none_text={"de": "keine", "en": "none"}.__getitem__,
                result=ss.StreamResult(),
                user="u",
                strip_cjk=True,
            )
            return [f for f in [x async for x in stream] if isinstance(f, fr.Docs)]

        return asyncio.run(collect())[0].content[0]["page_content"]

    assert note({"language": "en"}) == "none"
    assert note({"language": "de"}) == "keine"
    assert note({"messages": [SimpleNamespace(content="How do I apply for leave?")]}) == "none"
    assert note({}) == "keine"


def test_run_stream_takes_the_scope_fallback_from_the_search_node():
    def run(output):
        event = {"event": "on_chain_end", "name": "get_docs", "metadata": {}, "data": {"output": output}}

        class Chain:
            async def astream_events(self, *args, **kwargs):
                yield event

        result = ss.StreamResult()

        async def collect():
            stream = ss.run_stream(
                Chain(),
                {},
                {},
                formats_on=False,
                variant=None,
                sources_mode="none",
                none_text=lambda language: "",
                result=result,
                user="u",
                strip_cjk=True,
            )
            return [f async for f in stream]

        asyncio.run(collect())
        return result.scope_fallback

    assert run({"raw_docs": ["d"], "scope_fallback": True}) is True
    assert run({"raw_docs": [], "scope_fallback": False}) is False
    assert run({"raw_docs": ["d"]}) is False  # a searching intent: the key is absent


def test_run_stream_records_the_turns_results():
    answer_msg = SimpleNamespace(content="Antwort [1].", type="ai")
    events = [
        {
            "event": "on_chain_end",
            "name": "rewrite_follow_up_question",
            "metadata": {},
            "data": {
                "output": {
                    "answer_shape": "steps",
                    "intent": "lookup",
                    "standalone_question": "S?",
                    "language": "de",
                }
            },
        },
        {
            "event": "on_chain_end",
            "name": "format_docs",
            "metadata": {"langgraph_node": "format_docs"},
            "data": {
                "input": {"raw_docs": ["raw"]},
                "output": {"grouped_raw_docs": ["grouped"], "formatted_docs": "f"},
            },
        },
        {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {"langgraph_node": "generate"},
            "data": {"chunk": SimpleNamespace(text="Antwort")},
        },
        {
            "event": "on_chain_end",
            "name": "generate",
            "metadata": {"langgraph_node": "generate"},
            "data": {
                "output": {
                    "messages": [answer_msg],
                    "generation": "Antwort [1].",
                    "context_sources": [{"n": 1}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 30},
                }
            },
        },
    ]

    class Chain:
        async def astream_events(self, inputs, config, version):
            for e in events:
                yield e

    def run(result):
        async def collect():
            stream = ss.run_stream(
                Chain(),
                {},
                {},
                formats_on=True,
                variant=None,
                sources_mode="none",
                none_text=lambda language: "keine",
                result=result,
                user="u",
                strip_cjk=True,
            )
            return [f async for f in stream]

        asyncio.run(collect())

    result = ss.StreamResult()
    run(result)
    assert result.analysis["standalone_question"] == "S?" and result.requested_shape == "steps"
    assert result.retrieved_docs == ["raw"] and result.candidates == ["grouped"]
    assert result.answer == "Antwort [1]." and result.sources == [{"n": 1}] and result.first_chunk is not None
    assert result.usage == {"prompt_tokens": 120, "completion_tokens": 30}  # the turn's usage, for the store
    events[-1]["data"]["output"] = {"messages": [], "generation": "Fallback", "context_sources": []}
    fallback = ss.StreamResult()
    run(fallback)
    assert fallback.answer is None  # a fallback text is no model message


def test_text_the_node_appended_after_the_stream_is_sent_as_one_more_chunk():
    stream_event = {
        "event": "on_chat_model_stream",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": "generate"},
        "data": {"chunk": AIMessageChunk(content="Die Antwort")},
    }

    def end(generation):
        return {
            "event": "on_chain_end",
            "name": "generate",
            "metadata": {"langgraph_node": "generate"},
            "data": {"output": {"generation": generation, "messages": [], "context_sources": []}},
        }

    def sent(generation):
        class Chain:
            async def astream_events(self, *args, **kwargs):
                yield stream_event
                yield end(generation)

        async def run():
            stream = ss.run_stream(
                Chain(),
                {},
                {},
                formats_on=False,
                variant=None,
                sources_mode="before",
                none_text=lambda language: "",
                result=ss.StreamResult(),
                user="u",
                strip_cjk=True,
            )
            return [fr.wire(f)["content"] async for f in stream if isinstance(f, fr.Chunk)]

        return asyncio.run(run())

    assert sent("Die Antwort") == ["Die Antwort"]
    assert sent("Die Antwort\n\nHinweis: x") == ["Die Antwort", "\n\nHinweis: x"]


def test_foreign_script_characters_are_dropped_from_the_chunks_and_the_stored_answer_with_one_warning(caplog):
    def stream_event(content):
        return {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {"langgraph_node": "generate"},
            "data": {"chunk": AIMessageChunk(content=content)},
        }

    raw = "Der Urlaub 中文 gilt.\n\nHinweis かな"
    end = {
        "event": "on_chain_end",
        "name": "generate",
        "metadata": {"langgraph_node": "generate"},
        "data": {"output": {"generation": raw, "messages": [AIMessage(content=raw)], "context_sources": []}},
    }

    class Chain:
        async def astream_events(self, *args, **kwargs):
            yield stream_event("Der Urlaub 中")
            yield stream_event("文 gilt.")
            yield end

    result = ss.StreamResult()

    async def run():
        stream = ss.run_stream(
            Chain(),
            {},
            {},
            formats_on=False,
            variant=None,
            sources_mode="before",
            none_text=lambda language: "",
            result=result,
            user="u",
            strip_cjk=True,
        )
        return [fr.wire(f)["content"] async for f in stream if isinstance(f, fr.Chunk)]

    with caplog.at_level("WARNING"):
        sent = asyncio.run(run())
    assert "".join(sent) == result.answer == "Der Urlaub  gilt.\n\nHinweis "
    warnings = [r.getMessage() for r in caplog.records if "[ForeignScript]" in r.getMessage()]
    assert len(warnings) == 1 and "中" not in warnings[0]


def test_any_chunk_resets_idle_timer():
    """A chunk without text (a reasoning model thinking) is activity for the guard, though nothing of it is shown."""
    from assistant_api.chat.guard import PROGRESS

    def seen(chunk):
        event = {"event": "on_chat_model_stream", "name": "x", "metadata": {"langgraph_node": "generate"}}
        event["data"] = {"chunk": chunk}

        class Chain:
            async def astream_events(self, *args, **kwargs):
                yield event

        async def run():
            stream = ss.run_stream(
                Chain(),
                {},
                {},
                formats_on=False,
                variant=None,
                sources_mode="before",
                none_text=lambda language: "",
                result=ss.StreamResult(),
                user="u",
                strip_cjk=True,
            )
            return [f async for f in stream]

        return asyncio.run(run())

    thinking = AIMessageChunk(content="", additional_kwargs={"reasoning_content": "denke"})
    assert seen(thinking) == [PROGRESS]  # activity, no frame: the reasoning text is neither sent nor kept
    assert seen(AIMessageChunk(content=[{"type": "reasoning", "reasoning": "hm"}])) == [PROGRESS]
    assert seen(AIMessageChunk(content="Text")) == [fr.Chunk(content="Text")]


def test_a_failure_after_part_of_the_answer_follows_the_part_with_the_fallback_text():
    def sent(streamed_text):
        events = [
            {
                "event": "on_chat_model_stream",
                "name": "x",
                "metadata": {"langgraph_node": "generate"},
                "data": {"chunk": AIMessageChunk(content=streamed_text)},
            }
            if streamed_text
            else {"event": "on_chain_start", "name": "x", "metadata": {}, "data": {}},
            {
                "event": "on_chain_end",
                "name": "generate",
                "metadata": {"langgraph_node": "generate"},
                "data": {"output": {"generation": "Dauert zu lange.", "messages": [], "context_sources": []}},
            },
        ]

        class Chain:
            async def astream_events(self, *args, **kwargs):
                for event in events:
                    yield event

        async def run():
            stream = ss.run_stream(
                Chain(),
                {},
                {},
                formats_on=False,
                variant=None,
                sources_mode="none",
                none_text=lambda language: "",
                result=ss.StreamResult(),
                user="u",
                strip_cjk=True,
            )
            return [fr.wire(f)["content"] async for f in stream if isinstance(f, fr.Chunk)]

        return asyncio.run(run())

    assert sent("Ein Teil") == ["Ein Teil", "\n\nDauert zu lange."]
    assert sent("") == ["Dauert zu lange."]


def test_the_urgent_contact_line_reaches_the_page_after_a_streamed_text_and_after_a_failure():
    from rag_core.domain.notes import urgent_line

    line = "Notrufe: 112 (Feuerwehr) und 110 (Polizei)."

    def sent(streamed_text, generation, produced):
        events = [
            {
                "event": "on_chat_model_stream",
                "name": "x",
                "metadata": {"langgraph_node": "generate"},
                "data": {"chunk": AIMessageChunk(content=streamed_text)},
            },
            {
                "event": "on_chain_end",
                "name": "generate",
                "metadata": {"langgraph_node": "generate"},
                "data": {"output": {"generation": generation, "messages": produced, "context_sources": []}},
            },
        ]

        class Chain:
            async def astream_events(self, *args, **kwargs):
                for event in events:
                    yield event

        async def run():
            stream = ss.run_stream(
                Chain(),
                {},
                {},
                formats_on=False,
                variant=None,
                sources_mode="none",
                none_text=lambda language: "",
                result=ss.StreamResult(),
                user="u",
                strip_cjk=True,
            )
            return [fr.wire(f)["content"] async for f in stream if isinstance(f, fr.Chunk)]

        return asyncio.run(run())

    # the model's text begins and ends with whitespace: the stored text keeps it, so the line is the chunk after it
    text = "\n Rufen Sie 112 an. \n"
    assert urgent_line(text, line) == line
    assert sent(text, text + "\n\n" + line, [AIMessage(content=text + "\n\n" + line)]) == [text, "\n\n" + line]
    # a failure after part of the text: the fallback text and the line follow it
    assert sent("Ein Teil ", "Dauert zu lange.\n\n" + line, []) == ["Ein Teil ", "\n\nDauert zu lange.\n\n" + line]
