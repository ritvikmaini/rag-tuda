"""The answer stream's frames: the encoded bytes plus the terminator, NaN as null, the schema the page reads."""

import json
import math
from pathlib import Path

import pytest
from assistant_api.sse import frames as fr

ROOT = Path(__file__).resolve().parents[3]


def encoded(kind, content):  # the reference encoding of a frame
    return "data: " + json.dumps({"type": kind, "content": content})


@pytest.mark.parametrize(
    ("frame", "kind", "content"),
    [
        (
            fr.Status(content={"step": "understood", "question": "Wer?", "intent": "lookup"}),
            "status",
            {"step": "understood", "question": "Wer?", "intent": "lookup"},
        ),
        (fr.ShapeFrame(content="table"), "shape", "table"),
        (
            fr.Docs(content=[{"page_content": "Grüße 😀", "metadata": {"section_id": "p", "page_number": 2}}]),
            "docs",
            [{"page_content": "Grüße 😀", "metadata": {"section_id": "p", "page_number": 2}}],
        ),
        (fr.Chunk(content="data: [1] – "), "chunk", "data: [1] – "),
        (
            fr.Citations(
                content=[
                    fr.Citation(n=1, url="u", page=None, title="t", type="T", origin="EXAMPLE", date="2024", quote="q")
                ]
            ),
            "citations",
            [
                {
                    "n": 1,
                    "url": "u",
                    "page": None,
                    "title": "t",
                    "type": "T",
                    "origin": "EXAMPLE",
                    "date": "2024",
                    "quote": "q",
                }
            ],
        ),
        (fr.variant("table", True), "variant", {"name": "table", "ok": True}),
    ],
)
def test_each_frame_is_its_recorded_bytes_plus_the_terminator(frame, kind, content):
    assert fr.encode(frame) == encoded(kind, content) + "\n\n"


def test_code_follows_type_and_content():
    assert (
        fr.encode(fr.Error(content="timeout", code="timeout"))
        == 'data: {"type": "error", "content": "timeout", "code": "timeout"}\n\n'
    )
    assert fr.encode(fr.variant("x" * 30, False, "failed")) == (
        'data: {"type": "variant", "content": {"name": "' + "x" * 20 + '", "ok": false}, "code": "failed"}\n\n'
    )


def test_meta_and_done():
    meta = fr.meta("0b0e7a52-6a1f-4a8e-9c7e-1f2e3d4c5b6a", None, 240.0, 90.0, "qwen3.6:35b-a3b", "ollama")
    assert fr.encode(meta) == (
        'data: {"type": "meta", "content": {"format": 1, "turn_id": "0b0e7a52-6a1f-4a8e-9c7e-1f2e3d4c5b6a", '
        '"trace_id": null, "deadline_s": 240.0, "idle_s": 90.0, "model": "qwen3.6:35b-a3b", "provider": "ollama"}}\n\n'
    )
    assert fr.encode(fr.done("too_long")) == 'data: {"type": "done", "content": {"outcome": "too_long"}}\n\n'


def test_done_names_the_generation_and_prompts_only_when_known():
    assert fr.encode(fr.done("completed", 7, {"rag-generation": 3, "query-analysis": None})) == (
        'data: {"type": "done", "content": {"outcome": "completed", "generation": 7, '
        '"prompts": {"rag-generation": 3, "query-analysis": null}}}\n\n'
    )
    assert "prompts" not in fr.encode(fr.done("busy", 7))
    assert fr.encode(fr.done("too_long")) == 'data: {"type": "done", "content": {"outcome": "too_long"}}\n\n'


def test_done_names_the_scope_fallback_only_when_true():
    """A fallback turn's done frame carries the key, every other keeps its exact bytes."""
    assert fr.encode(fr.done("completed", 7, {"rag-generation": 3}, True)) == (
        'data: {"type": "done", "content": {"outcome": "completed", "generation": 7, '
        '"prompts": {"rag-generation": 3}, "scope_fallback": true}}\n\n'
    )
    assert fr.encode(fr.done("completed", 7, {"rag-generation": 3}, None)) == fr.encode(
        fr.done("completed", 7, {"rag-generation": 3})
    )
    assert fr.encode(fr.done("completed")) == 'data: {"type": "done", "content": {"outcome": "completed"}}\n\n'


def test_nan_and_infinity_go_out_as_null_at_any_depth():
    docs = fr.Docs(
        content=[{"page_content": "x", "metadata": {"cross_encoder_score": math.nan, "s": [math.inf, -math.inf, 1.5]}}]
    )
    assert fr.encode(docs) == (
        'data: {"type": "docs", "content": [{"page_content": "x", "metadata": {"cross_encoder_score": null, "s": [null, null, 1.5]}}]}\n\n'
    )


def test_an_unknown_code_or_shape_is_refused():
    with pytest.raises(ValueError):
        fr.Error(content="x", code="nope")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError):
        fr.ShapeFrame(content="poster")  # pyright: ignore[reportArgumentType]


def test_the_committed_schema_is_the_models_schema():
    committed = json.loads((ROOT / "apps" / "assistant_api" / "stream_v1.schema.json").read_text(encoding="utf-8"))
    assert committed == fr.schema()
    assert set(committed["discriminator"]["mapping"]) == {
        "meta",
        "status",
        "shape",
        "docs",
        "chunk",
        "citations",
        "variant",
        "error",
        "done",
    }
