"""The harness's client side: frames read to the last complete one and written back byte-exact, a reply's frame order, and the
index snapshot's normalisation."""

import json

from rag_testing import harness, index_view


def test_frames_are_read_to_the_last_complete_one_and_written_back_byte_exact():
    raw = harness.serialise(
        [{"type": "status", "content": {"step": "analysing"}}, {"type": "chunk", "content": "Ä [1]"}]
    )
    assert (
        raw
        == 'data: {"type": "status", "content": {"step": "analysing"}}data: {"type": "chunk", "content": "\\u00c4 [1]"}'
    )
    items, end = harness.frames(raw + 'data: {"type": "ch')
    assert harness.serialise(items) == raw and end == len(raw)
    assert harness.path_of(items + [{"type": "chunk", "content": "x"}, {"type": "citations", "content": []}]) == [
        "status:analysing",
        "chunk*",
        "citations",
    ]


def test_the_snapshot_drops_the_run_and_numbers_children_by_their_parent():
    parents = [
        ("section-b", "Text B", {"url": "u", "page_number": 2, "section_id": "section-b", "scrapedAt": "now"}),
        ("section-a", "Text A", json.dumps({"url": "u", "page_number": 1, "crawl_date": "today"})),
    ]
    children = [("K", "[0.1,0.2]", {"section_id": "section-b", "chunk_index": 0, "unique_id": "section-b_0"})]
    snap = index_view.snapshot(parents, children)
    assert [p["position"] for p in snap["parents"]] == [1, 2]
    assert snap["parents"][1]["metadata"] == {"url": "u", "page_number": 2}
    assert snap["children"] == [
        {
            "parent": 1,
            "sha": index_view.sha("K"),
            "embedding": index_view.sha("[0.1,0.2]"),
            "metadata": {"chunk_index": 0},
        }
    ]


V1 = [
    {"type": "meta", "content": {"format": 1, "turn_id": "t", "trace_id": None}},
    {"type": "status", "content": {"step": "analysing"}},
    {"type": "error", "content": "timeout", "code": "timeout"},
    {"type": "chunk", "content": "Die Antwort dauert"},
    {"type": "variant", "content": {"name": "table", "ok": False}, "code": "failed"},
    {"type": "done", "content": {"outcome": "failed"}},
]


def test_serialise_v1_writes_the_terminated_form_and_frames_reads_it():
    raw = harness.serialise_v1(V1)
    assert raw.startswith('data: {"type": "meta", "content": {"format": 1, "turn_id": "t", "trace_id": null}}\n\n')
    assert raw.endswith('data: {"type": "done", "content": {"outcome": "failed"}}\n\n')
    items, end = harness.frames(raw)
    assert items == V1 and end == len(raw) - 2


def test_the_legacy_view_drops_meta_done_and_code():
    assert harness.legacy(V1) == [
        {"type": "status", "content": {"step": "analysing"}},
        {"type": "error", "content": "timeout"},
        {"type": "chunk", "content": "Die Antwort dauert"},
        {"type": "variant", "content": {"name": "table", "ok": False}},
    ]


def test_a_reply_keeps_the_stream_and_shows_the_legacy_view():
    reply = harness.Reply(200, stream=list(V1), raw=harness.serialise_v1(V1))
    assert reply.types == ["status", "error", "chunk", "variant"]
    assert reply.path == ["status:analysing", "error", "chunk*", "variant"]
    assert reply.first("error") == "timeout" and reply.text == "Die Antwort dauert"
    assert [f["type"] for f in reply.stream][::5] == ["meta", "done"]
