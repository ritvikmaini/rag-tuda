"""The transcript redaction: equal length and character class, markers, step numbers and fence info kept, a fixed point of
itself, and only where document text is."""

import json

from hypothesis import given, strategies as st

from rag_testing import harness, redact


def test_the_mask_keeps_markers_step_numbers_and_fence_info_and_nothing_else():
    text = "Antwort mit 42 Plätzen [1, 2].\n\n1. Schritt [3]\n12) Zwölf\n```mindmap\n# Thema Ölbaum\n```\nÄrger 7"
    assert redact.mask(text) == (
        "Xxxxxxx xxx 00 Xxxxxxx [1, 2].\n\n1. Xxxxxxx [3]\n12) Xxxxx\n```mindmap\n# Xxxxx Xxxxxx\n```\nXxxxx 0"
    )


@given(st.text(max_size=300))
def test_the_mask_keeps_the_length_and_is_a_fixed_point(text):
    masked = redact.mask(text)
    assert len(masked) == len(text) and redact.mask(masked) == masked


def test_a_frame_is_redacted_where_document_text_is_and_nowhere_else():
    docs = {
        "type": "docs",
        "content": [
            {
                "page_content": "Text 5",
                "metadata": {
                    "title": "Titel 5",
                    "url": "https://a/5",
                    "matched_child_text": "Kind 5",
                    "context_header": "Kopf 5",
                    "page_number": 3,
                },
            }
        ],
    }
    assert redact.redact_frame(docs)["content"] == [
        {
            "page_content": "Xxxx 0",
            "metadata": {
                "title": "Titel 5",
                "url": "https://a/5",
                "matched_child_text": "Xxxx 0",
                "context_header": "Xxxx 0",
                "page_number": 3,
            },
        }
    ]
    citations = {"type": "citations", "content": [{"n": 1, "title": "T", "quote": "Zitat 9"}]}
    assert redact.redact_frame(citations)["content"] == [{"n": 1, "title": "T", "quote": "Xxxxx 0"}]
    status = {"type": "status", "content": {"step": "understood", "question": "Haushalt 2026?"}}
    assert redact.redact_frame(status) == status


def test_a_follow_up_turns_status_question_and_queries_are_masked_and_a_first_turns_kept():
    understood = {
        "type": "status",
        "content": {"step": "understood", "question": "Gilt das für 418 Tage?", "intent": "lookup"},
    }
    searching = {"type": "status", "content": {"step": "searching", "queries": ["Tage 418", "Urlaub"]}}
    assert redact.redact_frame(understood) == understood and redact.redact_frame(searching) == searching
    assert redact.redact_frame(understood, follow_up=True)["content"] == {
        "step": "understood",
        "question": "Xxxx xxx xxx 000 Xxxx?",
        "intent": "lookup",
    }
    assert redact.redact_frame(searching, follow_up=True)["content"] == {
        "step": "searching",
        "queries": ["Xxxx 000", "Xxxxxx"],
    }


def test_strings_nested_under_a_text_key_are_masked():
    docs = {
        "type": "docs",
        "content": [
            {
                "page_content": "P",
                "metadata": {"section_path": ["Kapitel 1", "Teil 2"], "matched_child_text": {"text": "Kind 5", "n": 3}},
            }
        ],
    }
    meta = redact.redact_frame(docs)["content"][0]["metadata"]
    assert meta == {"section_path": ["Xxxxxxx 0", "Xxxx 0"], "matched_child_text": {"text": "Xxxx 0", "n": 3}}
    assert redact.leaks(redact.redact_frame(docs)) == [] and len(redact.leaks(docs)) == 4


def test_leaks_names_unmasked_text_and_long_metadata_of_any_type_under_an_unknown_key():
    docs = {
        "type": "docs",
        "content": [
            {
                "page_content": "x",
                "metadata": {"title": "T" * 120, "notes": ["n" * 90], "extra": {"a": "y" * 90}, "page_number": 3},
            }
        ],
    }
    assert redact.leaks(docs) == [
        "docs metadata 'notes' carries 94 characters: redact it or list it",
        "docs metadata 'extra' carries 99 characters: redact it or list it",
    ]
    assert redact.leaks({"type": "chunk", "content": "Text 9"}) == ["not masked: 'Text 9'"]
    assert redact.leaks({"type": "status", "content": {"question": "Frage 2"}}, follow_up=True) == [
        "not masked: 'Frage 2'"
    ]
    assert redact.leaks({"type": "status", "content": {"question": "Frage 2"}}) == []


STATUS = {"type": "status", "content": {"step": "understood", "question": "Wie viele Tage gilt das 2026?"}}
ANSWER = {"type": "chunk", "content": "Es sind 30 Tage [1]."}
BUSY = {"type": "error", "content": "busy"}
# plan -> conversation id -> the frames of each turn
RECORDED = {
    "plain": {"p1": [[STATUS, ANSWER], [STATUS, ANSWER]]},
    "formats": {"f1": [[ANSWER]]},
    "busy": {"b1": [[ANSWER]], "b2": [[BUSY]], "b3": [[BUSY]], "b4": [[ANSWER]]},
}


def record(root):
    """The plan files and the raw recordings as `evals run --probes <plan> --record <dir>` leaves them."""
    plans, raw = root / "plans", root / "raw"
    plans.mkdir()
    for plan, conversations in RECORDED.items():
        (raw / plan).mkdir(parents=True)
        spec = {
            "conversations": [{"id": cid, "turns": [{"path": "lookup"}] * len(t)} for cid, t in conversations.items()]
        }
        (plans / ("plan_%s.json" % plan)).write_text(json.dumps(spec), encoding="utf-8")
        for cid, turns in conversations.items():
            for n, items in enumerate(turns, 1):
                stem = raw / plan / ("%s_t%d" % (cid, n))
                stem.with_suffix(".raw").write_text(harness.serialise(items), encoding="utf-8")
                sidecar = {"status": 200, "params": {"conversation_id": "probe", "formats": "v1"}}
                stem.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return raw, plans


def test_the_command_writes_every_turn_redacted_and_the_busy_plans_first_busy_and_first_answered_reply(
    tmp_path, capsys
):
    raw, plans = record(tmp_path)
    out = tmp_path / "out"
    assert redact.main([str(raw), str(plans), str(out)]) == 0
    assert capsys.readouterr().out == "5 transcripts written\n"
    names = ["busy_b1_t1", "busy_b2_t1", "formats_f1_t1", "plain_p1_t1", "plain_p1_t2"]
    assert sorted(p.name for p in out.iterdir()) == sorted(n + s for n in names for s in (".json", ".sse"))
    answer = {"type": "chunk", "content": "Xx xxxx 00 Xxxx [1]."}
    follow_up = {"type": "status", "content": {"step": "understood", "question": "Xxx xxxxx Xxxx xxxx xxx 0000?"}}
    assert (out / "plain_p1_t1.sse").read_text(encoding="utf-8") == harness.serialise([STATUS, answer])
    assert (out / "plain_p1_t2.sse").read_text(encoding="utf-8") == harness.serialise([follow_up, answer])
    meta = {"plan": "busy", "id": "b2", "turn": 1, "path": "busy", "status": 200, "params": {"formats": "v1"}}
    assert json.loads((out / "busy_b2_t1.json").read_text(encoding="utf-8")) == meta
    assert json.loads((out / "busy_b1_t1.json").read_text(encoding="utf-8"))["path"] == "lookup"


def test_the_round_trip_check_writes_nothing_and_names_a_differing_recording_and_a_stopped_requests_tail(
    tmp_path, capsys
):
    raw, plans = record(tmp_path)
    out = tmp_path / "out"
    stopped, spaced = raw / "formats" / "f1_t1.raw", raw / "plain" / "p1_t2.raw"
    stopped.write_text(stopped.read_text(encoding="utf-8") + 'data: {"type": "ch', encoding="utf-8")
    args = [str(raw), str(plans), str(out), "--check-round-trip"]
    assert redact.main(args) == 0
    tail = "%s: 18 bytes after the last complete frame left out (a stopped request)\n" % stopped
    assert capsys.readouterr().out == tail + "round trip: 0 differ\n"
    spaced.write_text(spaced.read_text(encoding="utf-8").replace("data: ", "data:  "), encoding="utf-8")
    assert redact.main(args) == 1
    assert capsys.readouterr().out == "ROUND TRIP DIFFERS: %s\n" % spaced + tail + "round trip: 1 differ\n"
    assert list(out.iterdir()) == []
