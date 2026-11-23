import hashlib
import json

import httpx

from evals import checks, sse
from evals import runner as rp


def test_parse_sse_frames_without_terminators():
    buf = 'data: {"type": "chunk", "content": "Hal"}data: {"type": "chunk", "content": "lo"}'
    events, rest = sse.parse_sse(buf)
    assert [e["content"] for e in events] == ["Hal", "lo"] and rest == ""


def test_parse_sse_incomplete_tail_is_kept():
    events, rest = sse.parse_sse('data: {"type": "chunk", "content": "a"}data: {"type": "ch')
    assert len(events) == 1 and rest == 'data: {"type": "ch'
    events2, rest2 = sse.parse_sse(rest + 'unk", "content": "b"}')
    assert events2[0]["content"] == "b" and rest2 == ""


def test_parse_sse_ignores_data_marker_inside_strings():
    events, _ = sse.parse_sse('data: {"type": "chunk", "content": "x data: y"}data: {"type": "error", "content": "e"}')
    assert [e["type"] for e in events] == ["chunk", "error"]


def test_parse_sse_accepts_blank_line_terminators():
    events, _ = sse.parse_sse('data: {"type": "chunk", "content": "a"}\n\ndata: {"type": "chunk", "content": "b"}\n\n')
    assert len(events) == 2


def test_a_source_is_internal_only_by_its_audience_metadata():
    url = "https://docs.example.org/x"
    docs = [
        {"page_content": "", "metadata": {"url": url, "audience": "internal"}},
        {"page_content": "", "metadata": {"url": url}},
    ]
    assert [s["internal"] for s in rp.sources_from_event(docs)] == [True, False]


def test_sources_carry_the_section_identity():
    docs = [
        {
            "page_content": "Abschnitt A",
            "metadata": {"download_url": "https://docs.example.org/x.pdf", "part_index": 3, "page_number": 7},
        },
        {"page_content": "Erster Abschnitt", "metadata": {"url": "https://docs.example.org/a", "part_index": 0}},
        {"page_content": "Seite", "metadata": {"url": "https://docs.example.org/y", "page_number": 2}},
        {"page_content": "Ohne", "metadata": {"url": "https://docs.example.org/z"}},
    ]

    def sha(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert [s["section"] for s in rp.sources_from_event(docs)] == [
        "https://docs.example.org/x.pdf|3|" + sha("Abschnitt A"),
        "https://docs.example.org/a|0|" + sha("Erster Abschnitt"),
        "https://docs.example.org/y|2|" + sha("Seite"),
        "https://docs.example.org/z||" + sha("Ohne"),
    ]


def test_evaluate_digits_and_contains():
    src = [{"url": "https://docs.example.org/leben/personalabteilung", "title": "", "internal": False}]
    c = checks.evaluate_turn(
        "Sie erreichen die Personalabteilung unter 0123 45-6789 [1].",
        src,
        [],
        {"digits_any": ["0123456789"], "source_url_any": ["personalabteilung"]},
    )
    assert c["auto_pass"], c["failed"]
    assert c["citations"] == [1]


def test_evaluate_flags_style_and_leaks():
    src = [{"url": "https://docs.example.org/internal/a", "title": "", "internal": True}]
    c = checks.evaluate_turn("Klar, du kannst das so machen 😊 [Page 3]", src, [], {})
    assert set(c["failed"]) == {"emoji", "informal_address", "leak"} and c["internal_sources"] == 1


def test_evaluate_no_sources_and_contains_none():
    c = checks.evaluate_turn(
        "Der Präsident ist Mustermann.",
        [{"url": "u", "title": "", "internal": False}],
        [],
        {"no_sources": True, "contains_none": ["Mustermann"]},
    )
    assert "no_sources" in c["failed"] and "contains_none:Mustermann" in c["failed"]


def test_build_question_repeat():
    q = rp.build_question({"q": "Frage?", "repeat": {"text": "abc", "times": 3}})
    assert q == "abc abc abc Frage?"


def test_run_folder_names_carry_seconds():
    assert len(rp.run_stamp()) == len("20260925-151612")


def test_a_source_link_is_no_leak_the_other_markers_are():
    link = "Siehe https://example.org/personal/x/?layout=file (Seite 2)."
    assert "leak" not in checks.evaluate_turn(link, [], [], {})["failed"]
    assert "leak" in checks.evaluate_turn("Laut den Dokumentenauszügen gilt das.", [], [], {})["failed"]


def test_non_latin_script_in_an_answer_is_flagged():
    # Qwen can switch language mid-sentence ("明确"); umlauts, € and dashes are fine.
    assert "non_latin" in checks.evaluate_turn("Die Regel gilt 明确 ab 2024.", [], [], {})["failed"]
    assert (
        "non_latin"
        not in checks.evaluate_turn("Größe: 5 € – gültig ab 01.06.2024. Yes, it applies.", [], [], {})["failed"]
    )


def test_source_url_page_match():
    """'source_url_match': 'page' (G31, G32, G37, G38): a section path passes only on its own page (accordion anchors of
    that page included), not on sub-pages; the default substring rule is unchanged."""

    def src(url):
        return [{"url": url, "title": "", "internal": False}]

    exp = {"source_url_any": ["docs.example.org/leben/standesamt"], "source_url_match": "page"}
    assert (
        "source_url_any"
        not in checks.evaluate_turn("x", src("https://docs.example.org/leben/standesamt"), [], exp)["failed"]
    )
    assert (
        "source_url_any"
        not in checks.evaluate_turn("x", src("https://docs.example.org/leben/standesamt/"), [], exp)["failed"]
    )
    assert (
        "source_url_any"
        not in checks.evaluate_turn("x", src("https://docs.example.org/leben/standesamt#Eheschliessung"), [], exp)[
            "failed"
        ]
    )
    assert (
        "source_url_any"
        in checks.evaluate_turn("x", src("https://docs.example.org/leben/standesamt/geburt"), [], exp)["failed"]
    )
    loose = {"source_url_any": ["docs.example.org/leben/standesamt"]}
    assert (
        "source_url_any"
        not in checks.evaluate_turn("x", src("https://docs.example.org/leben/standesamt/geburt"), [], loose)["failed"]
    )
    plan = {"source_url_any": ["maps.example.org/#m=osm&zoomto=true&uid=358172"], "source_url_match": "page"}
    assert (
        "source_url_any"
        not in checks.evaluate_turn("x", src("https://maps.example.org/#m=osm&zoomto=true&uid=358172"), [], plan)[
            "failed"
        ]
    )
    assert (
        "source_url_any"
        in checks.evaluate_turn("x", src("https://maps.example.org/#m=osm&zoomto=true&uid=3581721"), [], plan)["failed"]
    )


def test_ask_records_the_bytes_exactly_and_the_http_facts_without_the_token(tmp_path, lookup_body):
    pieces = [lookup_body[:7], lookup_body[7:61], lookup_body[61:]]  # read boundaries inside a frame and a marker
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream; charset=utf-8"}, content=iter(pieces)
            )
        )
    )
    stem = str(tmp_path / "G01_t1")
    recorded = rp.ask(client, "http://backend", "secret-token", "Frage?", "u", 10, turn_id="tid", meta={}, record=stem)
    plain = rp.ask(client, "http://backend", "secret-token", "Frage?", "u", 10, turn_id="tid", meta={})
    assert (tmp_path / "G01_t1.raw").read_bytes() == lookup_body
    sidecar = (tmp_path / "G01_t1.json").read_text(encoding="utf-8")
    assert "secret-token" not in sidecar
    facts = json.loads(sidecar)
    assert facts["status"] == 200 and facts["params"] == {
        "question": "Frage?",
        "conversation_id": "u",
        "turn_id": "tid",
    }
    assert facts["headers"]["content-type"].startswith("text/event-stream")
    assert recorded[0] == plain[0] == "Die Antwort [1]." and recorded[5] == plain[5]


def test_ask_hands_the_status_steps_to_meta(lookup_body):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=lookup_body)))
    meta = {}
    rp.ask(client, "http://b", "t", "Frage?", "u", 10, meta=meta)
    assert meta["steps"] == ["analysing", "understood", "searching", "found", "writing"] and meta["citations"] == [1]


def test_rows_carry_variant_ok():
    def ask_fn(question, user, opts):
        if "variant" in opts and opts["variant"]:
            opts["meta"]["variant_ok"] = False
        return "x [1]", [], [], 0.1, 0.2, ["chunk"]

    conv = {"id": "FV9", "turns": [{"q": "Frage"}, {"variant": "shorter", "expect": {"variant_ok": True}}]}
    rows = rp.run_conversation(conv, ask_fn, 0.0, "t", "s", lambda row: None, formats="v1")
    assert [r["variant_ok"] for r in rows] == [None, False] and rows[1]["checks"]["failed"] == ["variant"]


def test_ask_hands_the_done_frames_generation_and_prompts_to_meta(lookup_body):
    done = b'data: {"type": "done", "content": {"outcome": "completed", "generation": 3, "prompts": {"p": 2}}}\n\n'
    with_done = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=lookup_body + done))
    )
    meta = {}
    rp.ask(with_done, "http://b", "t", "Frage?", "u", 10, meta=meta)
    assert meta["generation"] == 3 and meta["prompts"] == {"p": 2}
    plain = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=lookup_body)))
    meta = {}
    rp.ask(plain, "http://b", "t", "Frage?", "u", 10, meta=meta)
    assert "generation" not in meta and "prompts" not in meta


def test_ask_keeps_the_scope_fallback_of_the_done_frame_in_meta_and_rows(lookup_body):
    done = b'data: {"type": "done", "content": {"outcome": "completed", "scope_fallback": true}}\n\n'
    with_done = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=lookup_body + done))
    )
    meta = {}
    rp.ask(with_done, "http://b", "t", "Frage?", "u", 10, meta=meta)
    assert meta["scope_fallback"] is True
    plain = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=lookup_body)))
    meta = {}
    rp.ask(plain, "http://b", "t", "Frage?", "u", 10, meta=meta)
    assert "scope_fallback" not in meta

    def fell_back(question, user, opts):
        opts["meta"]["scope_fallback"] = True
        return "x [1]", [], [], 0.1, 0.2, ["chunk"]

    def ordinary(question, user, opts):
        return "x [1]", [], [], 0.1, 0.2, ["chunk"]

    conv = {"id": "G9", "turns": [{"q": "Frage"}]}
    [row] = rp.run_conversation(conv, fell_back, 0.0, "t", "s", lambda row: None)
    assert row["scope_fallback"] is True
    [row] = rp.run_conversation(conv, ordinary, 0.0, "t", "s", lambda row: None)
    assert row["scope_fallback"] is None


def test_rows_carry_generation_and_prompts_or_none():
    def recorded(question, user, opts):
        opts["meta"].update(generation=3, prompts={"p": 2})
        return "x [1]", [], [], 0.1, 0.2, ["chunk"]

    def unrecorded(question, user, opts):
        return "x [1]", [], [], 0.1, 0.2, ["chunk"]

    conv = {"id": "G9", "turns": [{"q": "Frage"}]}
    [row] = rp.run_conversation(conv, recorded, 0.0, "t", "s", lambda row: None)
    assert (row["generation"], row["prompts"]) == (3, {"p": 2})
    [row] = rp.run_conversation(conv, unrecorded, 0.0, "t", "s", lambda row: None)
    assert (row["generation"], row["prompts"]) == (None, None)


def test_each_set_writes_label_set_stamp(env, fake_backend):
    assert rp.main(["gold", "figures", "--label", "t", "--ids", "G01,F01"], client_factory=fake_backend()) == 0
    assert sorted(p.name.rsplit("_", 1)[0] for p in env.iterdir()) == ["t-figures", "t-gold"]
    rows = [
        json.loads(line)
        for p in env.iterdir()
        for line in (p / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(r["id"] for r in rows) == ["F01", "G01"] and all("variant_ok" in r for r in rows)
    assert all(r["sources"][0]["section"].startswith("https://docs.example.org/a|0|") for r in rows)


def test_the_formats_set_sends_formats_v1_by_itself(env, fake_backend):
    seen = []
    assert rp.main(["formats", "--label", "t", "--ids", "FT01"], client_factory=fake_backend(seen)) == 0
    assert seen[0]["formats"] == "v1"
    seen.clear()
    assert rp.main(["gold", "--label", "u", "--ids", "G01"], client_factory=fake_backend(seen)) == 0
    assert "formats" not in seen[0]


def test_a_probe_file_runs_with_its_own_label_and_the_formats_flag(env, fake_backend, tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"conversations": [{"id": "Q1", "turns": [{"q": "Frage?"}]}]}), encoding="utf-8")
    seen = []
    assert rp.main(["--probes", str(plan), "--label", "rec", "--formats", "v1"], client_factory=fake_backend(seen)) == 0
    assert seen[0]["formats"] == "v1" and [p.name.rsplit("_", 1)[0] for p in env.iterdir() if p.is_dir()] == ["rec"]


def test_record_keeps_one_raw_and_one_sidecar_per_turn(env, fake_backend, lookup_body):
    record = env / "rec"
    assert (
        rp.main(["gold", "--label", "t", "--ids", "G01", "--record", str(record)], client_factory=fake_backend()) == 0
    )
    assert sorted(p.name for p in record.iterdir()) == ["G01_t1.json", "G01_t1.raw"]
    assert (record / "G01_t1.raw").read_bytes() == lookup_body


def test_a_run_without_a_token_writes_nothing(env, fake_backend, monkeypatch):
    monkeypatch.delenv("RAG__EVALS__TOKEN")
    assert rp.main(["gold", "--label", "t"], client_factory=fake_backend()) == 2
    assert list(env.iterdir()) == []


def test_a_run_with_an_empty_token_writes_nothing(env, fake_backend, monkeypatch):
    monkeypatch.setenv("RAG__EVALS__TOKEN", "")
    assert rp.main(["gold", "--label", "t"], client_factory=fake_backend()) == 2
    assert list(env.iterdir()) == []
