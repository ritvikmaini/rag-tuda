import json
import urllib.parse
from pathlib import Path

import pytest

from rag_tuda import ask

FIXTURE = Path(__file__).parent / "fixtures" / "stream_mietrecht.txt"


def frames(text: str | None = None) -> list[dict]:
    return list(ask.parse((text if text is not None else FIXTURE.read_text(encoding="utf-8")).splitlines()))


def sse(*objs, tail="") -> list[str]:
    out = []
    for o in objs:
        out += ["data: " + json.dumps(o), ""]
    return out + ([tail] if tail else [])


def test_parse_fixture_order_and_types():
    got = [f["type"] for f in frames()]
    assert got == [
        "meta",
        "status",
        "status",
        "status",
        "status",
        "status",
        "shape",
        "docs",
        "chunk",
        "chunk",
        "chunk",
        "chunk",
        "citations",
        "done",
    ]


def test_parse_skips_comments_and_other_fields():
    lines = [": progress", "event: message", "id: 4", "data: " + json.dumps({"type": "chunk", "content": "a"}), ""]
    assert list(ask.parse(lines)) == [{"type": "chunk", "content": "a"}]


def test_parse_accepts_crlf_and_trailing_event_without_blank_line():
    lines = ['data: {"type": "chunk", "content": "a"}\r\n', "\r\n", 'data: {"type": "chunk", "content": "b"}']
    assert [f["content"] for f in ask.parse(lines)] == ["a", "b"]


def test_parse_joins_multiline_data():
    lines = ['data: {"type": "chunk",', 'data: "content": "x"}', ""]
    assert list(ask.parse(lines)) == [{"type": "chunk", "content": "x"}]


def test_parse_drops_unreadable_and_untyped_frames():
    lines = [
        "data: not json",
        "",
        'data: {"no": "type"}',
        "",
        "data: [1]",
        "",
        'data: {"type": "chunk", "content": "ok"}',
        "",
    ]
    assert list(ask.parse(lines)) == [{"type": "chunk", "content": "ok"}]


def test_parse_is_lazy():
    def source():
        yield 'data: {"type": "chunk", "content": "a"}'
        yield ""
        raise AssertionError("read too far")

    assert next(iter(ask.parse(source())))["content"] == "a"


def test_render_fixture():
    text = ask.render(frames())
    steps, answer, sources = text.split("\n\n")
    assert steps.splitlines() == [
        "- analysing",
        "- understood: Welche Frist gilt für die Kündigung durch den Mieter?",
        "- searching: Kündigungsfrist Mieter Wohnraum; § 573c BGB Fristen",
        "- found: 2 documents, 4 passages",
        "- writing",
    ]
    assert answer == (
        "Der Mieter kann spätestens am dritten Werktag eines Monats [1] zum Ablauf des übernächsten Monats kündigen. "
        "Abweichungen zu seinen Lasten sind unwirksam [2]."
    )
    assert sources.splitlines() == [
        "[1] § 573c BGB Fristen der ordentlichen Kündigung - https://example.test/bgb/573c",
        "[2] § 573c Abs. 4 BGB - https://example.test/bgb/573c#4",
    ]
    assert text.endswith("\n")


def test_render_dims_steps_only_with_color():
    plain = ask.render(frames())
    assert "\x1b" not in plain
    colored = ask.render(frames(), color=True)
    assert "\x1b[2m- analysing\x1b[0m" in colored
    assert colored.replace("\x1b[2m", "").replace("\x1b[0m", "") == plain


def test_pieces_stream_before_the_end():
    it = ask.pieces(
        ask.parse(sse({"type": "status", "content": {"step": "analysing"}}, {"type": "chunk", "content": "Hi"}))
    )
    assert next(it) == "- analysing\n"
    assert "".join(it) == "\nHi\n"


def test_render_answer_only_has_no_blank_prefix():
    assert ask.render([{"type": "chunk", "content": "Hallo"}]) == "Hallo\n"


def test_render_source_without_url_or_title():
    cit = {
        "type": "citations",
        "content": [{"n": 1, "url": "", "title": "Nur Titel"}, {"n": 2, "url": "http://x.test/a", "title": ""}],
    }
    out = ask.render([{"type": "chunk", "content": "a"}, cit])
    assert out.splitlines()[-2:] == ["[1] Nur Titel", "[2] http://x.test/a"]


def test_render_sorts_sources_by_n():
    cit = {"type": "citations", "content": [{"n": 2, "title": "B", "url": "u2"}, {"n": 1, "title": "A", "url": "u1"}]}
    assert ask.render([cit]).splitlines() == ["[1] A - u1", "[2] B - u2"]


def test_render_error_and_outcome():
    err = {"type": "error", "code": "busy", "content": "Try again."}
    out = ask.render([{"type": "chunk", "content": "Partial"}, err, {"type": "done", "content": {"outcome": "busy"}}])
    assert out.splitlines()[-1] == "Error (busy): Try again."
    assert "Outcome" not in out  # the error already says it
    out = ask.render([{"type": "done", "content": {"outcome": "too_long"}}])
    assert out == "Outcome: too_long\n"
    assert ask.render([{"type": "done", "content": {"outcome": "completed"}}]) == ""


def test_render_ignores_frames_it_does_not_show():
    ignored = [
        {"type": "meta", "content": {"format": 1, "turn_id": "t", "trace_id": None}},
        {"type": "shape", "content": "table"},
        {"type": "docs", "content": [{"page_content": "x", "metadata": {"section_id": "s"}}]},
        {"type": "variant", "content": {"name": "table", "ok": True}},
        {"type": "status", "content": {"step": "analysing", "x": 1}},
        {"type": "status", "content": {}},
        {"type": "future-frame", "content": 1},
    ]
    assert ask.render(ignored) == "- analysing\n"


def test_failed_helper():
    assert ask.failed(frames()) is False
    assert ask.failed([{"type": "error", "content": "x", "code": "timeout"}]) is True
    assert ask.failed([{"type": "done", "content": {"outcome": "failed"}}]) is True
    assert ask.failed([{"type": "done", "content": {"outcome": "refused"}}]) is False


def test_build_request():
    req = ask.build_request("http://127.0.0.1:8700/", "tok", "Was regelt § 535 BGB?", "cli-1", formats="table")
    parts = urllib.parse.urlsplit(req.full_url)
    assert (parts.scheme, parts.netloc, parts.path) == ("http", "127.0.0.1:8700", "/api/chat")
    assert urllib.parse.parse_qs(parts.query) == {
        "question": ["Was regelt § 535 BGB?"],
        "conversation_id": ["cli-1"],
        "formats": ["table"],
    }
    assert req.get_header("Authorization") == "Bearer tok"
    assert req.get_header("Accept") == "text/event-stream"
    assert req.get_method() == "GET"


def test_build_request_without_formats():
    req = ask.build_request("http://127.0.0.1:8700", "t", "q", "c")
    assert "formats" not in urllib.parse.parse_qs(urllib.parse.urlsplit(req.full_url).query)


def test_stream_reads_lines_from_opener(monkeypatch):
    body = FIXTURE.read_bytes()
    seen = {}

    class Resp:
        def __init__(self):
            self.lines = body.splitlines(keepends=True)

        def __iter__(self):
            return iter(self.lines)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=None):
        seen["auth"] = req.get_header("Authorization")
        return Resp()

    monkeypatch.setattr(ask.urllib.request, "urlopen", fake_open)
    got = list(ask.stream("http://127.0.0.1:8700", "tok", "q", "cli"))
    assert seen["auth"] == "Bearer tok"
    assert got[0]["type"] == "meta" and got[-1]["type"] == "done"
    assert "Kündigungsfrist" in ask.render(got)


def test_stream_turns_http_errors_into_ask_errors(monkeypatch):
    import io
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b""))

    monkeypatch.setattr(ask.urllib.request, "urlopen", boom)
    with pytest.raises(ask.AskError, match="token"):
        list(ask.stream("http://127.0.0.1:8700", "bad", "q", "c"))

    def refused(req, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError())

    monkeypatch.setattr(ask.urllib.request, "urlopen", refused)
    with pytest.raises(ask.AskError, match="rag up"):
        list(ask.stream("http://127.0.0.1:8700", "t", "q", "c"))
