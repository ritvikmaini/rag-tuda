import json
from pathlib import Path

import pytest

from rag_tuda import ask, cli, sources

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def fake_stream(frames, seen=None):
    def stream(base_url, token, question, conversation_id, formats=None, filters=None):
        if seen is not None:
            seen.update(
                base_url=base_url, token=token, question=question, conversation_id=conversation_id, filters=filters
            )
        yield from frames

    return stream


def fixture_frames():
    return list(ask.parse((FIX / "stream_mietrecht.txt").read_text(encoding="utf-8").splitlines()))


def test_add_registers_and_points_to_reindex(project, capsys):
    assert cli.main(["add", str(FIX / "sample_docs"), "--name", "house", "--only", "--topic", "house rules"]) == 0
    out = capsys.readouterr().out
    assert "house" in out and "folder" in out and "rag reindex" in out
    reg = sources.load(project)
    assert [s.name for s in reg.sources] == ["house"] and reg.only and reg.topic == "house rules"


def test_add_is_idempotent(project, capsys):
    cli.main(["add", str(FIX / "sample_docs")])
    before = (project / ".rag" / "sources.json").read_text()
    assert cli.main(["add", str(FIX / "sample_docs")]) == 0
    assert (project / ".rag" / "sources.json").read_text() == before


def test_add_rejects_a_bad_target(project, capsys):
    assert cli.main(["add", "ftp://example.org"]) == 1
    assert "http" in capsys.readouterr().err
    assert not (project / ".rag" / "sources.json").exists()


def test_ask_prints_the_rendered_answer(project, monkeypatch, capsys):
    (project / ".rag").mkdir()
    (project / ".rag" / "token").write_text("tok\n")
    seen = {}
    monkeypatch.setattr(ask, "stream", fake_stream(fixture_frames(), seen))
    assert cli.main(["ask", "Welche Frist gilt?"]) == 0
    assert capsys.readouterr().out == ask.render(fixture_frames())
    assert seen["token"] == "tok" and seen["question"] == "Welche Frist gilt?"
    assert seen["base_url"] == "http://127.0.0.1:8700" and seen["conversation_id"].startswith("cli-")


def test_ask_flags_map_to_params(project, monkeypatch, capsys):
    (project / ".rag").mkdir()
    (project / ".rag" / "token").write_text("tok")
    seen = {}
    monkeypatch.setattr(ask, "stream", fake_stream(fixture_frames(), seen))
    assert cli.main(["ask", "q", "--degree", "master", "--programme", "inf-m", "--po", "2015", "--lang", "en"]) == 0
    assert seen["filters"] == {"degree": "master", "programme": "inf-m", "po": 2015, "lang": "en"}
    assert cli.main(["ask", "q"]) == 0
    assert not seen["filters"]  # no flag, no parameter: the backend's defaults apply


@pytest.mark.parametrize("flags", [["--lang", "fr"], ["--degree", "diploma"], ["--po", "x"]])
def test_ask_refuses_a_value_the_flag_does_not_take(project, monkeypatch, capsys, flags):
    monkeypatch.setattr(ask, "stream", fake_stream(fixture_frames()))
    with pytest.raises(SystemExit) as exc:
        cli.main(["ask", "q", *flags])
    assert exc.value.code == 2


def test_ask_json_prints_one_frame_per_line(project, monkeypatch, capsys):
    (project / ".rag").mkdir()
    (project / ".rag" / "token").write_text("tok")
    monkeypatch.setattr(ask, "stream", fake_stream(fixture_frames()))
    assert cli.main(["ask", "q", "--json"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(x)["type"] for x in lines] == [f["type"] for f in fixture_frames()]


def test_ask_without_token_says_what_to_do(project, capsys):
    assert cli.main(["ask", "q"]) == 1
    assert "rag up" in capsys.readouterr().err


def test_ask_exit_code_on_error_frame(project, monkeypatch, capsys):
    (project / ".rag").mkdir()
    (project / ".rag" / "token").write_text("tok")
    frames = [{"type": "error", "code": "busy", "content": "Try again."}]
    monkeypatch.setattr(ask, "stream", fake_stream(frames))
    assert cli.main(["ask", "q"]) == 1
    assert "Error (busy)" in capsys.readouterr().out


def test_ask_connection_error(project, monkeypatch, capsys):
    (project / ".rag").mkdir()
    (project / ".rag" / "token").write_text("tok")

    def stream(*a, **k):
        raise ask.AskError("cannot reach it")
        yield

    monkeypatch.setattr(ask, "stream", stream)
    assert cli.main(["ask", "q"]) == 1
    assert "cannot reach it" in capsys.readouterr().err
