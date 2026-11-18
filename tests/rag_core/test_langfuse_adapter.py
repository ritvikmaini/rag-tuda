"""Langfuse for the assistant: built when the runtime starts, not at import; the tries and log lines; a client whose auth
check failed is still returned (it traces once Langfuse answers)."""

import logging

from rag_core.adapters import langfuse as lf


class FakeLangfuse:
    authenticated = False

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def auth_check(self) -> bool:
        return FakeLangfuse.authenticated


def keys(tmp_path):
    (tmp_path / "langfuse_public_key").write_text("pk\n", encoding="utf-8")
    (tmp_path / "langfuse_secret_key").write_text("sk\n", encoding="utf-8")
    return tmp_path


def test_missing_key_files_mean_no_client(tmp_path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    assert lf.init_langfuse("http://langfuse:3000", secrets_dir=tmp_path) == (None, False)
    assert "Langfuse setup failed" in caplog.text


def test_an_authenticated_client(tmp_path, monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(lf, "Langfuse", FakeLangfuse)
    FakeLangfuse.authenticated = True
    client, ok = lf.init_langfuse("http://langfuse:3000", secrets_dir=keys(tmp_path))
    assert ok and client.kwargs == {  # pyright: ignore[reportOptionalMemberAccess]
        "public_key": "pk",
        "secret_key": "sk",
        "host": "http://langfuse:3000",
        "timeout": 5,
    }
    assert "Langfuse client is authenticated and ready!" in caplog.text


def test_five_tries_then_the_client_without_tracing(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(lf, "Langfuse", FakeLangfuse)
    monkeypatch.setattr(lf.time, "sleep", lambda s: None)  # pyright: ignore[reportPrivateImportUsage]
    FakeLangfuse.authenticated = False
    client, ok = lf.init_langfuse("http://langfuse:3000", secrets_dir=keys(tmp_path))
    assert client is not None and not ok
    assert caplog.text.count("Langfuse auth check failed") == 5
    assert "Langfuse authentication failed after all retries. Running without Langfuse." in caplog.text
