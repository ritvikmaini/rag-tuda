"""The evals job's settings: technical defaults, the backend token as a secret file."""

import pytest

from evals.settings import load_settings
from rag_core.settings import SettingsError


def test_the_defaults_reach_the_assistant_inside_the_compose_network(tmp_path):
    s = load_settings(environ={}, secrets_dir=str(tmp_path)).evals
    assert (s.assistant_url, s.timeout_s, s.min_interval_s, s.runs_dir, s.out_dir, s.profile_dir) == (
        "http://backend:8000",
        240.0,
        4.0,
        "/app/data/eval_runs",
        "/app/out",
        "/app/profile",
    )
    assert s.token is None


def test_the_token_comes_from_the_backend_token_secret_file(tmp_path):
    (tmp_path / "backend_token").write_text("tok\n", encoding="utf-8")
    token = load_settings(environ={}, secrets_dir=str(tmp_path)).evals.token
    assert token is not None and token.get_secret_value() == "tok"


def test_the_environment_is_the_fallback_for_the_token_and_the_mac(tmp_path):
    s = load_settings(
        environ={"RAG__EVALS__TOKEN": "t", "RAG__EVALS__OUT_DIR": "data/eval_runs"}, secrets_dir=str(tmp_path)
    ).evals
    assert s.token is not None and s.token.get_secret_value() == "t" and s.out_dir == "data/eval_runs"


@pytest.mark.parametrize(
    "name,value",
    [
        ("TIMEOUT_S", "0"),
        ("TIMEOUT_S", "-1"),
        ("TIMEOUT_S", "nan"),
        ("TIMEOUT_S", "inf"),
        ("ASSISTANT_URL", "backend:8000"),
        ("ASSISTANT_URL", "ftp://backend"),
    ],
)
def test_an_unusable_timeout_or_assistant_url_is_refused(tmp_path, name, value):
    with pytest.raises(SettingsError, match="evals.%s: " % name.lower()):
        load_settings(environ={"RAG__EVALS__" + name: value}, secrets_dir=str(tmp_path))


def test_an_https_assistant_url_is_fine(tmp_path):
    assert (
        load_settings(
            environ={"RAG__EVALS__ASSISTANT_URL": "https://x.example"}, secrets_dir=str(tmp_path)
        ).evals.assistant_url
        == "https://x.example"
    )


@pytest.mark.parametrize(
    "environ,file", [({"RAG__EVALS__TOKEN": ""}, None), ({"RAG__EVALS__TOKEN": " \t"}, None), ({}, ""), ({}, "\n")]
)
def test_a_blank_token_is_no_token(tmp_path, environ, file):
    if file is not None:
        (tmp_path / "backend_token").write_text(file, encoding="utf-8")
    assert load_settings(environ=environ, secrets_dir=str(tmp_path)).evals.token is None
