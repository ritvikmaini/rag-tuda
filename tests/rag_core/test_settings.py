"""rag_core.settings: the overlay file, the environment and secret files in that order, every rejection naming its key and
source, secrets never shown, and the RAG__ key parser (a Hypothesis property)."""

from pathlib import Path

import pytest
from hypothesis import assume, given, strategies as st
from pydantic import SecretStr, ValidationError

from rag_core.settings import OverlaySection, SettingsError, env_key, load, secret, secrets_path


class Http(OverlaySection):
    url: str = "http://default"
    timeout_s: float = 10.0
    token: SecretStr | None = secret("http_token")


class Model(OverlaySection):
    http: Http = Http()


def overlay(tmp_path, body):
    path = tmp_path / "overlay.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_defaults_without_any_source(tmp_path):
    settings = load(Model, environ={}, secrets_dir=str(tmp_path))
    assert (settings.http.url, settings.http.timeout_s, settings.http.token) == ("http://default", 10.0, None)


def test_the_overlay_overrides_the_default(tmp_path):
    settings = load(
        Model, overlay=overlay(tmp_path, "http:\n  url: http://overlay\n"), environ={}, secrets_dir=str(tmp_path)
    )
    assert settings.http.url == "http://overlay"


def test_the_environment_overrides_the_overlay(tmp_path):
    settings = load(
        Model,
        overlay=overlay(tmp_path, "http:\n  url: http://overlay\n"),
        environ={"RAG__HTTP__URL": "http://env"},
        secrets_dir=str(tmp_path),
    )
    assert settings.http.url == "http://env"


def test_a_secret_file_overrides_the_environment(tmp_path):
    (tmp_path / "http_token").write_text("from-file\n", encoding="utf-8")
    settings = load(Model, environ={"RAG__HTTP__TOKEN": "from-env"}, secrets_dir=str(tmp_path))
    assert settings.http.token is not None and settings.http.token.get_secret_value() == "from-file"


def test_the_environment_alone_can_carry_a_secret(tmp_path):
    settings = load(Model, environ={"RAG__HTTP__TOKEN": "from-env"}, secrets_dir=str(tmp_path))
    assert settings.http.token is not None and settings.http.token.get_secret_value() == "from-env"


def test_environment_values_are_coerced_to_the_field_type(tmp_path):
    assert load(Model, environ={"RAG__HTTP__TIMEOUT_S": "2.5"}, secrets_dir=str(tmp_path)).http.timeout_s == 2.5


def test_variables_without_the_prefix_are_ignored(tmp_path):
    assert (
        load(
            Model, environ={"PATH": "/bin", "RAG_HTTP": "x", "rag__http__url": "x"}, secrets_dir=str(tmp_path)
        ).http.url
        == "http://default"
    )


@pytest.mark.parametrize(
    "environ,message",
    [
        ({"RAG__HTTP__TYPO": "x"}, "http.typo: unknown key (RAG__HTTP__TYPO (environment))"),
        ({"RAG__NOPE__URL": "x"}, "nope: unknown section (RAG__NOPE__URL (environment))"),
        ({"RAG__HTTP": "x"}, "RAG__HTTP: expected RAG__<SECTION>__<KEY> (environment)"),
        ({"RAG____URL": "x"}, "RAG____URL: expected RAG__<SECTION>__<KEY> (environment)"),
    ],
)
def test_the_environment_is_refused_with_the_key_and_its_source(tmp_path, environ, message):
    with pytest.raises(SettingsError) as error:
        load(Model, environ=environ, secrets_dir=str(tmp_path))
    assert str(error.value) == message


@pytest.mark.parametrize(
    "body,message",
    [
        ("http:\n  typo: 1\n", "http.typo: unknown key (overlay "),
        ("other:\n  url: x\n", "other: unknown section (overlay "),
        ("- a\n- b\n", "expected a mapping of sections"),
        ("http: 3\n", "http: expected a mapping of keys"),
    ],
)
def test_the_overlay_is_refused_with_the_key_and_the_file(tmp_path, body, message):
    with pytest.raises(SettingsError, match=message.replace("(", r"\(")):
        load(Model, overlay=overlay(tmp_path, body), environ={}, secrets_dir=str(tmp_path))


def test_a_missing_overlay_file_is_refused(tmp_path):
    with pytest.raises(SettingsError, match="cannot be read"):
        load(Model, overlay=str(tmp_path / "missing.yaml"), environ={}, secrets_dir=str(tmp_path))


def test_an_invalid_value_names_its_source_but_never_a_value(tmp_path):
    (tmp_path / "http_token").write_text("s3cr3t", encoding="utf-8")
    with pytest.raises(SettingsError) as error:
        load(Model, environ={"RAG__HTTP__TIMEOUT_S": "soon"}, secrets_dir=str(tmp_path))
    assert "http.timeout_s:" in str(error.value) and "(RAG__HTTP__TIMEOUT_S (environment))" in str(error.value)
    assert "soon" not in str(error.value) and "s3cr3t" not in str(error.value)


def test_a_secret_never_appears_in_a_repr(tmp_path):
    (tmp_path / "http_token").write_text("s3cr3t", encoding="utf-8")
    settings = load(Model, environ={}, secrets_dir=str(tmp_path))
    assert "s3cr3t" not in repr(settings) and "s3cr3t" not in str(settings)


def test_sections_are_frozen(tmp_path):
    settings = load(Model, environ={}, secrets_dir=str(tmp_path))
    with pytest.raises(ValidationError):
        settings.http.url = "x"  # type: ignore[misc]


def test_a_model_field_that_is_not_a_section_is_refused(tmp_path):
    class Flat(OverlaySection):
        url: str = "x"

    with pytest.raises(SettingsError, match="sections only"):
        load(Flat, environ={}, secrets_dir=str(tmp_path))


def test_a_blank_secret_file_does_not_hide_the_environment(tmp_path):
    (tmp_path / "http_token").write_text("\n", encoding="utf-8")
    settings = load(Model, environ={"RAG__HTTP__TOKEN": "from-env"}, secrets_dir=str(tmp_path))
    assert settings.http.token is not None and settings.http.token.get_secret_value() == "from-env"


def test_a_key_set_twice_in_different_case_is_refused(tmp_path):
    with pytest.raises(SettingsError) as error:
        load(Model, environ={"RAG__HTTP__URL": "http://a", "RAG__http__url": "http://b"}, secrets_dir=str(tmp_path))
    assert str(error.value) == "http.url: set twice (RAG__HTTP__URL (environment) and RAG__http__url (environment))"


NAME = st.from_regex(r"[a-z][a-z0-9]{0,7}(_[a-z0-9]{1,5}){0,2}", fullmatch=True)


@given(section=NAME, key=NAME)
def test_every_section_and_key_round_trips_through_the_variable_name(section, key):
    assert env_key("RAG__%s__%s" % (section.upper(), key.upper())) == (section, key)


@given(st.text())
def test_a_name_without_the_prefix_is_no_setting(name):
    assume(not name.startswith("RAG__"))
    assert env_key(name) is None


def test_the_secrets_folder_is_the_argument_else_the_environment_else_run_secrets(tmp_path):
    here, there = tmp_path / "here", tmp_path / "there"
    assert secrets_path({"RAG_SECRETS_DIR": str(there)}, here) == here
    assert secrets_path({"RAG_SECRETS_DIR": str(there)}) == there
    assert secrets_path({}) == Path("/run/secrets")


def test_load_reads_the_secret_files_of_the_environments_folder(tmp_path):
    (tmp_path / "http_token").write_text("tok\n", encoding="utf-8")
    token = load(Model, environ={"RAG_SECRETS_DIR": str(tmp_path)}).http.token
    assert token is not None and token.get_secret_value() == "tok"
