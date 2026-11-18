"""Bearer token checks: constant-time; a missing, empty, wrong or non-ASCII token never matches
and never raises (a TypeError would turn a stray header into a 500 instead of a 401)."""

import importlib.util
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COPIES = {"backend": "apps/assistant_api/src/assistant_api/api/token_auth.py"}


def load(app):
    spec = importlib.util.spec_from_file_location("token_auth_" + app, os.path.join(ROOT, COPIES[app]))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=sorted(COPIES))
def auth(request):
    return load(request.param)


def test_the_right_token_matches(auth):
    assert auth.token_matches("a" * 64, "a" * 64) is True


def test_a_wrong_token_does_not_match(auth):
    assert auth.token_matches("b" * 64, "a" * 64) is False


def test_no_expected_token_never_matches(auth):
    assert [auth.token_matches(given, expected) for given in ("", "x") for expected in (None, "")] == [False] * 4


def test_a_non_ascii_token_is_refused_not_an_error(auth):
    assert auth.token_matches("tökén", "token") is False
    assert auth.token_matches(None, "token") is False


def test_the_comparison_is_constant_time(auth, monkeypatch):
    calls, real = [], auth.hmac.compare_digest
    monkeypatch.setattr(auth.hmac, "compare_digest", lambda a, b: calls.append((type(a), type(b))) or real(a, b))
    assert auth.token_matches("abc", "abc") is True
    assert calls == [(bytes, bytes)]
