"""Runtime hardening: loopback ports, no secret with a default, no env_file, no token in the backend image and
constant-time token checks."""

import ast
import os
import re

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMPOSE = ("deploy/compose.yaml",)


def text(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def load(name):
    return yaml.safe_load(text(name))


def published_ports():
    for name in COMPOSE:
        for service, spec in (load(name).get("services") or {}).items():
            for entry in (spec or {}).get("ports") or []:
                yield name, service, entry


def test_every_published_port_is_bound_to_loopback():
    ports = list(published_ports())
    open_to_all = [
        (n, s, e)
        for n, s, e in ports
        if not (e.get("host_ip") == "127.0.0.1" if isinstance(e, dict) else str(e).startswith("127.0.0.1:"))
    ]
    assert open_to_all == []


SECRET_NAME = re.compile(r"(TOKEN|PASSWORD|SECRET|SALT|ENCRYPTION_KEY|AUTH|_KEY)$")
OPTIONAL_SECRETS = {"HUGGING_FACE_HUB_TOKEN"}  # vLLM serves a public model: the token is optional, empty is allowed
REFERENCE = re.compile(r"\$\{([A-Z0-9_]+)(:\?|:-|-|\?|\})")


def test_no_secret_has_a_default_in_compose():
    found = [
        (name, var, op)
        for name in COMPOSE
        for var, op in REFERENCE.findall(text(name))
        if SECRET_NAME.search(var) and var not in OPTIONAL_SECRETS and op != ":?"
    ]
    assert found == []


def test_backend_and_indexer_read_no_env_file():
    found = [
        (name, service)
        for name in COMPOSE
        for service in ("backend", "indexer")
        if "env_file" in ((load(name).get("services") or {}).get(service) or {})
    ]
    assert found == []


def test_the_backend_image_carries_no_token():
    assert not re.search(r"^\s*(ENV|ARG)\s+\S*TOKEN", text("apps/assistant_api/Dockerfile"), re.M)


APPS = ("apps/assistant_api/src/assistant_api/api/deps.py",)  # where each app's verify_token lives


def parse(path):
    return ast.parse(text(path))


def test_the_app_checks_its_token_with_token_matches():
    for path in APPS:
        verify = next(
            n for n in ast.walk(parse(path)) if isinstance(n, ast.AsyncFunctionDef) and n.name == "verify_token"
        )
        called = {c.func.id for c in ast.walk(verify) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        compared = [c for c in ast.walk(verify) if isinstance(c, ast.Compare)]
        assert "token_matches" in called and compared == [], path
