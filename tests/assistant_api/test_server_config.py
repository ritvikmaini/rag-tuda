"""The backend's real server stack (gunicorn_conf.py with the uvicorn worker), serving a one-route app that answers the
rate-limit key of the request: a forwarded-for header from the caller does not change the key, and the server writes no
access-log line (the question is a GET parameter)."""

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

CONF = Path(__file__).resolve().parents[2] / "apps" / "assistant_api" / "gunicorn_conf.py"
APP = """
from fastapi import FastAPI, Request
from assistant_api.api.deps import _rate_limit_key

app = FastAPI()


@app.get("/who")
def who(request: Request) -> dict[str, str]:
    return {"key": _rate_limit_key(request)}
"""


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, subprocess.Popen[str]]]:
    folder = tmp_path_factory.mktemp("who")
    (folder / "who_app.py").write_text(APP, encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "GUNICORN_BIND": f"127.0.0.1:{port}", "PYTHONPATH": str(folder)}
    env.pop("FORWARDED_ALLOW_IPS", None)
    process = subprocess.Popen(
        [sys.executable, "-m", "gunicorn", "who_app:app", "-c", str(CONF)],
        cwd=folder,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                httpx.get(url + "/who", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            raise AssertionError("the server did not start")
        yield url, process
    finally:
        process.terminate()


def test_forwarded_for_does_not_change_the_rate_limit_key(server: tuple[str, subprocess.Popen[str]]) -> None:
    url, _ = server
    plain = httpx.get(url + "/who").json()
    spoofed = httpx.get(url + "/who", headers={"X-Forwarded-For": "9.9.9.9", "X-Real-IP": "9.9.9.9"}).json()
    assert plain == spoofed == {"key": "127.0.0.1"}


def test_the_server_writes_no_access_log_line(server: tuple[str, subprocess.Popen[str]]) -> None:
    url, process = server
    httpx.get(url + "/who", params={"question": "kanarienvogel"})
    process.terminate()
    _, log = process.communicate(timeout=30)
    assert "kanarienvogel" not in log and "GET /who" not in log
