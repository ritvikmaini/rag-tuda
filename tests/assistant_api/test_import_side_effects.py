"""Every module of the assistant and of rag_core imports with no /app, no secrets, no profile, no locale change and no network;
the image's factory (its CMD) builds the app object only."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = """
import importlib, locale, pkgutil, socket, sys
def refuse(*args, **kwargs):
    raise SystemExit("network at import")
socket.socket.connect = refuse
before = locale.setlocale(locale.LC_TIME)
import assistant_api, rag_core
for package in (assistant_api, rag_core):
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module.name)
from gunicorn.util import import_app
assert type(import_app("assistant_api.app:create_app()")).__name__ == "FastAPI"
assert locale.setlocale(locale.LC_TIME) == before, "a module set the locale"
print("ok")
"""


def test_importing_the_app_does_nothing() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LANGFUSE_"))}
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(ROOT / "apps" / "assistant_api")],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-2000:]
