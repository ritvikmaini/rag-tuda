"""gunicorn_conf.py defaults to one worker and refuses to start with more, because the GPU pipeline slot is an in-process
semaphore. The real file runs in a subprocess with a stub uvicorn.logging (the test venv has no uvicorn; gunicorn_conf only
names the formatter class in its logging configuration)."""

import json
import os
import subprocess
import sys

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "apps", "assistant_api"))


def run(
    tmp_path,
    workers=None,
    code="import gunicorn_conf; print('workers', gunicorn_conf.workers, 'max', gunicorn_conf.MAX_WORKERS)",
):
    stub = tmp_path / "stub" / "uvicorn"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("", encoding="utf-8")
    (stub / "logging.py").write_text(
        "import logging\n\n\nclass DefaultFormatter(logging.Formatter):\n"
        "    def __init__(self, fmt=None, datefmt=None, **kwargs):\n"
        "        super().__init__(fmt, datefmt)\n",
        encoding="utf-8",
    )
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(stub.parent)}
    if workers is not None:
        env["UVICORN_WORKERS"] = workers
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_default_is_one_worker(tmp_path):
    result = run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "workers 1 max 1"


def test_one_worker_starts(tmp_path):
    assert run(tmp_path, "1").returncode == 0


def test_more_workers_than_the_slot_refuse_to_start(tmp_path):
    result = run(tmp_path, "2")
    assert result.returncode == 1
    assert "UVICORN_WORKERS=2" in result.stderr and "one GPU, one pipeline slot" in result.stderr
    assert "workers" not in result.stdout


def test_the_config_stamps_the_ref_and_drops_the_access_log(tmp_path):
    """The one handler every logger reaches carries RefFilter and a format with ref=; gunicorn.access (which
    UvicornWorker copies onto uvicorn.access) has no handler and level WARNING, so the access line, which holds the question, goes."""
    code = (
        "import gunicorn_conf, logging, json\n"
        "h = logging.getLogger().handlers[0]\n"
        "a = logging.getLogger('gunicorn.access')\n"
        "print(json.dumps([[type(f).__name__ for f in h.filters], h.formatter._fmt, [x.name for x in a.handlers], a.level, type(h.formatter).__name__]))"
    )
    result = run(tmp_path, code=code)
    assert result.returncode == 0, result.stderr
    filters, fmt, handlers, level, formatter = json.loads(result.stdout.strip().splitlines()[-1])
    assert (filters, handlers, level, formatter) == (["RefFilter"], [], 30, "ClassOnlyFormatter")
    assert "ref=%(ref)s" in fmt
