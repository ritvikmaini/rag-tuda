"""down: processes stop, stale pid files are cleared, and a second down changes nothing. Real child processes, no Docker."""

import os
import signal
import subprocess
import sys
import time

import pytest

from rag_tuda import state, stack

posix = pytest.mark.skipif(sys.platform == "win32", reason="process groups")


class NoDocker(stack.System):
    def __init__(self, ran):
        super().__init__({})
        self.ran = ran

    def run(self, argv, cwd, env, timeout_s=0, inherit=()):
        self.ran.append(argv)
        return 0


def spawn(current, name, argv=("sleep", "60")):
    """Starts a process through the executor, as `up` does."""
    step = stack.Step(
        "spawn",
        name=name,
        argv=argv,
        env={},
        path=f"{current.dir}/pids/{name}.pid",
        log=f"{current.dir}/logs/{name}.log",
    )
    system = stack.System({})
    stack.execute([step], system, lambda line: None)
    pid, started = stack.read_pid(step.path)
    return system, pid, started


def down(tmp_path, current, ran, *, docker=True, purge=False):
    system = NoDocker(ran)
    facts = stack.gather_down(tmp_path, current.dir, type("H", (), {"docker": type("D", (), {"ok": docker})()})())
    lines: list[str] = []
    stack.execute(stack.plan_down(current.dir, facts, purge), system, lines.append)
    return lines


@posix
def test_down_stops_a_process_and_clears_its_pid_file(tmp_path):
    current = state.ensure(tmp_path)
    _, pid, started = spawn(current, "embedding")
    assert started and started == stack.System({}).start_time(pid) and stack.System({}).alive(pid)
    ran: list = []
    lines = down(tmp_path, current, ran)
    assert not stack.System({}).alive(pid)
    assert not (current.dir / "pids" / "embedding.pid").exists()
    assert "embedding: stopping" in lines and "backend: not running" in lines
    assert ran == [("docker", "compose", "-p", "rag-tuda", "down", "--remove-orphans")]


@posix
def test_down_twice(tmp_path):
    current = state.ensure(tmp_path)
    _, pid, _ = spawn(current, "reranker")
    ran: list = []
    first = down(tmp_path, current, ran)
    second = down(tmp_path, current, ran)
    assert "reranker: stopping" in first and "reranker: not running" in second
    assert not stack.System({}).alive(pid) and list((current.dir / "pids").iterdir()) == []
    assert len(ran) == 2  # compose down is safe to repeat


@posix
def test_a_stale_pid_file_is_cleared(tmp_path):
    current = state.ensure(tmp_path)
    done = subprocess.Popen(["true"])
    done.wait()
    (current.dir / "pids" / "backend.pid").write_text(f"{done.pid}\nMon  1 Jan 00:00:00 2024\n")
    lines = down(tmp_path, current, [])
    assert "backend: stopping" in lines and not (current.dir / "pids" / "backend.pid").exists()


@posix
def test_a_reused_pid_is_not_signalled(tmp_path):
    """The pid file records another start time than the process now behind the pid has: it is not signalled, the file
    stays and the line says so."""
    current = state.ensure(tmp_path)
    other = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        (current.dir / "pids" / "embedding.pid").write_text(f"{other.pid}\nMon  1 Jan 00:00:00 2024\n")
        lines = down(tmp_path, current, [])
        assert other.poll() is None and (current.dir / "pids" / "embedding.pid").exists()
        assert any(f"pid {other.pid} is running" in line and "not stopped" in line for line in lines)
    finally:
        os.killpg(other.pid, signal.SIGKILL)
        other.wait()


@posix
def test_a_leader_with_uv_in_its_command_line_is_not_taken_for_ours(tmp_path):
    """The check is the recorded start time, not a word of the command line."""
    current = state.ensure(tmp_path)
    other = subprocess.Popen(["sh", "-c", "sleep 60; : uvicorn uvx"], start_new_session=True)
    try:
        (current.dir / "pids" / "backend.pid").write_text(f"{other.pid}\nMon  1 Jan 00:00:00 2024\n")
        down(tmp_path, current, [])
        assert other.poll() is None and (current.dir / "pids" / "backend.pid").exists()
    finally:
        os.killpg(other.pid, signal.SIGKILL)
        other.wait()


@posix
def test_the_whole_process_group_stops(tmp_path):
    """`uv run` leaves its child as a separate process: stopping the leader stops the group."""
    current = state.ensure(tmp_path)
    marker = tmp_path / "child.pid"
    script = f"sleep 60 & echo $! > {marker}; wait"
    _, pid, _ = spawn(current, "embedding", ("sh", "-c", script))
    for _ in range(50):
        if marker.exists() and marker.read_text().strip():
            break
        time.sleep(0.1)
    child = int(marker.read_text())
    stack.execute([stack.Step("stop", name="embedding", path=f"{current.dir}/pids/embedding.pid")], stack.System({}))
    time.sleep(0.2)
    assert not stack.System({}).alive(pid) and not stack.System({}).alive(child)


def test_down_without_docker_says_so_and_is_still_safe(tmp_path):
    current = state.ensure(tmp_path)
    ran: list = []
    assert "Docker is not running: no containers to stop" in down(tmp_path, current, ran, docker=False)
    assert ran == []


def test_purge_deletes_the_crawled_content_only(tmp_path):
    current = state.ensure(tmp_path)
    for name in ("crawl", "snapshots", "state"):
        (current.dir / name).mkdir()
        (current.dir / name / "x").write_text("x")
    down(tmp_path, current, [], purge=True)
    assert sorted(
        p.name for p in current.dir.iterdir() if p.is_dir() and p.name in ("crawl", "snapshots", "state")
    ) == ["state"]
    down(tmp_path, current, [], purge=True)  # nothing left to delete: still fine


@posix
def test_a_pid_whose_start_time_was_never_recorded_is_kept_and_named(tmp_path):
    current = state.ensure(tmp_path)
    other = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        (current.dir / "pids" / "backend.pid").write_text(f"{other.pid}\n\n")
        lines = down(tmp_path, current, [])
        assert other.poll() is None and (current.dir / "pids" / "backend.pid").exists()
        assert any(f"pid {other.pid}" in line for line in lines)
    finally:
        os.killpg(other.pid, signal.SIGKILL)
        other.wait()


def test_the_start_time_is_read_in_the_c_locale_and_utc(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"], seen["env"] = argv, kwargs["env"]
        return subprocess.CompletedProcess(argv, 0, stdout="Sat Nov 21 10:45:56 2026\n")

    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    system = stack.System({"PATH": "/bin", "LANG": "de_DE.UTF-8", "LC_ALL": "de_DE.UTF-8", "TZ": "Europe/Berlin"})
    assert system.start_time(7) == "Sat Nov 21 10:45:56 2026"
    assert seen["argv"] == ("ps", "-o", "lstart=", "-p", "7")
    assert seen["env"]["LC_ALL"] == "C" and seen["env"]["TZ"] == "UTC" and seen["env"]["PATH"] == "/bin"
