"""Hardware and prerequisite detection for `rag doctor`. Read-only: nothing is written."""

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import urllib.request
from dataclasses import asdict, dataclass

from rag_tuda import classes
from rag_tuda.classes import SizeClass

OLLAMA_URL = "http://127.0.0.1:11434/api/version"
MIN_RAM_GB = 16
RAM_TOLERANCE_GB = 1  # firmware and iGPUs reserve memory, so "16 GB" reads as ~15.5
GIB = 2**30


@dataclass
class Docker:
    installed: bool = False
    ok: bool = False
    server_version: str = ""
    runtime_hint: str = ""


@dataclass
class Ollama:
    installed: bool = False
    version: str = ""
    reachable: bool = False


@dataclass
class Facts:
    os: str
    arch: str
    is_wsl: bool
    cpu_name: str
    cores: int
    ram_gb: float
    apple_chip: str
    unified_gb: float
    nvidia: list[tuple[str, float]]
    docker: Docker
    ollama: Ollama
    disk_free_gb: float
    brew: bool = False


# Parsers: captured text in, values out. None or garbage never raises.


def parse_sysctl(text: str | None) -> tuple[float, str, int] | None:
    """Output of `sysctl -n hw.memsize machdep.cpu.brand_string hw.ncpu`: (GiB, brand, cpus)."""
    lines = (text or "").strip().splitlines()
    if len(lines) < 3 or not lines[0].strip().isdigit() or not lines[2].strip().isdigit():
        return None
    return int(lines[0]) / GIB, lines[1].strip(), int(lines[2])


def parse_nvidia(text: str | None) -> list[tuple[str, float]]:
    """Lines `name, MiB` to (name, GB); MiB / 1000 so a 12 GB card counts as 12 GB."""
    gpus = []
    for line in (text or "").splitlines():
        name, _, mib = line.rpartition(",")
        if name and mib.strip().isdigit():
            gpus.append((name.strip(), int(mib) / 1000))
    return gpus


def parse_docker(text: str | None) -> Docker:
    """Output of `docker info --format '{{json .}}'`; a daemon that is down still prints JSON."""
    try:
        info = json.loads(text or "")
    except ValueError:
        return Docker()
    if not isinstance(info, dict):
        return Docker()
    version = str(info.get("ServerVersion") or "")
    if not version:
        return Docker(installed=True)
    system = f"{info.get('OperatingSystem', '')} {info.get('Name', '')}".lower()
    hint = (
        "Docker Desktop"
        if "docker desktop" in system
        else "OrbStack"
        if "orbstack" in system
        else "Colima"
        if "colima" in system
        else "Docker Engine"
    )
    return Docker(True, True, version, hint)


def parse_ollama_cli(text: str | None) -> str | None:
    m = re.search(r"version is (\S+)", text or "")
    return m.group(1) if m else None


def parse_ollama_api(text: str | None) -> str | None:
    try:
        return str(json.loads(text or "")["version"])
    except (ValueError, KeyError, TypeError):
        return None


def parse_meminfo(text: str | None) -> float | None:
    m = re.search(r"^MemTotal:\s+(\d+) kB", text or "", re.M)
    return int(m.group(1)) / 1024 / 1024 if m else None


def is_wsl_text(text: str | None) -> bool:
    return "microsoft" in (text or "").lower()


def parse_cpuinfo(text: str | None) -> str:
    m = re.search(r"^model name\s*:\s*(.+)$", text or "", re.M)
    return m.group(1).strip() if m else ""


def assemble(
    *,
    system: str,
    machine: str,
    cores: int,
    disk_free_gb: float,
    brew: bool = False,
    sysctl: str | None = None,
    nvidia: str | None = None,
    docker: str | None = None,
    ollama_cli: str | None = None,
    ollama_api: str | None = None,
    meminfo: str | None = None,
    proc_version: str | None = None,
    cpuinfo: str | None = None,
    win_mem_bytes: int | None = None,
    cpu_name: str = "",
) -> Facts:
    """Builds Facts from captured probe texts; the pure half of `gather`."""
    arm_mac = system == "darwin" and machine == "arm64"
    ram = 0.0
    chip = ""
    if system == "darwin" and (sc := parse_sysctl(sysctl)):
        ram, cpu_name, cores = sc
        chip = cpu_name if arm_mac and cpu_name.startswith("Apple") else ""
    elif system == "linux":
        ram = parse_meminfo(meminfo) or 0.0
        cpu_name = parse_cpuinfo(cpuinfo) or cpu_name
    elif win_mem_bytes:
        ram = win_mem_bytes / GIB
    cli_version, api_version = parse_ollama_cli(ollama_cli), parse_ollama_api(ollama_api)
    return Facts(
        os=system,
        arch=machine,
        is_wsl=system == "linux" and is_wsl_text(proc_version),
        cpu_name=cpu_name,
        cores=cores,
        ram_gb=ram,
        apple_chip=chip,
        unified_gb=ram if arm_mac else 0.0,
        nvidia=parse_nvidia(nvidia),
        docker=parse_docker(docker),
        ollama=Ollama(bool(cli_version or api_version), cli_version or api_version or "", api_version is not None),
        disk_free_gb=disk_free_gb,
        brew=brew,
    )


# Probes: the only code that touches the machine.


def _run(cmd: list[str], *, merge_stderr: bool = False, need_ok: bool = True) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if need_ok and p.returncode != 0:
        return None
    return p.stdout + (p.stderr if merge_stderr else "")


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _ollama_api() -> str | None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(OLLAMA_URL, timeout=1) as r:
            return r.read(4096).decode("utf-8", "replace")
    except (OSError, ValueError):
        return None


def _windows_mem_bytes() -> int | None:
    class Status(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    s = Status()
    s.dwLength = ctypes.sizeof(Status)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    return s.ullTotalPhys if ok else None


def gather() -> Facts:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine == "aarch64":
        machine = "arm64"
    docker = _run(["docker", "info", "--format", "{{json .}}"], need_ok=False)
    if docker is None and shutil.which("docker"):
        docker = "{}"
    return assemble(
        system=system,
        machine=machine,
        cores=os.cpu_count() or 0,
        disk_free_gb=shutil.disk_usage(os.getcwd()).free / 1e9,
        brew=shutil.which("brew") is not None,
        sysctl=_run(["sysctl", "-n", "hw.memsize", "machdep.cpu.brand_string", "hw.ncpu"])
        if system == "darwin"
        else None,
        nvidia=_run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]),
        docker=docker,
        ollama_cli=_run(["ollama", "--version"], merge_stderr=True),
        ollama_api=_ollama_api(),
        meminfo=_read("/proc/meminfo") if system == "linux" else None,
        proc_version=_read("/proc/version") if system == "linux" else None,
        cpuinfo=_read("/proc/cpuinfo") if system == "linux" else None,
        win_mem_bytes=_windows_mem_bytes() if system == "windows" else None,
        cpu_name=platform.processor(),
    )


# Decisions and output.


def is_mac_native(f: Facts) -> bool:
    return f.os == "darwin" and f.arch == "arm64"


def runtime_mode(f: Facts) -> str:
    if is_mac_native(f):
        return "Mac native (Ollama on Metal, model servers on MPS, Docker for Postgres)"
    mode = "compose (docker compose -p rag-tuda)"
    if f.os == "windows":
        mode += "; native Windows is best effort, WSL2 is recommended"
    return mode


def problems(f: Facts) -> list[str]:
    """Missing prerequisites for `rag up`, each with how to get it."""
    out = []
    if not f.docker.installed:
        out.append(
            "Docker not found. Install Docker Engine (Linux, WSL2), or Colima, OrbStack or Docker Desktop (macOS, Windows)."
        )
    elif not f.docker.ok:
        out.append(
            "Docker is installed but not running. Start Docker Desktop, OrbStack or Colima (`colima start`), or the docker service."
        )
    if is_mac_native(f) and not f.ollama.installed:
        out.append(
            "Ollama is not installed: run `brew install ollama`."
            if f.brew
            else "Ollama not found and Homebrew is missing. Install Homebrew (https://brew.sh), then `brew install ollama`."
        )
    return out


def warnings(f: Facts, cls: SizeClass) -> list[str]:
    out = []
    if f.ram_gb and f.ram_gb < MIN_RAM_GB - RAM_TOLERANCE_GB:
        out.append(
            f"Only {f.ram_gb:.1f} GB RAM; 16 GB or more is recommended. Staying on class S, expect it to be slow."
        )
    total = classes.download_total_gb(cls)
    if f.disk_free_gb < total:
        out.append(f"Only {f.disk_free_gb:.1f} GB free disk; the downloads need about {total:.1f} GB.")
    return out


def _hardware_line(f: Facts) -> str:
    if f.nvidia:
        names = ", ".join(f"{n} ({v:.1f} GB)" for n, v in f.nvidia)
        return f"GPU: {len(f.nvidia)} x NVIDIA, {sum(v for _, v in f.nvidia):.1f} GB VRAM: {names}"
    if is_mac_native(f):
        return f"Memory: {f.unified_gb:.1f} GB unified ({f.apple_chip or f.cpu_name}), about 65% usable = {classes.usable_gb(f):.1f} GB"
    return f"GPU: none usable for models (CPU, AMD or Intel graphics); RAM {f.ram_gb:.1f} GB"


def report(f: Facts, cls: SizeClass) -> str:
    where = "WSL2" if f.is_wsl else {"darwin": "macOS", "linux": "Linux", "windows": "Windows"}.get(f.os, f.os)
    lines = [
        f"Platform: {where} {f.arch}, {f.cpu_name or 'unknown CPU'}, {f.cores} cores, {f.ram_gb:.1f} GB RAM, {f.disk_free_gb:.0f} GB free disk",
        _hardware_line(f),
        f"Class: {cls.name} ({cls.rule}); usable memory {classes.usable_gb(f):.1f} GB",
        f"Default card: {cls.default_card or 'none (pick one with --card)'}",
        "Models:",
        f"  answer (Ollama): {cls.ollama_tag + f' ({cls.download_gb:g} GB, {cls.context_k}k context)' if cls.ollama_tag else 'none defined for this class'}",
        f"  answer (vLLM, NVIDIA, opt-in): {cls.vllm_model}",
    ]
    lines += [f"  {m.name} ({m.gb:g} GB)" for m in classes.FIXED_MODELS]
    lines.append(f"Download total: {classes.download_total_gb(cls):.1f} GB")
    lines.append(f"Suits: {cls.use_case}")
    lines.append(f"Mode: {runtime_mode(f)}")
    lines.append(
        f"Docker: {f.docker.runtime_hint} {f.docker.server_version}" if f.docker.ok else "Docker: not available"
    )
    lines.append(
        f"Ollama: {f.ollama.version} ({'running' if f.ollama.reachable else 'not running'})"
        if f.ollama.installed
        else "Ollama: not installed"
    )
    lines += [f"Warning: {w}" for w in warnings(f, cls)]
    missing = problems(f)
    lines += [f"Missing: {m}" for m in missing]
    lines.append(
        "Next: uv run rag up" if not missing else "Next: fix the missing items, then run `uv run rag doctor` again"
    )
    return "\n".join(lines)


def report_json(f: Facts, cls: SizeClass) -> str:
    missing = problems(f)
    return json.dumps(
        {
            "facts": asdict(f),
            "class": asdict(cls),
            "default_card": cls.default_card,
            "usable_gb": round(classes.usable_gb(f), 1),
            "fixed_models": [asdict(m) for m in classes.FIXED_MODELS],
            "download_total_gb": round(classes.download_total_gb(cls), 1),
            "mode": runtime_mode(f),
            "warnings": warnings(f, cls),
            "missing": missing,
            "ok": not missing,
            "next": "uv run rag up" if not missing else "rag doctor",
        },
        indent=2,
    )
