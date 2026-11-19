"""User sources in `.rag/sources.json`: which connector reads what, plus the `--only` setting.

Registration is pure file work; the indexer reads the file on the next reindex."""

import json
import os
import re
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")


@dataclass(frozen=True)
class Source:
    name: str
    connector: str  # folder | web | records
    target: str
    allowed_hosts: tuple[str, ...] = ()


@dataclass
class Registry:
    sources: list[Source] = field(default_factory=list)
    only: bool = False


def _file(root: str | Path) -> Path:
    return Path(root) / ".rag" / "sources.json"


def load(root: str | Path) -> Registry:
    path = _file(root)
    if not path.is_file():
        return Registry()
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [Source(s["name"], s["connector"], s["target"], tuple(s.get("allowed_hosts", ()))) for s in data["sources"]]
    return Registry(items, bool(data.get("only")))


def _save(root: str | Path, reg: Registry) -> None:
    path = _file(root)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    sources = []
    for s in reg.sources:
        d = asdict(s)
        d["allowed_hosts"] = list(s.allowed_hosts)
        if not s.allowed_hosts:
            del d["allowed_hosts"]
        sources.append(d)
    text = json.dumps({"version": 1, "only": reg.only, "sources": sources}, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _classify(target: str | Path) -> tuple[str, str, str, tuple[str, ...]]:
    """(connector, normalised target, default name, allowed hosts); ValueError when the target is not usable."""
    text = str(target).strip()
    if not text:
        raise ValueError("no folder, URL or CSV file given")
    if "://" in text:
        parts = urllib.parse.urlsplit(text)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            raise ValueError(f"only http(s) URLs with a host are supported: {text}")
        if parts.username or parts.password:
            raise ValueError("a URL with a user name or password is not stored")
        host = parts.hostname.lower()
        netloc = host + (f":{parts.port}" if parts.port else "")
        url = urllib.parse.urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))
        return "web", url, host, (host,)
    path = Path(text).expanduser().resolve()
    if path.is_dir():
        return "folder", str(path), path.name, ()
    if path.is_file() and path.suffix.lower() == ".csv":
        return "records", str(path), path.stem, ()
    raise ValueError(f"not a folder, an http(s) URL or an existing .csv file: {text}")


def register(root: str | Path, target: str | Path, name: str | None = None, only: bool = False) -> Source:
    """Add the source, or update it when the same target is registered already.

    A default name that is taken by another target gets `-2`, `-3`, ...; an explicit name taken by another target is an error.
    `only` is sticky: a later add without it leaves it as it was."""
    connector, norm, default, hosts = _classify(target)
    if name is not None and not NAME.fullmatch(name):
        raise ValueError(f"a source name is 1-64 letters, digits, '.', '_' or '-': {name!r}")
    reg = load(root)
    taken = {s.name: s for s in reg.sources}
    existing = next((s for s in reg.sources if s.target == norm), None)
    if name is not None:
        if name in taken and taken[name] is not existing:
            raise ValueError(f"the name {name!r} is used by {taken[name].target}")
        chosen = name
    elif existing is not None:
        chosen = existing.name
    else:
        base = re.sub(r"[^A-Za-z0-9._-]", "_", default)[:60] or "source"
        chosen, n = base, 2
        while chosen in taken:
            chosen, n = f"{base}-{n}", n + 1
    source = Source(chosen, connector, norm, hosts)
    if existing is None:
        reg.sources.append(source)
    else:
        reg.sources[reg.sources.index(existing)] = source
    reg.only = reg.only or only
    _save(root, reg)
    return source
