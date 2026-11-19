"""The folders `rag add` registers, in `.rag/sources.json`: `{"version": 1, "sources": [{"name", "target"}]}`.

Registration is pure file work; the indexer reads the file on the next build and indexes each folder with its folder
connector (the paths setting `user_sources` names the file)."""

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")


@dataclass(frozen=True)
class Source:
    name: str
    target: str  # the folder's absolute path


def _file(root: str | Path) -> Path:
    return Path(root) / ".rag" / "sources.json"


def load(root: str | Path) -> list[Source]:
    path = _file(root)
    if not path.is_file():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))["sources"]
        if not isinstance(entries, list):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        raise ValueError(f"{path}: not a sources file (fix it or delete it, then add the folders again)") from None
    for i, entry in enumerate(entries):
        for key in ("name", "target"):
            if not isinstance(entry, dict) or not isinstance(entry.get(key), str) or not entry[key]:
                raise ValueError(f'{path}: sources[{i}] needs a "{key}" (a string), got {entry!r}')
    return [Source(s["name"], s["target"]) for s in entries]


def _save(root: str | Path, items: list[Source]) -> None:
    path = _file(root)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    text = json.dumps({"version": 1, "sources": [asdict(s) for s in items]}, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def register(root: str | Path, target: str | Path, name: str | None = None) -> Source:
    """Add the folder, or update it when the same folder is registered already.

    A default name that is taken by another folder gets `-2`, `-3`, ...; an explicit name taken by another folder is an
    error."""
    text = str(target).strip()
    if not text:
        raise ValueError("no folder given")
    path = Path(text).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"not a folder: {text}")
    if name is not None and not NAME.fullmatch(name):
        raise ValueError(f"a source name is 1-64 letters, digits, '.', '_' or '-': {name!r}")
    items = load(root)
    taken = {s.name: s for s in items}
    existing = next((s for s in items if s.target == str(path)), None)
    if name is not None:
        if name in taken and taken[name] is not existing:
            raise ValueError(f"the name {name!r} is used by {taken[name].target}")
        chosen = name
    elif existing is not None:
        chosen = existing.name
    else:
        base = re.sub(r"[^A-Za-z0-9._-]", "_", path.name)[:60] or "source"
        chosen, n = base, 2
        while chosen in taken:
            chosen, n = f"{base}-{n}", n + 1
    source = Source(chosen, str(path))
    if existing is None:
        items.append(source)
    else:
        items[items.index(existing)] = source
    _save(root, items)
    return source
