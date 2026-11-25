"""The list safeguard (full syncs only). A crawl can lose whole branches at random, so a registered link missing from
this run's list keeps its PDF and its extraction cache entry and is returned as retained, until it has been missing from
`grace_runs` different lists in a row. A list shorter than (1 - max_shrink) of the last full list counts no misses at all
until it has persisted `grace_runs` times. The list a run already counted (two builds on the same
crawl) counts nothing a second time. The state lives in the state store as LIST_STATE."""

import dataclasses
import hashlib
import json
import logging
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_core.domain.sources import CorruptStateError

from indexer.sync.registry import PdfEntry, file_hash, local_path

LIST_STATE = "pdf_list_state.json"


@dataclass(frozen=True)
class Safeguard:
    grace_runs: int = 3
    max_shrink: float = 0.2

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "Safeguard":
        """From a source's profile entry: `safeguard: {grace_runs, max_shrink}`."""
        section = settings.get("safeguard") or {}
        return cls(max(1, int(section.get("grace_runs", 3))), float(section.get("max_shrink", 0.2)))

    def apply(
        self,
        entries: list[PdfEntry],
        registry: dict[str, str],
        state_path: Path,
        cache_dir: Path,
        pdf_dir: str | Path,
        dry_run: bool = False,
        dropped: Collection[str] = (),
    ) -> tuple[list[PdfEntry], dict, int]:
        """Counts this list against the state at state_path. Links no longer listed are removed from `registry` (and
        their file deleted) after grace_runs misses. dry_run (frozen inputs): the retained entries are computed from the
        current state; nothing is counted, removed or saved. A registered link in `dropped` (left out on purpose, not lost) is
        removed with its file at once, without a miss, and does not count towards the list's shrinking. Returns (retained entries, list check report, number of
        removed links)."""
        grace = self.grace_runs
        state = _load_state(state_path)
        today = datetime.now().strftime("%Y-%m-%d")
        current = {entry.url for entry in entries}
        gone = (set(dropped) & registry.keys()) - current
        for entry in entries:
            state["known"][entry.url] = dataclasses.asdict(entry)
            state["missing"].pop(entry.url, None)

        fingerprint = hashlib.sha256(
            "\n".join(sorted(f"{entry.url} {entry.sha256 or ''}" for entry in entries)).encode("utf-8")
        ).hexdigest()
        repeat = fingerprint == state["last_list"]
        state["last_list"] = fingerprint
        last = state["last_full_count"]
        shrunk, accepted = bool(last) and len(entries) + len(gone) < (1 - self.max_shrink) * last, False
        if dry_run:
            logging.info("[PdfSync] frozen inputs: the list safeguard runs dry (nothing counted, removed or saved)")
        elif repeat:
            logging.warning("[PdfSync] the same list as the last run: nothing is counted or removed this run")
        elif shrunk:
            state["shrunk_runs"] += 1
            if state["shrunk_runs"] >= grace:
                shrunk, accepted = False, True
                logging.warning(
                    f"[PdfSync] the shorter list ({len(entries)}, last full list {last}) persisted {grace} runs "
                    "and is accepted"
                )
            else:
                logging.warning(
                    f"[PdfSync] list shrank to {len(entries)} entries (last full list {last}): "
                    f"nothing is counted or removed this run ({state['shrunk_runs']}/{grace})"
                )
        report = {
            "entries": len(entries),
            "last_full_count": last,
            "shrunk": shrunk,
            "shrunk_runs": state["shrunk_runs"],
            "accepted": accepted,
            "repeat": repeat,
            "dry_run": dry_run,
        }
        counting = not shrunk and not repeat and not dry_run
        if counting:
            state["shrunk_runs"] = 0
            state["last_full_count"] = len(entries)
        elif not dry_run:
            state["last_full_count"] = max(
                0, last - len(gone)
            )  # the links removed on purpose are no part of the last list

        retained, removed, missing = [], 0, 0
        for link in list(registry):
            if link in current:
                continue
            if link in gone:
                if not dry_run:
                    local_path(link, pdf_dir).unlink(missing_ok=True)
                    for table in (registry, state["known"], state["missing"]):
                        table.pop(link, None)
                    removed += 1
                continue
            missing += 1
            record = state["missing"].setdefault(link, {"runs": 0, "since": today})
            if counting:
                record["runs"] += 1
            path = local_path(link, pdf_dir)
            if counting and record["runs"] >= grace:
                # the extraction cache entry stays: another listed link may hold the same file, and hand-repaired
                # entries must not be lost; an orphaned entry only costs disk space
                path.unlink(missing_ok=True)
                registry.pop(link, None)
                state["known"].pop(link, None)
                state["missing"].pop(link, None)
                removed += 1
            elif path.exists():
                known = state["known"].get(link)
                retained.append(PdfEntry(**known) if known else _minimal_entry(link, registry, path, cache_dir))
        if retained:
            logging.warning(
                f"[PdfSync] {len(retained)} PDFs missing from this list are kept (removed after {grace} misses in a row)"
            )
        logging.info(
            f"[PdfSync] list check: {len(entries)} entries (last full list {last}), {missing} registered links "
            f"missing, {len(retained)} kept, {removed} removed, shrunk {shrunk}, repeat {repeat}"
        )
        if not dry_run:
            save_state(state_path, state)
        return retained, report, removed

    def keep_missing(
        self, registry: dict[str, str], entries: list[PdfEntry], cache_dir: Path, pdf_dir: str | Path
    ) -> list[PdfEntry]:
        """What a failed safeguard falls back to: never delete or drop anything, keep every missing link that still has
        its file."""
        current = {entry.url for entry in entries}
        return [
            _minimal_entry(link, registry, local_path(link, pdf_dir), cache_dir)
            for link in registry
            if link not in current and local_path(link, pdf_dir).exists()
        ]


def _minimal_entry(link: str, registry: dict[str, str], path: Path, cache_dir: Path) -> PdfEntry:
    """The entry of a registered link without a known list entry; its title comes from its extraction cache entry."""
    title = ""
    try:
        cached = json.loads((Path(cache_dir) / f"{file_hash(path)}.json").read_text(encoding="utf-8"))
        title = str(cached.get("download_title") or "") if isinstance(cached, dict) else ""
    except (OSError, ValueError):
        pass
    return PdfEntry(link, title, registry.get(link) or None)


def _load_state(path: Path) -> dict:
    """A missing file is the first run. A file that exists but cannot be read, is not an object or carries fields of the
    wrong type raises CorruptStateError: the caller keeps every missing PDF, the file stays as it is for diagnosis and the
    build is not published."""
    if not path.exists():
        return {"last_full_count": 0, "shrunk_runs": 0, "last_list": "", "known": {}, "missing": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CorruptStateError(
            f"{path} cannot be read ({exc}); fix or delete it (a missing file is a first run)"
        ) from exc
    why = ""
    known, missing, last_full_count, shrunk_runs = {}, {}, 0, 0  # each set below unless why is
    if not isinstance(state, dict):
        why = "not a JSON object"
    else:
        known, missing = state.get("known", {}), state.get("missing", {})
        if not isinstance(known, dict) or not all(isinstance(v, dict) for v in known.values()):
            why = "'known' is not an object of objects"
        elif not all(_is_entry(v) for v in known.values()):
            why = "'known' holds an entry that is not a PdfEntry (a state of an older format)"
        elif not isinstance(missing, dict) or not all(
            isinstance(v, dict) and isinstance(v.get("runs"), int) for v in missing.values()
        ):
            why = "'missing' is not an object of {runs: int} records"
        else:
            try:
                last_full_count, shrunk_runs = (
                    int(state.get("last_full_count") or 0),
                    int(state.get("shrunk_runs") or 0),
                )
            except (TypeError, ValueError):
                why = "'last_full_count' or 'shrunk_runs' is not a number"
    if why:
        raise CorruptStateError(
            f"{path} does not hold a valid list state ({why}); fix or delete it (a missing file is a first run)"
        )
    return {
        "last_full_count": last_full_count,
        "shrunk_runs": shrunk_runs,
        "last_list": str(state.get("last_list") or ""),
        "known": known,
        "missing": missing,
    }


def _is_entry(record: dict) -> bool:
    try:
        PdfEntry(**record)
    except TypeError:
        return False
    return True


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
