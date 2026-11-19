"""The build gates: the size reference of the last swapped build."""

import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from rag_core.domain.sources import CorruptStateError

from indexer.extract.quality import write_json_atomic

STATS_NAME = "corpus_build_stats.json"  # in the state folder


def previous_stats(state_dir: str):
    """The last swapped build's counts, the size gate's reference. No file: no gate. A file that exists but is not a
    JSON object raises CorruptStateError: the build is refused instead of running without its gate."""
    stats_file = os.path.join(state_dir, STATS_NAME)
    if not os.path.exists(stats_file):
        return {}
    try:
        with open(stats_file, encoding="utf-8") as fh:
            stats = json.load(fh)
    except (OSError, ValueError) as exc:
        raise CorruptStateError(f"{stats_file} cannot be read ({exc}); fix or delete it before the next build") from exc
    if not isinstance(stats, dict):
        raise CorruptStateError(f"{stats_file} does not hold a JSON object; fix or delete it before the next build")
    return stats


def record_build_stats(state_dir: str, result):
    """Called after a successful swap; the next build is compared against these counts.
    Returns False (and logs) when the file cannot be written: the swap has happened, so
    the reply must not fail because of the stats file."""
    try:
        os.makedirs(state_dir, exist_ok=True)  # a native first build starts without the folder
        write_json_atomic(
            os.path.join(state_dir, STATS_NAME),
            {k: result[k] for k in ("parents", "children", "documents") if k in result}
            | {"built_at": datetime.now().isoformat()},
        )
        return True
    except OSError as exc:
        logging.warning(f"[Corpus] build stats not written ({exc}); the next build has no size gate")
        return False


def source_gate(name: str, count: int, minimum: int, kind: str) -> dict[str, Any]:
    """kind 'documents' (PageItems) or 'pdfs' (PdfItems that produced a document)."""
    return {"source": name, "kind": kind, "count": count, "minimum": minimum, "passed": count >= minimum}


def hosts_gate(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Passes when no source reports a host of its set as down (`hosts_down` in its report): a lost host is not a smaller
    corpus the ratio may absorb."""
    down = [d["host"] for report in reports.values() for d in report.get("hosts_down", ())]
    return {"down": down, "passed": not down}


def size_gate(parents: int, previous: Mapping[str, Any], ratio: float) -> dict[str, Any]:
    last = previous.get("parents")
    return {"count": parents, "last": last, "ratio": ratio, "passed": not last or parents >= ratio * last}


def refusal(gates: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The first failed gate's message, in the order the gates were added (the sources in profile order, then the hosts, then size)."""
    for name, gate in gates.items():
        if gate["passed"]:
            continue
        if name == "hosts":
            return f"{', '.join(gate['down'])} could not be reached (robots.txt unreadable); swap refused"
        if name == "size":
            return f"corpus shrank to {gate['count']} parents (last build {gate['last']}); swap refused"
        what = "PDFs produced documents" if gate["kind"] == "pdfs" else "documents"
        return f"only {gate['count']} {gate['source']} {what} (minimum {gate['minimum']}); swap refused"
    return None
