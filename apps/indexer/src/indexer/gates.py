"""The build gates: the size reference of the last swapped build."""

import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from rag_core.domain.sources import CorruptStateError

from indexer.extract.quality import write_json_atomic
from indexer.state import LOCAL_DATA_FOLDER

STATS_FILE = os.path.join(LOCAL_DATA_FOLDER, "vector_database_contents", "corpus_build_stats.json")


def previous_stats():
    """The last swapped build's counts, the size gate's reference. No file: no gate. A file that exists but is not a
    JSON object raises CorruptStateError: the build is refused instead of running without its gate."""
    if not os.path.exists(STATS_FILE):
        return {}
    try:
        with open(STATS_FILE, encoding="utf-8") as fh:
            stats = json.load(fh)
    except (OSError, ValueError) as exc:
        raise CorruptStateError(f"{STATS_FILE} cannot be read ({exc}); fix or delete it before the next build") from exc
    if not isinstance(stats, dict):
        raise CorruptStateError(f"{STATS_FILE} does not hold a JSON object; fix or delete it before the next build")
    return stats


def record_build_stats(result):
    """Called after a successful swap; the next build is compared against these counts.
    Returns False (and logs) when the file cannot be written: the swap has happened, so
    the reply must not fail because of the stats file."""
    try:
        write_json_atomic(
            STATS_FILE,
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


def size_gate(parents: int, previous: Mapping[str, Any], ratio: float) -> dict[str, Any]:
    last = previous.get("parents")
    return {"count": parents, "last": last, "ratio": ratio, "passed": not last or parents >= ratio * last}


def refusal(gates: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The first failed gate's message, in the order the gates were added (the sources in profile order, then size)."""
    for name, gate in gates.items():
        if gate["passed"]:
            continue
        if name == "size":
            return f"corpus shrank to {gate['count']} parents (last build {gate['last']}); swap refused"
        what = "PDFs produced documents" if gate["kind"] == "pdfs" else "documents"
        return f"only {gate['count']} {gate['source']} {what} (minimum {gate['minimum']}); swap refused"
    return None
