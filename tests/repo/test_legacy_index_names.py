"""Nothing tracked reads the legacy index any more: no product file,
script, profile or the readme names the two legacy views or their two legacy columns. Tests are not scanned (they name
the legacy relations to prove a fresh database has none)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("rag_production", "langchain_id", "langchain_metadata", "content_tsv")
ROOTS = (
    "apps",
    "packages",
    "deploy",
    "tools",
    "profiles",
    "README.md",
)


def test_no_tracked_product_file_names_the_legacy_index():
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *ROOTS], cwd=ROOT, capture_output=True, text=True, check=True
    )
    hits = sorted(
        f
        for f in listed.stdout.split("\0")
        if f and any(n in (ROOT / f).read_text(encoding="utf-8", errors="ignore") for n in NAMES)
    )
    assert hits == []
