"""Nothing tracked reads the legacy index any more: no product file,
script, profile or the readme names the two legacy views or their two legacy columns. Allowed: the cut and its reverse
and a misuse probe's question text. Tests are not scanned (they name the legacy relations to prove the reverse)."""

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
ALLOWED = {
    "deploy/sql/contract.sql": "the cut",
    "deploy/sql/contract_reverse.sql": "the rollback's reverse",
}


def test_no_tracked_product_file_names_the_legacy_index():
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *ROOTS], cwd=ROOT, capture_output=True, text=True, check=True
    )
    hits = sorted(
        f
        for f in listed.stdout.split("\0")
        if f and f not in ALLOWED and any(n in (ROOT / f).read_text(encoding="utf-8", errors="ignore") for n in NAMES)
    )
    assert hits == []
