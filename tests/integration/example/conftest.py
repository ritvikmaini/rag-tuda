"""The example profile's leg: collected only when `just harness example` runs it."""

import os

if os.environ.get("RAG_CI_PROFILE") != "example":
    collect_ignore_glob = ["test_*.py"]
