"""The TU profile's leg: collected only when `just harness tu` runs it."""

import os

if os.environ.get("RAG_CI_PROFILE") != "tu":
    collect_ignore_glob = ["test_*.py"]
