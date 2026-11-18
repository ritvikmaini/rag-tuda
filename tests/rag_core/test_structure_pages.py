"""The structure chunker's page marks against the backend's page mapping."""

import importlib.util
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from rag_core.domain.context_budget import page_of_position  # noqa: E402
from rag_core.domain.source_meta import strip_source_text  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "structure_chunker", os.path.join(ROOT, "apps", "indexer", "src", "indexer", "chunking", "structure.py")
)
assert _spec is not None and _spec.loader is not None
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def test_every_position_maps_to_its_page():
    pages = [
        (4, "§ 2 Kernzeit\n\nDie Kernzeit beginnt um 9 Uhr"),
        (5, "und endet um 15 Uhr."),
        (6, "Ausnahmen regelt § 3."),
    ]
    [(text, page, _)] = sc.structure_parents(pages, 1200)
    assert page == 4
    assert page_of_position(text, text.index("§ 2"), -1) == 4
    assert page_of_position(text, text.index("Die Kernzeit"), -1) == 4
    assert page_of_position(text, text.index("und endet"), -1) == 5
    assert page_of_position(text, text.index("Ausnahmen"), -1) == 6
    assert "[Page" not in strip_source_text(text)
