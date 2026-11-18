"""The child's own text without the index prefix: quotes and window anchors."""

from rag_core.domain import citations as ct  # noqa: E402
from rag_core.domain.child_text import child_body  # noqa: E402
from rag_core.domain.context_budget import _window_and_page  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

HEADER = "Dienstanweisung KI · Dienstanweisung/Dienstvereinbarung · Stand 2025 · § 3 Erlaubte Anwendungen: "


def test_child_body():
    assert child_body({"context_header": HEADER, "title": "Dienstanweisung KI"}, HEADER + "Text.") == "Text."
    assert child_body({"title": "Dienstanweisung KI"}, "Dienstanweisung KI: Text.") == "Text."
    assert child_body({"context_header": HEADER}, "Anderer Text.") == "Anderer Text."
    assert child_body(None, None) == ""


def test_quote_and_window_without_the_header():
    meta = {
        "section_id": "a",
        "title": "Dienstanweisung KI",
        "context_header": HEADER,
        "page_number": 7,
        "matched_child_text": HEADER + "Verlaufsdaten sind zu deaktivieren.",
    }
    source = {
        "n": 1,
        "title": "Dienstanweisung KI",
        "origin": "Archiv",
        "type": "Dienstanweisung",
        "date": "2025",
        "url": "u",
        "page": 7,
        "section_id": "a",
    }
    entries = ct.entries([source], [Document(page_content="x", metadata=meta)])
    assert entries[0]["quote"] == "Verlaufsdaten sind zu deaktivieren."
    parent = "[Page 7]\nEinleitung.\n\n[Page 8]\nVerlaufsdaten sind zu deaktivieren."
    window, page = _window_and_page(Document(page_content=parent, metadata=meta), 20)
    assert page == 8 and "Verlaufsdaten" in window
