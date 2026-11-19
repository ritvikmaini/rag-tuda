"""Citation frames for clickable citations on the page."""

from assistant_api.sse import frames as fr
from rag_core.domain import citations as ct  # noqa: E402

URL = "https://docs.example.org/archiv/dienstanweisung-ki-2025-02/?layout=file"


def content(frame):
    obj = fr.wire(frame)
    assert obj["type"] == "citations"
    return obj["content"]


def citations_frame(sources, docs):
    from assistant_api.sse.events import citations_frame as frame

    return frame(sources, docs)


class Doc:
    def __init__(self, metadata):
        self.metadata = metadata
        self.page_content = "parent text"


def source(n, section_id, url=URL, page=4, **extra):
    return {
        "n": n,
        "title": "Dienstanweisung KI",
        "origin": "Archiv",
        "type": "Dienstanweisung",
        "date": "2025",
        "url": url,
        "page": page,
        "section_id": section_id,
        "kind": "full",
        **extra,
    }


def doc(section_id, child, title="Dienstanweisung KI"):
    return Doc({"section_id": section_id, "title": title, "matched_child_text": child})


def test_entries_follow_the_numbers_with_link_page_and_facts():
    entries = ct.entries(
        [source(2, "b", page=12), source(1, "a")], [doc("a", "Erster Satz."), doc("b", "Zweiter Satz.")]
    )
    assert [e["n"] for e in entries] == [1, 2]
    assert entries[0] == {
        "n": 1,
        "url": URL,
        "page": 4,
        "title": "Dienstanweisung KI",
        "type": "Dienstanweisung",
        "origin": "Archiv",
        "date": "2025",
        "quote": "Erster Satz.",
    }
    assert entries[1]["page"] == 12 and entries[1]["quote"] == "Zweiter Satz."


def test_page_is_null_unless_a_positive_page_number():
    entries = ct.entries(
        [
            source(1, "a", page=-1),
            source(2, "b", page=None),  # pyright: ignore[reportArgumentType]
            source(3, "c", page=0),
            source(4, "d", page=True),
        ],
        [],
    )
    assert [e["page"] for e in entries] == [None, None, None, None]


def test_quote_drops_the_title_prefix_page_markers_and_extra_whitespace():
    child = "Dienstanweisung KI: [Page 4]\nVerlaufsdaten sind   zu deaktivieren. <!-- image -->"
    entries = ct.entries([source(1, "a")], [doc("a", child)])
    assert entries[0]["quote"] == "Verlaufsdaten sind zu deaktivieren."


def test_a_long_quote_is_cut_at_a_word_boundary():
    child = " ".join(["Dienstvereinbarung"] * 30)
    quote = ct.entries([source(1, "a")], [doc("a", child)])[0]["quote"]
    assert len(quote) <= ct.QUOTE_CHARS + 1
    assert quote.endswith("…")
    assert quote[:-1].split(" ")[-1] == "Dienstvereinbarung"


def test_duplicate_or_unknown_ids_give_no_quote():
    docs = [doc("a", "Eins."), doc("a", "Zwei."), doc("b", "Drei.")]
    entries = ct.entries([source(1, "a"), source(2, "x"), source(3, None)], docs)
    assert [e["quote"] for e in entries] == ["", "", ""]


def test_dict_documents_are_read_like_document_objects():
    docs = [{"page_content": "x", "metadata": {"section_id": "a", "title": "", "matched_child_text": "Aus dem Dict."}}]
    assert ct.entries([source(1, "a")], docs)[0]["quote"] == "Aus dem Dict."


def test_an_entry_without_url_is_kept_for_the_page_to_decide():
    entries = ct.entries([source(1, "a", url="")], [])
    assert entries[0]["url"] == ""


def test_invalid_and_repeated_numbers_are_skipped():
    sources = [
        source(0, "a"),
        source("2", "b"),
        source(True, "c"),
        source(3, "d", page=5),
        source(3, "e", page=9),
        "kein dict",
    ]
    entries = ct.entries(sources, [])
    assert [(e["n"], e["page"]) for e in entries] == [(3, 5)]


def test_no_numbered_sources_give_no_frame():
    assert citations_frame([], []) is None
    assert citations_frame(None, None) is None
    assert citations_frame([{"n": 0}], []) is None


def test_unexpected_input_never_raises():
    assert citations_frame(5, []) is None
    assert citations_frame([source(1, "a")], 5) is None
    frame = citations_frame([source(1, ["not", "hashable"])], [Doc(None), object(), {"metadata": "x"}])
    assert content(frame)[0]["quote"] == ""


def test_frames_are_ascii_json_like_the_other_frames():
    frame = citations_frame(
        [source(1, "a", title="Gebührenübersicht")], [doc("a", "Größe … Maß", title="Gebührenübersicht")]
    )
    assert frame is not None and fr.encode(frame).isascii()
    assert content(frame)[0]["title"] == "Gebührenübersicht"
    assert content(frame)[0]["quote"] == "Größe … Maß"


def test_quotes_drop_the_markdown_heading_markers_of_the_chunk():
    chunk = "## § 2 Geltungsbereich\nDiese Ordnung gilt für Studierende.\n### (1) Absatz\nText # kein Marker."
    [entry] = ct.entries([source(1, "a")], [doc("a", chunk)])
    assert entry["quote"] == "§ 2 Geltungsbereich Diese Ordnung gilt für Studierende. (1) Absatz Text # kein Marker."
