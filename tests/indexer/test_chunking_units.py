"""The chunker: parents and children."""

import pytest
from rag_core.settings.profile import ScopeVocabulary

from indexer.chunking import structure, units


def _doc(text, **meta):
    base = {
        "title": "Gebührensatzung",
        "url": "https://docs.example.org/x.pdf",
        "source_type": "pdf",
        "doc_type": "Satzung/Verordnung",
        "origin": "website",
        "doc_key": "https://docs.example.org/x.pdf",
        "page_number": 3,
    }
    base.update(meta)
    return {"page_content": text, "metadata": base}


def test_parents_are_bounded_and_keep_page_metadata():
    text = ("§ 1 Gebührengegenstand. Die Nutzung der Bibliothek auf dem Campus unterliegt der Gebühr. " * 40).strip()
    parents = units.chunk_documents([_doc(text)])
    assert len(parents) > 1
    assert all(len(p.text) <= units.PARENT_CHARS for p in parents)
    assert all(p.metadata["page_number"] == 3 and p.metadata["origin"] == "website" for p in parents)
    assert [p.metadata["part_index"] for p in parents] == list(range(len(parents)))
    assert all("section_id" not in p.metadata for p in parents)  # the writer sets the section id


def test_children_carry_title_prefix_and_parent_id():
    parents = units.chunk_documents([_doc("§ 2 Gebührenpflicht. " * 60)])
    child = parents[0].children[0]
    assert child.text.startswith("Gebührensatzung: ")
    assert all(len(c.text) <= units.CHILD_CHARS + len("Gebührensatzung: ") for p in parents for c in p.children)


def test_flat_documents_children_have_no_prefix():
    parents = units.chunk_documents(
        [
            _doc(
                "Mensa Nord\nAdresse: Beispielstraße 1",
                chunk="flat",
                title="Mensa Nord",
                doc_key="flat1",
            )
        ]
    )
    assert parents[0].children[0].text.startswith("Mensa Nord\n")


def test_part_index_counts_per_document():
    parents = units.chunk_documents(
        [_doc("a" * 50, doc_key="d1"), _doc("b" * 50, doc_key="d2"), _doc("c" * 50, doc_key="d1")]
    )
    assert [(p.metadata["doc_key"], p.metadata["part_index"]) for p in parents] == [("d1", 0), ("d2", 0), ("d1", 1)]


def _plain_chunk(docs):
    """A plain two-level recursive splitter chunker (the reference for recursive mode)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
    out = []
    for doc in docs:
        meta = doc["metadata"]
        for part in parent_splitter.split_text(doc["page_content"]):
            prefix = "" if meta.get("chunk") == "flat" else (meta.get("title") or "") + ": "
            out.append(
                (part, [prefix + c for c in child_splitter.split_text(part)], meta.get("page_number"), sorted(meta))
            )
    return out


def test_recursive_mode_matches_the_plain_text_splitter():
    text = ("§ 1 Gebührengegenstand. Die Nutzung der Bibliothek auf dem Campus unterliegt der Gebühr. " * 40).strip()
    docs = [_doc(text), _doc("Kurz.", page_number=4), _doc("Eintrag Bibliothek", chunk="flat", title="Bibliothek")]
    new = [
        (
            p.text,
            [c.text for c in p.children],
            p.metadata["page_number"],
            sorted(k for k in p.metadata if k not in ("section_id", "part_index", "filename", "scrapedAt")),
        )
        for p in units.chunk_documents(docs)
    ]
    assert new == _plain_chunk(docs)


def test_structure_mode_pages_metadata_and_header():
    meta = {"doc_type": "Dienstanweisung/Dienstvereinbarung", "doc_year": 2025}
    p1 = _doc("§ 2 Kernzeit\n\nDie Kernzeit beginnt um 9 Uhr", page_number=1, **meta)
    p2 = _doc("und endet um 15 Uhr.", page_number=2, **meta)
    web = {
        "page_content": "Text der Seite über Kernzeiten.",
        "metadata": {
            "title": "Kernzeit",
            "url": "https://docs.example.org/k",
            "source_type": "page",
            "doc_type": "Webseite",
            "origin": "website",
            "doc_key": "https://docs.example.org/k",
            "page_number": -1,
            "title_path": "Bibliothek > Personal",
        },
    }
    pdf, page = units.chunk_documents([p1, p2, web], mode="structure", header="tier1")
    header = pdf.metadata["context_header"]
    assert header == "Gebührensatzung · Dienstanweisung/Dienstvereinbarung · Stand 2025 · § 2 Kernzeit: "
    assert pdf.metadata["page_number"] == 1 and pdf.text.startswith("[Page 1]\n")
    assert pdf.metadata["section_path"] == "§ 2 Kernzeit"
    for child in pdf.children:
        body = child.text[len(header) :]
        assert child.text.startswith(header) and "[Page" not in body and body[:120] in pdf.text
    assert page.metadata["page_number"] == -1 and page.metadata["section_path"] == "Bibliothek > Personal"
    assert page.metadata["context_header"] == "Kernzeit · Webseite · Bibliothek > Personal: "
    ctx = units.chunk_documents(
        [p1], mode="structure", contexts={"https://docs.example.org/x.pdf": "Regelt die Arbeitszeit."}
    )
    assert ctx[0].metadata["context_header"] == "Regelt die Arbeitszeit. Gebührensatzung: "
    assert ctx[0].children[0].text.startswith("Regelt die Arbeitszeit. Gebührensatzung: ")


SCOPE = ScopeVocabulary(groups={"Beamtinnen und Beamte": ("beamt",)}, areas=(), compilations=())


def test_a_documents_scope_reaches_every_parent_and_no_child():
    """The key is section metadata only; children and their prefix stay byte for byte."""
    p1 = _doc(
        "Rundschreiben 2025\nAn alle Beamtinnen und Beamten der Fachbereiche\n" + "Text zur Ansparung. " * 120,
        page_number=1,
    )
    p2 = _doc("Weiterer Text. " * 160, page_number=2)
    flat = _doc("Mensa\nAn alle Beamtinnen und Beamten", chunk="flat", doc_key="flat1", title="Mensa")
    scoped = units.chunk_documents([p1, p2, flat], mode="structure", header="tier1", scope=SCOPE)
    plain = units.chunk_documents([p1, p2, flat], mode="structure", header="tier1")
    pdf = [p for p in scoped if "chunk" not in p.metadata]
    assert len(pdf) > 1 and {p.metadata["scope"] for p in pdf} == {"An alle Beamtinnen und Beamten der Fachbereiche"}
    assert all("scope" not in p.metadata for p in scoped if p.metadata.get("chunk") == "flat")
    assert [(p.text, p.prefix, [c.text for c in p.children]) for p in scoped] == [
        (p.text, p.prefix, [c.text for c in p.children]) for p in plain
    ]
    unscoped = units.chunk_documents([_doc("An alle Fachbereiche und Dezernate\nText.")], mode="structure", scope=SCOPE)
    assert "scope" not in unscoped[0].metadata
    recursive = units.chunk_documents([p1], scope=SCOPE)  # recursive mode: pages stand alone, no document scope
    assert all("scope" not in p.metadata for p in recursive)


def test_unknown_chunking_option_fails_the_build():

    with pytest.raises(ValueError):
        units.chunk_documents([_doc("Text.")], mode="semantic")


class _Recorder:
    sizes: list = []


def _record_splitters(monkeypatch):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    sizes = []

    class Recording(RecursiveCharacterTextSplitter):
        def __init__(self, chunk_size=4000, chunk_overlap=200, **kwargs):
            sizes.append((chunk_size, chunk_overlap))
            super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    monkeypatch.setattr(units, "RecursiveCharacterTextSplitter", Recording)
    monkeypatch.setattr(structure, "RecursiveCharacterTextSplitter", Recording)
    return sizes


def test_the_profile_values_reach_the_splitters(monkeypatch):
    sizes = _record_splitters(monkeypatch)
    units.chunk_documents([_doc("Satz eins. " * 200)])
    assert (1200, 100) in sizes and (450, 50) in sizes
    sizes.clear()
    units.chunk_documents(
        [_doc("Satz eins. " * 200)], parent_chars=999, parent_overlap=77, child_chars=333, child_overlap=22
    )
    assert (999, 77) in sizes and (333, 22) in sizes and (1200, 100) not in sizes and (450, 50) not in sizes


def test_every_child_starts_with_its_parents_prefix_and_the_prefix_is_recorded():
    meta = {"doc_type": "Dienstanweisung/Dienstvereinbarung"}
    tier1 = units.chunk_documents(
        [_doc("§ 2 Kernzeit\n\nDie Kernzeit beginnt um 9 Uhr " * 3, page_number=1, **meta)],
        mode="structure",
        header="tier1",
    )
    for p in tier1:
        assert p.prefix == p.metadata["context_header"] and p.children
        assert all(c.text.startswith(p.prefix) for c in p.children)
    [title] = units.chunk_documents([_doc("Kurz.")])
    assert title.prefix == "Gebührensatzung: " and "context_header" not in title.metadata
    assert all(c.text.startswith(title.prefix) for c in title.children)
    [flat] = units.chunk_documents([_doc("Bibliothek\nAdresse: Weg 1", chunk="flat", title="Bibliothek")])
    assert flat.prefix == "" and "context_header" not in flat.metadata


def test_each_parent_names_the_connector_of_its_document():
    docs = [
        {
            "page_content": "# A\n\nErster Text mit genug Inhalt.",
            "metadata": {"url": "https://a.example/", "title": "A", "source_type": "page"},
            "source": "one.connector",
        },
        {
            "page_content": "# B\n\nZweiter Text mit genug Inhalt.",
            "metadata": {"url": "https://b.example/", "title": "B", "source_type": "page"},
            "source": "two.connector",
        },
    ]
    parents = units.chunk_documents(docs, mode="structure", children="recursive", header="tier1")
    assert {p.metadata["url"]: p.source for p in parents} == {
        "https://a.example/": "one.connector",
        "https://b.example/": "two.connector",
    }
    assert all(
        "source" not in p.metadata for p in parents
    )  # the connector is no metadata key (sections.metadata is unchanged)
