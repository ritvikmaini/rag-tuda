"""Parents and children of a build."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_core.settings.profile import ScopeVocabulary

from indexer.chunking import structure as sc
from indexer.chunking.scope import document_scope

PARENT_CHARS, PARENT_OVERLAP = 1200, 100
CHILD_CHARS, CHILD_OVERLAP = 450, 50
FLAT = (
    "flat"  # metadata "chunk": the document is split recursively, whatever the mode, and its children carry no header
)


@dataclass(frozen=True)
class Child:
    text: str  # what is embedded and stored: the parent's prefix + the child text


@dataclass(frozen=True)
class Parent:
    text: str
    # part_index, filename, scrapedAt, the document's metadata, … (the writer adds section_id)
    metadata: dict[str, Any]
    children: tuple[Child, ...]
    prefix: str  # the header every child starts with ('' for a flat document); metadata["context_header"] when not the title
    source: str = ""  # the producing connector (ingest.sources.<name>.connector), not a metadata key


def _documents(docs, mode):
    """(metadata, [(page_number, text)], connector) per document: in structure mode the page documents of one PDF (consecutive,
    same doc_key) form one document with the first page's metadata and connector; otherwise every input document stands alone."""
    group = None
    for doc in docs:
        meta = doc["metadata"]
        if (
            mode == "structure"
            and group
            and meta.get("source_type") == "pdf"
            and group[0].get("source_type") == "pdf"
            and group[0].get("doc_key") == meta.get("doc_key")
        ):
            group[1].append((meta.get("page_number"), doc["page_content"]))
            continue
        if group:
            yield group
        group = (meta, [(meta.get("page_number"), doc["page_content"])], doc.get("source", ""))
    if group:
        yield group


def _parent_units(docs, mode, parent_chars, parent_overlap=PARENT_OVERLAP, scope=None):
    """(metadata, parent text, page_number or None, section path, connector) for every parent, in document order. In
    structure mode a document's own scope statement (scope.document_scope) goes into the metadata of
    every parent as "scope" when there is one; the children never see it (context_header reads other keys)."""
    recursive = RecursiveCharacterTextSplitter(chunk_size=parent_chars, chunk_overlap=parent_overlap)
    for meta, pages, source in _documents(docs, mode):
        if mode != "structure" or meta.get("chunk") == FLAT:
            for _, text in pages:
                for part in recursive.split_text(text):
                    yield meta, part, None, (), source
            continue
        line = document_scope(meta.get("title") or "", pages, scope) if scope else ""
        if line:
            meta = {**meta, "scope": line}
        trail = tuple(t for t in (meta.get("title_path") or "").split(" > ") if t)
        for text, page, path in sc.structure_parents(pages, parent_chars, parent_overlap):
            yield meta, text, page, trail + path, source


def chunk_documents(
    docs,
    mode="recursive",
    parent_chars=PARENT_CHARS,
    children="recursive",
    header="title",
    contexts=None,
    *,
    parent_overlap=PARENT_OVERLAP,
    child_chars=CHILD_CHARS,
    child_overlap=CHILD_OVERLAP,
    scope: ScopeVocabulary | None = None,
) -> list[Parent]:
    """Parents and their children (the writer's input).
    mode 'recursive': parents of about parent_chars characters,
    never across PDF pages; 'structure': structure_chunker (parents follow sections and may run across pages with
    '[Page N]' marks; metadata section_path). children 'recursive' (child_chars, overlap child_overlap) or 'sentences'
    (three-sentence windows), always inside one page of the parent. scope: the profile's ingest.scope; in structure mode
    each document's scope statement becomes the parents' metadata "scope". header 'title' (children start with '<title>: ')
    or 'tier1' (structure_chunker.context_header); contexts {doc key: tier-2 text} go in front of it. A prefix other
    than '<title>: ' is stored as the parent's context_header (the backend strips it from quotes). Documents whose metadata
    says chunk: flat: recursive in every mode, no prefix, no scope statement."""
    if (
        mode not in ("recursive", "structure")
        or children not in ("recursive", "sentences")
        or header not in ("title", "tier1")
    ):
        raise ValueError(f"unknown chunking option: mode={mode!r} children={children!r} header={header!r}")
    scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    part_index = {}
    parents = []
    for meta, text, page, path, source in _parent_units(docs, mode, parent_chars, parent_overlap, scope):
        key = meta.get("doc_key") or meta.get("url") or ""
        index = part_index.get(key, 0)
        part_index[key] = index + 1
        flat = meta.get("chunk") == FLAT
        title_prefix = (meta.get("title") or "") + ": "
        prefix = ""
        if not flat:
            prefix = sc.context_header(meta, path) if header == "tier1" else title_prefix
            context = (contexts or {}).get(meta.get("doc_key") or meta.get("download_url") or meta.get("url") or "")
            if context:
                prefix = context.strip() + " " + prefix
        kids = sc.child_texts(text, "recursive" if flat else children, child_chars, child_overlap)
        parent_meta = {
            **meta,
            "part_index": index,
            "filename": meta.get("title") or key,
            "scrapedAt": scraped_at,
        }
        if page is not None:
            parent_meta["page_number"] = page
        if mode == "structure":
            parent_meta["section_path"] = " > ".join(path)
        if prefix and prefix != title_prefix:
            parent_meta["context_header"] = prefix
        parents.append(Parent(text, parent_meta, tuple(Child(prefix + kid) for kid in kids), prefix, source))
    return parents
