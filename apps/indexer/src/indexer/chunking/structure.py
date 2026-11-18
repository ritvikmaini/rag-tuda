"""
Structure-aware chunking.

A document (all pages of one PDF with '[Page N]' lines, or one web page) is read as blocks:
headings (Markdown '#', numbered '1.' / '1.2.3', '§ 5', 'Artikel 3', 'Teil'/'Abschnitt'/'Anlage',
all-caps title lines; a heading is a paragraph of one short line without sentence punctuation),
tables (Markdown '|' rows, box-drawing rows) and text paragraphs. Headings give every block a
section path. Parents are sections: consecutive small sections merged up to a target size, a long
section cut at block boundaries, a long table split by rows (a short Markdown header repeated), a
long paragraph split by the recursive splitter; a heading never ends a parent while text follows.
A parent on more than one page starts with its first page's mark and carries a mark where each
further page starts (rag_core's context_budget.page_of_position); its page is that of its first
character. `paragraphs` lists the '§ n' a parent starts a paragraph with. Children are cut inside one page segment of
a parent (never across a mark), so each child is a substring of its parent. Imports only re, dataclasses and
langchain_text_splitters, so the tests and the offline harness can load this file by path.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

PAGE_LINE = re.compile(r"^\[Page (\d+)\]$")
PAGE_MARK = re.compile(r"\[Page \d+\]\n")
MD_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
NUMBERED = re.compile(r"^((?:\d{1,2}\.){1,4}\d{0,2})\s+[A-ZÄÖÜ]")
PARAGRAPH = re.compile(
    r"^\s*§[ \t]*(\d+[a-z]?)(?:[ \t]+(?!(?:Abs|Absatz|Satz|Nr|Nummer|Ziffer|Buchstabe)\b)[A-ZÄÖÜ][a-zäöüß]|\s*$)"
)
LINE_END = re.compile(r"[.:;!?)]$")
LEGAL = re.compile(r"^(?:§{1,2}\s*\d+[a-z]?|Art(?:ikel|\.)\s*\d+[a-z]?)(?:\s|$)")
PART = re.compile(r"^(?:Teil|Abschnitt|Kapitel|Anlage|Anhang)\s+(?:[IVXLC]+|\d+[a-z]?)\b")
ALL_CAPS = re.compile(r"^[A-ZÄÖÜ0-9][A-ZÄÖÜß0-9 .,&/()–-]{3,79}$")
LEADER = re.compile(r"\.{4,}|…{2,}|\s{3,}\d{1,3}$|\t\d{1,3}$")
SENTENCE_END = re.compile(r"[.,;:!]$")
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ„\"(])")
BOX = set("│┃┌┐└┘├┤┬┴┼─━║═╔╗╚╝╠╣╦╩╬")
MAX_HEADING_CHARS = 100
MAX_HEADING_WORDS = 12
MAX_PATH_TITLE = 80
MAX_TABLE_HEADER = 300


@dataclass
class Block:
    kind: str  # "heading", "table" or "text"
    text: str
    page: int  # the page the block is on (-1: no page marks)
    level: int = 0  # headings: 1 = top
    title: str = ""  # headings: the title used in section paths
    path: Tuple[str, ...] = ()  # the section path of the block (a heading's includes itself)


def heading(line: str) -> Optional[Tuple[int, str]]:
    """(level, title) when a line reads as a heading, else None."""
    text = (line or "").strip()
    if not text or len(text) > MAX_HEADING_CHARS or LEADER.search(text):
        return None
    m = MD_HEADING.match(text)
    if m:
        return len(m.group(1)), m.group(2).strip()
    if SENTENCE_END.search(text) or len(text.split()) > MAX_HEADING_WORDS:
        return None
    if PART.match(text):
        return 1, text
    if LEGAL.match(text):
        return 2, text
    m = NUMBERED.match(text)
    if m:
        return len(re.findall(r"\d+", m.group(1))), text
    if ALL_CAPS.match(text) and sum(ch.isalpha() for ch in text) >= 4:
        return 1, text
    return None


def is_table_line(line: str) -> bool:
    s = (line or "").lstrip()
    return bool(s) and (s[0] == "|" or s[0] in BOX)


def parse_blocks(text: str, page: int = -1) -> List[Block]:
    """Blocks in reading order. '[Page N]' lines set the page and end the current block; blank lines end a paragraph;
    consecutive table lines form one table; a one-line paragraph that reads as a heading is a heading."""
    blocks: List[Block] = []
    lines: List[str] = []
    kind = None

    def flush():
        nonlocal lines, kind
        body = "\n".join(lines).strip("\n")
        if body.strip():
            found = heading(lines[0]) if kind == "text" and len(lines) == 1 else None
            if found:
                blocks.append(Block("heading", body.strip(), page, found[0], found[1][:MAX_PATH_TITLE]))
            else:
                blocks.append(Block(kind or "", body, page))  # a line was added, so kind is set
        lines, kind = [], None

    for line in (text or "").split("\n"):
        mark = PAGE_LINE.match(line.strip())
        if mark:
            flush()
            page = int(mark.group(1))
            continue
        if not line.strip():
            flush()
            continue
        line_kind = "table" if is_table_line(line) else "text"
        if kind is not None and line_kind != kind:
            flush()
        kind = line_kind
        lines.append(line.rstrip())
    flush()
    return blocks


def _numbers(title: str) -> Optional[Tuple[int, ...]]:
    m = NUMBERED.match(title or "")
    return tuple(int(x) for x in re.findall(r"\d+", m.group(1))) if m else None


def assign_paths(blocks: List[Block]) -> List[Block]:
    """Sets every block's section path from the headings before it: a heading closes those of its level and below, and a
    numbered heading also closes numbered sections whose number is not a prefix of its own (so '5.1.6' never sits under
    '3.' when '5.' and '5.1' were not recognised as headings)."""
    stack: List[Tuple[int, str, Optional[Tuple[int, ...]]]] = []
    for block in blocks:
        if block.kind == "heading":
            numbers = _numbers(block.title)
            while stack and (
                stack[-1][0] >= block.level
                or (numbers and stack[-1][2] and numbers[: len(stack[-1][2])] != stack[-1][2])
            ):
                stack.pop()
            stack.append((block.level, block.title, numbers))
        block.path = tuple(title for _, title, _ in stack)
    return blocks


def sections(blocks: Sequence[Block]) -> List[List[Block]]:
    """Runs of blocks that start at a heading (the first run may start without one)."""
    out: List[List[Block]] = []
    for block in blocks:
        if block.kind == "heading" or not out:
            out.append([block])
        else:
            out[-1].append(block)
    return out


def joined_len(blocks: Sequence[Block]) -> int:
    return sum(len(b.text) for b in blocks) + 2 * max(len(blocks) - 1, 0)


def split_block(block: Block, target: int, overlap: int) -> List[Block]:
    """A block longer than target as pieces: a table by rows (a Markdown header of at most MAX_TABLE_HEADER characters
    repeated in every piece; a single row longer than target stays whole), text by the recursive splitter."""
    if block.kind == "table":
        rows = block.text.split("\n")
        header: List[str] = []
        if (
            len(rows) > 2
            and rows[0].lstrip().startswith("|")
            and TABLE_SEPARATOR.match(rows[1].strip())
            and len(rows[0]) + len(rows[1]) + 1 <= MAX_TABLE_HEADER
        ):
            header, rows = rows[:2], rows[2:]
        pieces, current = [], list(header)
        for row in rows:
            if len(current) > len(header) and len("\n".join(current + [row])) > target:
                pieces.append("\n".join(current))
                current = list(header)
            current.append(row)
        if len(current) > len(header):
            pieces.append("\n".join(current))
    else:
        pieces = RecursiveCharacterTextSplitter(chunk_size=target, chunk_overlap=overlap).split_text(block.text)
    return [Block(block.kind, p, block.page, block.level, block.title, block.path) for p in pieces]


def pack(blocks: Sequence[Block], target: int, overlap: int) -> List[List[Block]]:
    """Parents as lists of blocks: a section stays whole while it fits, consecutive small sections are merged up to
    target, a longer section is cut at block boundaries, a block longer than target is split (split_block). Headings
    at the end of a full parent move to the next one."""
    parents: List[List[Block]] = []
    current: List[Block] = []

    def close():
        nonlocal current
        tail: List[Block] = []
        while current and current[-1].kind == "heading":
            tail.insert(0, current.pop())
        if current:
            parents.append(current)
        current = tail

    for section in sections(blocks):
        if joined_len(section) <= target:
            if current and joined_len(current + section) > target:
                close()
            current = current + section
            continue
        for block in section:
            for piece in split_block(block, target, overlap) if len(block.text) > target else [block]:
                if current and joined_len(current + [piece]) > target:
                    close()
                current.append(piece)
    if current:
        parents.append(current)
    return parents


def common_path(blocks: Sequence[Block]) -> Tuple[str, ...]:
    """The section path all blocks of a parent share (their longest common prefix)."""
    prefix: List[str] = []
    for parts in zip(*[b.path for b in blocks]):
        if any(p != parts[0] for p in parts):
            break
        prefix.append(parts[0])
    return tuple(prefix)


def assemble(blocks: Sequence[Block]) -> Tuple[str, int]:
    """A parent's text and page: blocks joined by blank lines; a parent on more than one page starts with its first
    page's mark and carries a mark where each further page starts."""
    multi = len({b.page for b in blocks if b.page > 0}) > 1
    parts, previous = [], None
    for block in blocks:
        mark = f"[Page {block.page}]\n" if multi and block.page > 0 and block.page != previous else ""
        parts.append(mark + block.text)
        previous = block.page
    return "\n\n".join(parts), blocks[0].page


def structure_parents(
    pages: Sequence[Tuple[Optional[int], str]], target: int, overlap: int = 100
) -> List[Tuple[str, Optional[int], Tuple[str, ...]]]:
    """[(parent text, page number or None, section path)] of one document given as [(page number, text)] (a web page:
    page None or -1)."""
    paged = any(p and p > 0 for p, _ in pages)
    text = "\n".join((f"[Page {p}]\n" if paged and p and p > 0 else "") + (t or "") for p, t in pages)
    out = []
    for group in pack(assign_paths(parse_blocks(text)), target, overlap):
        body, page = assemble(group)
        out.append((body, page if page > 0 else None, common_path(group)))
    return out


def _starts_paragraph(previous: str) -> bool:
    """Whether a line after `previous` can begin a paragraph: previous is blank, a page mark, a sentence end, another
    paragraph start or a title line (Markdown or all caps), not a line cut in the middle of a sentence."""
    line = previous.strip()
    return not line or bool(
        PAGE_LINE.match(line)
        or LINE_END.search(line)
        or PARAGRAPH.match(line)
        or MD_HEADING.match(line)
        or ALL_CAPS.match(line)
    )


def paragraphs(text: str) -> List[str]:
    """The '§ n' a parent's text starts a paragraph with, as '§ n' in reading order, each once: a line that begins with
    '§ n' and then ends or continues with a title-cased word (a heading, or a paragraph whose heading shares its line),
    after a blank line, a page mark, a sentence end, a title line or the start of the text. Small sections are merged
    into one parent and the parent's section path keeps only what they share, so this lists them. A '§ n' inside a
    sentence, a line wrapped before '§ 22 APB gilt' or '§ 24 Absatz 1', and a bare '§ 3' left by a wrap are references,
    not paragraphs. A wrapped reference that names a title-cased law ('§ 22 Prüfungsordnung der TU') after a sentence
    end still counts."""
    found: List[str] = []
    previous = ""
    for line in text.splitlines():
        m = PARAGRAPH.match(line)
        if m and _starts_paragraph(previous) and f"§ {m.group(1)}" not in found:
            found.append(f"§ {m.group(1)}")
        previous = line
    return found


def page_segments(text: str) -> List[str]:
    """A parent's text between its page marks (the marks left out), each stripped."""
    parts, start = [], 0
    for m in PAGE_MARK.finditer(text or ""):
        parts.append(text[start : m.start()])
        start = m.end()
    parts.append((text or "")[start:])
    return [p.strip() for p in parts if p.strip()]


def sentence_windows(
    text: str, size: int = 3, step: int = 2, max_chars: int = 700, chars: int = 450, overlap: int = 50
) -> List[str]:
    """Windows of `size` sentences moving by `step` (neighbours share one sentence), cut from the text itself so each
    window is a substring; a window longer than max_chars is split by the recursive splitter."""
    bounds = list(SENTENCE_BOUNDARY.finditer(text))
    starts = [0] + [m.end() for m in bounds]
    ends = [m.start() for m in bounds] + [len(text)]
    out, i, n = [], 0, len(starts)
    while i < n:
        j = min(i + size, n) - 1
        piece = text[starts[i] : ends[j]].strip()
        if len(piece) > max_chars:
            out += RecursiveCharacterTextSplitter(chunk_size=chars, chunk_overlap=overlap).split_text(piece)
        elif piece:
            out.append(piece)
        if j == n - 1:
            break
        i += step
    return out


def child_texts(parent_text: str, mode: str = "recursive", chars: int = 450, overlap: int = 50) -> List[str]:
    """The children of a parent, cut inside each page segment: 'recursive' (the recursive splitter) or 'sentences'."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chars, chunk_overlap=overlap)
    out: List[str] = []
    for segment in page_segments(parent_text):
        if mode == "sentences":
            out += sentence_windows(segment, chars=chars, overlap=overlap)
        else:
            out += splitter.split_text(segment)
    return [c for c in out if c.strip()]


def context_header(meta: dict, path: Sequence[str] = ()) -> str:
    """Context header: 'Titel · Typ · Stand <date or year> · <last three path entries>: ', each part
    only when known."""
    parts = [str(meta.get("title") or "").strip(), str(meta.get("doc_type") or "").strip()]
    stand = str(meta.get("doc_date") or "")[:10] or (str(meta["doc_year"]) if meta.get("doc_year") else "")
    if stand:
        parts.append("Stand " + stand)
    if path:
        parts.append(" > ".join(path[-3:]))
    return " · ".join(p for p in parts if p) + ": "
