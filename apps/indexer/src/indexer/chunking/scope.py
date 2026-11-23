"""A document's own scope statement for the source header.

Three cues, each only when it names a person group or an employer or area and states no exclusion: (a) the page-1
addressee, (b) a Geltungsbereich heading's first sentence or a sentence about the document itself, (c) a person group in the
page-1 heading (never the title alone: a file title often names a topic). A distribution list ("An alle Fachbereiche und
Dezernate") is no limit and gives nothing; an exclusion ("gilt nicht für …", "außer …") never prints, since a
model reading it contradicted it. The grammar below is German administrative language; the words that make a
limit are the profile's ingest.scope."""

import re
from collections.abc import Sequence

from rag_core.domain.source_meta import strip_source_text
from rag_core.settings.profile import ScopeVocabulary

from indexer.chunking.structure import heading as reads_as_heading

MAX_CHARS = 200
ADDRESSEE_LINES = 40  # page 1's first non-empty lines searched for the addressee
ADDRESSEE_MAX = 120  # a longer "An alle …" line is a paragraph, not an addressee
ADDRESSEE = re.compile(r"An\s+(alle|die|das|den)\b")
BARE_ADDRESSEE = re.compile(ADDRESSEE.pattern, re.I)
ADDRESSEE_STOP = re.compile(r"^(Sehr\s+geehrte|Liebe|Hier:|Betreff)|\d{1,2}\.\s?\d{1,2}\.\s?\d{2,4}", re.I)
SCOPE_HEADING = re.compile(
    r"(§\s*\d+\s*[.:]?\s*|\d+(\.\d+)*\.?\s*|[A-Za-z]\)\s*)?\(?((persönlicher|sachlicher|räumlicher|zeitlicher)\s+)?"
    r"(Geltungsbereich|Anwendungsbereich)\b",
    re.I,
)
SCOPE_VERB = re.compile(
    r"\b(gilt|gelten)\s+(ausschließlich\s+|nur\s+|nicht\s+|auch\s+)?für\b|\bfinde[nt]\s+Anwendung\s+(auf|für)\b", re.I
)
DOCUMENT_SUBJECT = re.compile(
    r"((Diese[rsmn]?|Die|Der|Das)\s+(Dienstanweisung|Dienstvereinbarung|Vereinbarung|Regelung|Richtlinie|Verfügung|"
    r"Anweisung|Rundschreiben|Anleitung|Ordnung|Satzung|Verordnung|Merkblatt|Hinweis|Information|Leitfaden|"
    r"Allgemeinverfügung|Gebührenordnung|Geschäftsordnung)\w*|((Die|Das|Der)\s+)?vorliegende[nrs]?\s+\w+)\s",
    re.I,
)
ENUMERATION = re.compile(r"(\(\d+\)|\d+(\.\d+)*\.?|§\s*\d+\w?\.?|[a-z]\))\s+")
RULE_LEVEL = re.compile(r"\bentsprechend\b|\bSatz\s+\d", re.I)
EXCLUSIONS = ("nicht", "kein", "keine", "keiner", "keinen", "außer", "ausgenommen", "ausnahme", "ohne")  # whole words
ABBREVIATIONS = set("Hl HI Bayer Abs Abt Nr bzw ggf ca vgl Art Ziff Dr Satz gem Str z B d h u a".split())
SENTENCE_END = re.compile(r"[.:]\s+")  # a semicolon ends no scope sentence: "gilt für …; ausgenommen …"
# a document opens with its Geltungsbereich; "§ 91 Anwendungsbereich" is a quoted law's part
SCOPE_HEADING_MAX_NUMBER = 3
WORDS = re.compile(r"[A-Za-zÄÖÜäöüß]{4}")


def _norm(line: str) -> str:
    return " ".join(line.strip("#*|_ \t-").split())


def _sentence(line: str, start: int, end: int) -> str:
    """The sentence of line around line[start:end]; a full stop after an abbreviation, a one- or two-digit number (a date,
    an ordinal) or before a lower-case word ends nothing; after a year it does."""
    ends = []
    for m in SENTENCE_END.finditer(line):
        word = re.search(r"(\w+)$", line[max(0, m.start() - 30) : m.start()])
        after = line[m.end() : m.end() + 1]
        if (
            m.group().startswith(".")
            and word
            and (
                word.group(1) in ABBREVIATIONS
                or (word.group(1).isdigit() and len(word.group(1)) <= 2)
                or after.islower()
            )
        ):
            continue
        ends.append(m)
    first = max([m.end() for m in ends if m.end() <= start], default=0)
    last = min([m.start() + 1 for m in ends if m.start() >= end], default=len(line))
    return line[first:last].strip()


def _unnumbered(text: str) -> str:
    m = ENUMERATION.match(text)
    return text[m.end() :] if m else text


def _has(stem: str, folded: str, whole: bool = False) -> bool:
    """A stem at a word start ("beamt" in "Beamtenrecht", never in "nichtbeamteter"); whole: the whole word ("außer", never
    "außerschulisch")."""
    return re.search(r"(?<![a-zäöüß])" + re.escape(stem) + (r"(?![a-zäöüß])" if whole else ""), folded) is not None


def _excludes(folded: str) -> bool:
    return any(_has(word, folded, whole=True) for word in EXCLUSIONS)


def _limits(text: str, vocabulary: ScopeVocabulary) -> bool:
    folded = text.lower()
    stems = [s for group in vocabulary.groups.values() for s in group] + list(vocabulary.areas)
    return not _excludes(folded) and any(_has(s, folded) for s in stems)


def _opens_document(heading: re.Match[str]) -> bool:
    number = re.search(r"\d+", heading.group(1) or "")
    return number is None or int(number.group()) <= SCOPE_HEADING_MAX_NUMBER


def _addressee(lines: Sequence[str], vocabulary: ScopeVocabulary) -> str:
    """The first page-1 addressee that names a limit, with its continuation lines (a lower-case or "(" opening, an open
    bracket, or a bare "An alle")."""
    lines = list(lines[:ADDRESSEE_LINES])
    for i, line in enumerate(lines):
        if len(line) > ADDRESSEE_MAX or not ADDRESSEE.match(line):
            continue
        text = line
        for nxt in lines[i + 1 : i + 5]:
            open_bracket = text.count("(") > text.count(")")
            if ADDRESSEE_STOP.search(nxt) or not (
                open_bracket or BARE_ADDRESSEE.fullmatch(text) or nxt[0].islower() or nxt[0] == "("
            ):
                break
            text += " " + nxt
        if _limits(text, vocabulary):
            return text
    return ""


def _statement(pages: Sequence[Sequence[tuple[str, str]]], vocabulary: ScopeVocabulary) -> str:
    """The first Geltungsbereich heading's first sentence or sentence about the document that names a limit."""
    for lines in pages:
        for i, (raw, line) in enumerate(lines):
            if raw.lstrip().startswith("|") or "...." in line:
                continue  # a table row or a contents line
            heading = SCOPE_HEADING.match(line)
            if heading and _opens_document(heading):
                rest, candidate = line[heading.end() :].lstrip(" :.)"), ""
                if len(rest) > 20:
                    candidate = _sentence(rest, 0, 1)
                elif not rest and len(line) <= 60 and i + 1 < len(lines):  # a short rest is a form label
                    nxt = lines[i + 1][1]
                    following = [nxt]  # a sentence runs across line breaks, never into the next heading
                    for _, text in lines[i + 2 : i + 4]:
                        if reads_as_heading(text):
                            break
                        following.append(text)
                    if not (nxt.endswith(":") or (":" in nxt and len(nxt) < 25)):  # a form field
                        candidate = _sentence(" ".join(following), 0, 1)
                if candidate and _limits(candidate, vocabulary):
                    return _unnumbered(candidate)
            for verb in SCOPE_VERB.finditer(line):
                sentence = _sentence(line, verb.start(), verb.end())
                subject = _unnumbered(sentence)
                if DOCUMENT_SUBJECT.match(subject) and not RULE_LEVEL.search(subject) and _limits(subject, vocabulary):
                    return subject
    return ""


def _page1_heading(lines: Sequence[str]) -> str:
    """The first of page 1's first eight lines that reads as a title (indexer.extract.documents.resolve_title's test)."""
    for line in lines[:8]:
        if ":" in line:
            continue  # a "label: value" row of a form, not a heading
        if 8 <= len(line) <= 120 and WORDS.search(line) and not line.lower().startswith("seite"):
            return line
    return ""


def _assemble(cues: Sequence[str], out: str = "") -> str:
    """out plus each whole cue while the line stays within MAX_CHARS; a first cue beyond it is cut at a word with "…"."""
    for cue in cues:
        joined = out + "; " + cue if out else cue
        if len(joined) <= MAX_CHARS:
            out = joined
        elif not out:
            cut = cue.rfind(" ", 0, MAX_CHARS - 1)
            out = cue[: cut if cut > 0 else MAX_CHARS - 1].rstrip() + "…"
    return out


def document_scope(title: str, pages: Sequence[tuple[int | None, str]], vocabulary: ScopeVocabulary) -> str:
    """The document's own scope statement: the addressee, then the statement, then the labels of the person groups the
    page-1 heading names and the printed text does not; joined with "; " within MAX_CHARS; "" when no cue names a limit."""
    split = []
    for _, text in pages:
        raw_lines = [raw for raw in strip_source_text(text).splitlines() if _norm(raw)]
        split.append([(raw, _norm(raw)) for raw in raw_lines])
    if not split:
        return ""
    page1 = [line for _, line in split[0]]
    folded_title = (title or "").lower()
    compilation = any(_has(stem, folded_title) for stem in vocabulary.compilations)
    out = _assemble(
        [c for c in (_addressee(page1, vocabulary), "" if compilation else _statement(split, vocabulary)) if c]
    )
    named, printed = _page1_heading(page1).lower(), out.lower()
    if _excludes(named):
        named = ""  # "Hinweise für Beamte (nicht für Anwärter)" must not print Anwärter
    labels = [
        label
        for label, stems in vocabulary.groups.items()
        if any(_has(s, named) for s in stems) and not any(_has(s, printed) for s in stems)
    ]
    return _assemble(labels, out)
