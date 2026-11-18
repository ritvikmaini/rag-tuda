"""Status groups (Beamte, Anwärter, Tarifbeschäftigte, Auszubildende, ...) read from a profile's group
table: what a question is about (its own group, a comparison, a subject, nothing) and which group a source document is for.
The code is generic over the table; the German lead-ins of the question kinds are the only fixed words."""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol
from urllib.parse import unquote, urlparse


class Group(Protocol):
    """One row of the profile's group table."""

    @property
    def label(self) -> str: ...  # the group's name in the answer texts
    @property
    def side(self) -> str: ...  # the groups of one side are contrasted with those of the other sides
    @property
    def part_of(self) -> str | None: ...  # the parent group of a subgroup
    @property
    def ask(self) -> Sequence[str]: ...  # question words: at a word start or the end of a compound
    @property
    def ask_self(self) -> Sequence[str]: ...  # question words that count only when the asker names itself
    @property
    def source(self) -> Sequence[str]: ...  # regular expressions, matched at a word start in a source's text


Groups = Mapping[str, Group]

SELF, COMPARE, SUBJECT, GENERAL = "self", "compare", "subject", "general"
LETTERS = "a-zäöüß"
TOKEN = re.compile(rf"[{LETTERS}]+(?:-[{LETTERS}]+)*")
ARTICLES = frozenset({"ein", "eine", "einer"})
NEGATIONS = frozenset({"nicht", "kein", "keine", "keiner", "keinen", "keinem", "keines"})
FOLLOW_UP_WORDS = 8  # a follow-up that counts as the asker's own group is at most this long
FOLLOW_UPS = ("und ", "was ist mit", "wie sieht es", "wie ist es", "wie ist das")
COMPARE_WORDS = ("unterschied", "unterscheid", "im vergleich", "gegenüber")
COMPARE_PATTERNS = tuple(
    re.compile(p)
    for p in (
        rf"\bauch\s+für\s+(?:(?:die|den|alle)\s+)?([{LETTERS}-]+)",
        rf"\bgilt\s+(?:das|dies|die|der)\b[^?.!;]*?\bfür\s+(?:(?:die|den|alle)\s+)?([{LETTERS}-]+)",
    )
)
SOWOHL_ALS_AUCH = re.compile(r"\bsowohl\b.*\bals\s+auch\b")
CONTACT_WORDS = (
    "an wen",
    "wer ist zuständig",
    "wer bearbeitet",
    "ansprechpartner",
    "wo finde ich",
    "telefon",
    "e-mail",
    "email",
    "kontakt",
)


@dataclass(frozen=True)
class Scope:
    """What a question says about status groups: its kind (self, compare, subject or general), the groups the answer is
    about (self and compare) or the question names (subject), and whether it asks for a contact or a place."""

    kind: str
    groups: frozenset[str]
    contact: bool


@lru_cache(maxsize=None)
def _ask_pattern(word: str) -> re.Pattern[str]:
    # the word after any compound start, then an ending of at most five letters ("Beamtinnen", "Verwaltungsbeamtin"),
    # then optionally another hyphenated word; "Beamtenrecht" and "Tarifrunde" are longer than an ending
    return re.compile(rf"[{LETTERS}-]*{re.escape(word)}[{LETTERS}]{{0,5}}(?:-[{LETTERS}-]*)?")


@lru_cache(maxsize=None)
def _source_pattern(word: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![{LETTERS}])(?:{word})")


def _token_groups(token: str, groups: Groups, with_self: bool) -> set[str]:
    return {
        key
        for key, group in groups.items()
        if any(_ask_pattern(w).fullmatch(token) for w in (*group.ask, *(group.ask_self if with_self else ())))
    }


def _negated(tokens: list[str], i: int) -> bool:
    return tokens[i].startswith("nicht") or any(t in NEGATIONS for t in tokens[max(0, i - 2) : i])


def _self_lead(tokens: list[str], i: int) -> bool:
    """True when the token follows "als", "bin" or "wir sind" (and an article)."""
    j = i - 1
    if j >= 0 and tokens[j] in ARTICLES:
        j -= 1
    return j >= 0 and (tokens[j] in ("als", "bin") or (tokens[j] == "sind" and j >= 1 and tokens[j - 1] == "wir"))


def _sides(groups: Groups, keys: Iterable[str]) -> set[str]:
    return {groups[k].side for k in keys}


@dataclass(frozen=True)
class _Read:
    kind: str
    groups: frozenset[str]  # the groups of the kind
    mentioned: frozenset[str]  # every group the text names (not negated, no self-only word)
    contact: bool


def _read(text: str, groups: Groups) -> _Read:
    low = text.lower()
    tokens = TOKEN.findall(low)
    mentioned: set[str] = set()
    own: set[str] = set()
    for i, token in enumerate(tokens):
        if _negated(tokens, i):
            continue
        mentioned |= _token_groups(token, groups, False)
        if _self_lead(tokens, i):
            own |= _token_groups(token, groups, True)
    named: set[str] = set()  # the groups after "auch für" or "gilt das ... für"
    for pattern in COMPARE_PATTERNS:
        for found in pattern.findall(low):
            named |= _token_groups(found, groups, True)
    keyword = any(w in low for w in COMPARE_WORDS) or SOWOHL_ALS_AUCH.search(low) is not None
    everything = frozenset(mentioned | own | named)
    contact = any(w in low for w in CONTACT_WORDS)
    if named or (keyword and everything) or len(_sides(groups, everything)) > 1:
        return _Read(COMPARE, everything, everything, contact)
    if own:
        return _Read(SELF, frozenset(own), everything, contact)
    return _Read(SUBJECT if mentioned else GENERAL, frozenset(mentioned), everything, contact)


def question_scope(groups: Groups, question: str, standalone: str = "") -> Scope:
    """The kind of a question, read from the user's text and the standalone question:
    - self: the asker names its own group ("als Beamtin", "ich bin ...", a short follow-up "Und für ...?");
    - compare: "auch für <group>", "Unterschied", two sides named;
    - subject: a group is named otherwise ("Beurteilungen für Beamte");
    - general: no group, or every mention negated.
    The contact flag (a question for a contact or a place) comes with any kind."""
    if not groups:
        return Scope(GENERAL, frozenset(), False)
    user, standing = _read(question, groups), _read(standalone, groups)
    contact = user.contact or standing.contact
    if user.kind == COMPARE:
        return Scope(COMPARE, user.groups | standing.mentioned, contact)
    if len(question.split()) <= FOLLOW_UP_WORDS and question.lower().lstrip().startswith(FOLLOW_UPS):
        for read in (user, standing):
            if len(read.mentioned) == 1:
                return Scope(SELF, read.mentioned, contact)
    reads = [r for r in (user, standing) if r.kind != GENERAL]
    mentioned = frozenset(k for r in reads for k in r.mentioned)
    if any(r.kind == COMPARE for r in reads):
        return Scope(COMPARE, mentioned, contact)
    own = frozenset(k for r in reads if r.kind == SELF for k in r.groups)
    if len(_sides(groups, mentioned)) > 1:
        return Scope(COMPARE, mentioned, contact)
    if own:
        return Scope(SELF, own, contact)
    return Scope(SUBJECT if mentioned else GENERAL, mentioned, contact)


def named_sides(text: str, groups: Groups) -> set[str]:
    """The sides whose groups a text (an answer) names."""
    return {groups[k].side for t in TOKEN.findall(text.lower()) for k in _token_groups(t, groups, False)}


def _source_named(text: str, groups: Groups) -> set[str]:
    low = text.lower()
    return {k for k, g in groups.items() if any(_source_pattern(w).search(low) for w in g.source)}


def source_group(
    metadata: Mapping[str, Any] | None, groups: Groups, overrides: Mapping[str, str] | None = None
) -> str | None:
    """The group a source document is for, None for a general one. The inputs, most specific first: the profile's override
    for a title substring ("alle" gives None), the document's own scope statement, the section path, the title, the file's
    name. The first input naming one group decides; an input naming groups of two sides gives None and ends the search."""
    m = metadata or {}
    title = str(m.get("title") or "")
    for part, group in (overrides or {}).items():
        if part.lower() in title.lower():
            return group if group in groups else None
    url = str(m.get("download_url") or m.get("url") or "")
    # the file's own name only: the folders above it are categories
    slug = unquote(urlparse(url).path).rstrip("/").rsplit("/", 1)[-1].replace("-", " ").replace("_", " ")
    path = " ".join(str(m.get(k) or "") for k in ("section_path", "title_path"))
    for text in (str(m.get("scope") or ""), path, title, slug):
        named = _source_named(text, groups)
        if len(_sides(groups, named)) > 1:
            return None
        top = [k for k in named if groups[k].part_of not in named]  # a group and its subgroup count as the group
        if len(top) == 1:
            return top[0]
    return None


def label_of_side(groups: Groups, side: str) -> str:
    """The label of a side's top group."""
    return next((g.label for g in groups.values() if g.side == side and not g.part_of), side)


def join_labels(labels: Sequence[str]) -> str:
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " sowie " + labels[-1]


def compare_labels(groups: Groups, asked: Iterable[str]) -> str:
    """The labels of the asked groups, plus the top group of every other side when the asked ones cover one side only."""
    keys = [k for k in groups if k in set(asked)]
    covered = _sides(groups, keys)
    labels = [groups[k].label for k in keys]
    if len(covered) < 2:
        labels += [label_of_side(groups, s) for s in dict.fromkeys(g.side for g in groups.values()) if s not in covered]
    return join_labels(labels)


@dataclass(frozen=True)
class Note:
    """A closing note: kind "other" (the asked group has no documents of its own) or "one" (the documents limit the rule to
    one group), the label of the group the documents are for, and the label of the asked groups or of the other sides."""

    kind: str
    group: str
    other: str


def note_for(groups: Groups, scope: Scope, cited: Sequence[str | None], answer: str) -> Note | None:
    """The note closing an answer, or None. The cited sources' groups (None = a general source) must all be for one side:
    - the asker named a group of another side: the answer rests on documents for the side it is not on;
    - a general question, or an asker of that side (not a subgroup that cites only its parent): the documents limit the
      rule to that side, unless the answer already names the other sides.
    Never for a contact question, a comparison or a question that only names a group as its subject; the "one" note never
    for a subgroup's documents or asker."""
    if scope.contact or scope.kind not in (SELF, GENERAL) or not cited or None in cited:
        return None
    keys = {c for c in cited if c}
    sides = _sides(groups, keys)
    if len(sides) != 1:
        return None
    side = next(iter(sides))
    # the documents' group: the subgroup when every cited source is that subgroup's, else the side's top group
    group = groups[next(iter(keys))].label if len(keys) == 1 else label_of_side(groups, side)
    away = [k for k in groups if k in scope.groups and groups[k].side != side]
    if scope.kind == SELF and away:
        return Note("other", group, join_labels([groups[k].label for k in away]))
    if scope.kind == SELF and any(groups[k].part_of in keys for k in scope.groups) and len(keys) == 1:
        return None
    # the limit is told between the top groups only: documents all of a subgroup, or a subgroup asker, get no "one" note
    if all(groups[k].part_of for k in keys) or any(groups[k].part_of for k in scope.groups):
        return None
    others = [s for s in dict.fromkeys(g.side for g in groups.values()) if s != side]
    if not others or named_sides(answer, groups) & set(others):
        return None
    return Note("one", group, join_labels([label_of_side(groups, s) for s in others]))
