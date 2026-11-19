"""Status groups (bachelor, master, promotion, the programmes) read from a profile's group table: what a question is about
(its own group, a comparison, a subject, nothing) and which group, programme and PO versions a source document is for, read
from the typed columns `degree`, `programme` and `po_version` of its metadata. From both the answer prompt gets its group
lines: a comparison, sources that differ by group or version, and the side-by-side table of versions. The code is generic
over the table; the German and English lead-ins of the question kinds are the only fixed words."""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol


class Group(Protocol):
    """One row of the profile's group table."""

    @property
    def label(self) -> str: ...  # the group's name in the answer texts
    @property
    def label_en(self) -> str | None: ...  # its name in English answer texts; None = the label
    @property
    def side(self) -> str: ...  # the groups of one side are contrasted with those of the other sides
    @property
    def part_of(self) -> str | None: ...  # the parent group of a subgroup
    @property
    def ask(self) -> Sequence[str]: ...  # question words: at a word start or the end of a compound
    @property
    def ask_self(self) -> Sequence[str]: ...  # question words that count only when the asker names itself


Groups = Mapping[str, Group]

SELF, COMPARE, SUBJECT, GENERAL = "self", "compare", "subject", "general"
LETTERS = "a-zäöüß"
TOKEN = re.compile(rf"[{LETTERS}]+(?:-[{LETTERS}]+)*")
ARTICLES = frozenset({"ein", "eine", "einer", "a", "an"})
NEGATIONS = frozenset(
    {"nicht", "kein", "keine", "keiner", "keinen", "keinem", "keines", "not", "no", "never", "without", "ohne"}
)
NEGATED_PREFIXES = ("nicht", "non-")
# the words before a group word that name the asker ("als Master", "I'm a PhD", "we are ..."); "i'm" is read as i, m
SELF_LEADS = (("als",), ("bin",), ("wir", "sind"), ("as",), ("i", "am"), ("i", "m"), ("we", "are"))
FOLLOW_UP_WORDS = 8  # a follow-up that counts as the asker's own group is at most this long
FOLLOW_UPS = (
    "und ",
    "was ist mit",
    "wie sieht es",
    "wie ist es",
    "wie ist das",
    "and ",
    "what about",
    "how about",
    "how is it",
    "what is it like",
)
COMPARE_WORDS = (
    "unterschied",
    "unterscheid",
    "im vergleich",
    "gegenüber",
    "differ",
    "compared to",
    "compared with",
    "versus",
)
COMPARE_PATTERNS = tuple(
    re.compile(p)
    for p in (
        rf"\bauch\s+für\s+(?:(?:die|den|alle)\s+)?([{LETTERS}-]+)",
        rf"\bgilt\s+(?:das|dies|die|der)\b[^?.!;]*?\bfür\s+(?:(?:die|den|alle)\s+)?([{LETTERS}-]+)",
        rf"\balso\s+(?:for|valid\s+for|applies?\s+to)\s+(?:(?:the|all)\s+)?([{LETTERS}-]+)",
        rf"\b(?:does|do|is|are)\s+(?:this|that|it|these|those)\b[^?.!;]*?\b(?:apply|applies|hold|holds|valid)\s+"
        rf"(?:to|for)\s+(?:(?:the|all)\s+)?([{LETTERS}-]+)",
    )
)
SOWOHL_ALS_AUCH = re.compile(r"\bsowohl\b.*\bals\s+auch\b|\bboth\b.*\band\b")
DEGREE_ABBREVIATION = re.compile(r"\b([bm])\.\s?sc\b\.?")  # "B.Sc." is read as bsc
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
    "who is responsible",
    "who handles",
    "who do i",
    "who should i",
    "where can i find",
    "contact",
    "phone",
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
    return tokens[i].startswith(NEGATED_PREFIXES) or any(t in NEGATIONS for t in tokens[max(0, i - 2) : i])


def _self_lead(tokens: list[str], i: int) -> bool:
    """True when the token follows a lead of SELF_LEADS ("als", "ich bin", "wir sind", "as", "I am", "we are"), and an
    article."""
    j = i - 1
    if j >= 0 and tokens[j] in ARTICLES:
        j -= 1
    return any(j + 1 >= len(lead) and tuple(tokens[j + 1 - len(lead) : j + 1]) == lead for lead in SELF_LEADS)


def _generic(token_groups: set[str], groups: Groups) -> bool:
    """True when every group of the token is the parent of a subgroup ("student", "studierende")."""
    return bool(token_groups) and all(any(g.part_of == key for g in groups.values()) for key in token_groups)


def _specific(keys: Iterable[str], groups: Groups) -> frozenset[str]:
    """The groups without those whose subgroup is also in the set: "Master-Studierende" names master, not the students."""
    found = set(keys)
    return frozenset(k for k in found if not any(groups[j].part_of == k for j in found))


def _sides(groups: Groups, keys: Iterable[str]) -> set[str]:
    return {groups[k].side for k in keys}


@dataclass(frozen=True)
class _Read:
    kind: str
    groups: frozenset[str]  # the groups of the kind
    mentioned: frozenset[str]  # every group the text names (not negated, no self-only word)
    contact: bool


def _read(text: str, groups: Groups) -> _Read:
    low = DEGREE_ABBREVIATION.sub(r"\1sc", text.lower())
    tokens = TOKEN.findall(low)
    heard: set[str] = set()  # every group a token names
    self_named: set[str] = set()  # the groups after a self lead
    previous: set[str] = set()  # the groups the token before names
    for i, token in enumerate(tokens):
        found = _token_groups(token, groups, False)
        last, previous = previous, found
        if _negated(tokens, i) or (_generic(found, groups) and last):  # "PhD student": the head noun adds no group
            continue
        heard |= found
        if _self_lead(tokens, i):
            self_named |= _token_groups(token, groups, True)
    after_for: set[str] = set()  # the groups after "auch für" or "gilt das ... für"
    for pattern in COMPARE_PATTERNS:
        for word in pattern.findall(low):
            after_for |= _token_groups(word, groups, True)
    keyword = any(w in low for w in COMPARE_WORDS) or SOWOHL_ALS_AUCH.search(low) is not None
    mentioned, named = _specific(heard, groups), _specific(after_for, groups)
    # "Ich bin Student im Master": the generic group the asker names yields to the subgroup the text names
    own = _specific(self_named | {m for m in mentioned if groups[m].part_of in self_named}, groups)
    everything = _specific(mentioned | own | named, groups)
    contact = any(w in low for w in CONTACT_WORDS)
    if named or (keyword and everything) or len(_sides(groups, everything)) > 1:
        return _Read(COMPARE, everything, everything, contact)
    if own:
        return _Read(SELF, own, everything, contact)
    return _Read(SUBJECT if mentioned else GENERAL, mentioned, everything, contact)


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


def _column(metadata: Mapping[str, Any] | None, name: str) -> list[Any]:
    """The values of a typed column in a source's metadata: a list, a bare value or nothing (NULL: applies to all)."""
    value: Any = (metadata or {}).get(name)
    items: list[Any] = [*value] if isinstance(value, (list, tuple)) else [value]  # pyright: ignore[reportUnknownVariableType]
    return [v for v in items if v not in (None, "")]


def _single_group(metadata: Mapping[str, Any] | None, name: str, groups: Groups) -> str | None:
    values = _column(metadata, name)
    return values[0] if len(values) == 1 and values[0] in groups else None


def source_group(metadata: Mapping[str, Any] | None, groups: Groups) -> str | None:
    """The group a source document is for: the group named by its one `degree`; None for a general source (NULL) and for
    one valid for several degrees."""
    return _single_group(metadata, "degree", groups)


def source_programme(metadata: Mapping[str, Any] | None, groups: Groups) -> str | None:
    """The programme group a source document is for: its one `programme` that the table has a group for, else None."""
    return _single_group(metadata, "programme", groups)


def source_versions(metadata: Mapping[str, Any] | None) -> tuple[int, ...]:
    """The PO years a source document is for (its `po_version`), oldest first; none for a source not bound to a version."""
    return tuple(sorted(v for v in _column(metadata, "po_version") if isinstance(v, int) and not isinstance(v, bool)))


Metadata = Sequence[Mapping[str, Any] | None]


def versions_of(sources: Metadata) -> list[int]:
    """The distinct PO years of the sources, newest first."""
    return sorted({v for m in sources for v in source_versions(m)}, reverse=True)


def sources_differ(sources: Metadata, groups: Groups) -> bool:
    """True when the sources are for more than one degree group or more than one programme. A general source (NULL) and one
    valid for several degrees name no group, so a programme's source beside a plain one of its degree does not differ."""
    degrees = {g for m in sources if (g := source_group(m, groups))}
    programmes = {g for m in sources if (g := source_programme(m, groups))}
    return len(degrees) > 1 or len(programmes) > 1


def label_of_side(groups: Groups, side: str) -> str:
    """The label of a side's top group."""
    return next((g.label for g in groups.values() if g.side == side and not g.part_of), side)


def _asked_sides(groups: Groups) -> list[str]:
    """The sides a question can name (a group of the side has question words); the programmes are read from sources only."""
    return list(dict.fromkeys(g.side for g in groups.values() if g.ask or g.ask_self))


def join_labels(labels: Sequence[str], conj: str = "sowie") -> str:
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + f" {conj} " + labels[-1]


def compare_labels(groups: Groups, asked: Iterable[str], degrees: Iterable[str] = (), english: bool = False) -> str:
    """The labels of the groups an answer compares, "" when fewer than two remain. A group the question's hard filter
    (`degrees`, the filter's degree groups; empty = none) excluded is never named: retrieval holds nothing for it. A single
    asked group is contrasted with its siblings (the other subgroups of its parent) or, for a top group, with the top group
    of every other side a question can name; a contrast the filter excluded is dropped."""
    kept = set(degrees)

    def allowed(key: str) -> bool:
        return not kept or key in kept or any(groups[d].part_of == key for d in kept if d in groups)

    keys = [k for k in groups if k in set(asked) and allowed(k)]
    if len(keys) == 1:
        group = groups[keys[0]]
        if group.part_of:
            contrast = [k for k, g in groups.items() if g.part_of == group.part_of and k != keys[0]]
        else:
            tops: dict[str, str] = {}
            for k, g in groups.items():
                if not g.part_of and g.side in _asked_sides(groups):
                    tops.setdefault(g.side, k)  # the first top group of the side
            contrast = [k for side, k in tops.items() if side != group.side]
        keys += [k for k in contrast if allowed(k)]
    if len(keys) < 2:
        return ""
    names = [(groups[k].label_en if english else None) or groups[k].label for k in keys]
    return join_labels(names, "and" if english else "sowie")


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
    others = [s for s in _asked_sides(groups) if s != side]
    if not others or named_sides(answer, groups) & set(others):
        return None
    return Note("one", group, join_labels([label_of_side(groups, s) for s in others]))


class Texts(Protocol):
    """The wording of the prompt lines (the profile's `answer` messages); an empty text leaves its line out."""

    @property
    def scope_compare(self) -> str: ...  # {groups}: the labels of the groups compared
    @property
    def scope_differ(self) -> str: ...  # the best sources are for different groups or programmes
    @property
    def scope_versions(self) -> str: ...  # {versions}: the PO years, newest first
    @property
    def version_table(self) -> str: ...  # the side-by-side format for an answer citing more than one version


def scope_lines(
    groups: Groups,
    scope: Scope,
    asked_versions: Iterable[int],
    top: Metadata,
    shown: Metadata,
    texts: Texts,
    *,
    compare: bool,
    differ: bool,
    version_table: bool,
    plain: bool = True,
    degrees: Iterable[str] = (),
    english: bool = False,
) -> list[str]:
    """The prompt lines for a question and the metadata of its sources, in this order:
    - compare (switch `compare`): a comparison of groups names the groups still in the question's filter (`degrees`);
    - differ (switch `differ`, plain-text answers): a general question (no contact) whose best sources (`top`) are for
      different groups or programmes asks to tell the groups apart;
    - versions (either switch, any answer shape): PO versions the question names (two or more) or, else, the best sources
      carry (more than one) ask to name each version, beside any group line;
    - version table (switch `version_table`, plain-text answers): sources in the context (`shown`) of more than one PO
      version ask for the versions side by side, one row each.
    The switches are the caller's decision; a text the profile leaves empty drops its line."""
    lines: list[str] = []
    if compare and scope.kind == COMPARE and texts.scope_compare:
        if labels := compare_labels(groups, scope.groups, degrees, english):
            lines.append(texts.scope_compare.format(groups=labels))
    if differ and plain and not scope.contact and scope.kind == GENERAL and texts.scope_differ:
        if sources_differ(top, groups):
            lines.append(texts.scope_differ)
    asked = sorted(set(asked_versions), reverse=True)
    years = asked if compare and len(asked) > 1 else versions_of(top)
    if (compare or differ) and len(years) > 1 and texts.scope_versions:
        names = join_labels([f"PO {y}" for y in years], "and" if english else "sowie")
        lines.append(texts.scope_versions.format(versions=names))
    if version_table and plain and len(versions_of(shown)) > 1 and texts.version_table:
        lines.append(texts.version_table)
    return lines
