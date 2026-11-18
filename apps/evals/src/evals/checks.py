"""The automatic checks of a probe turn. The shape checks are a copy of rag_core/domain/answer_shapes.py's, kept equal by
tests/contracts/test_answer_shapes.py; evals never imports the assistant. Every run is an internal run: a link to an internal document
is not a leak and there is no internal-source check."""

import re
from typing import Any

LEAK_PATTERN = re.compile(r"PdfPipelineOptions|Image not available|\[Page \d+\]|Dokumentenauszüge")
EMOJI_PATTERN = re.compile("[\U0001f300-\U0001faff\U0001f1e6-\U0001f1ff☀-➿⭐⭕]")
# Scripts that never belong in a German (or English) answer: Cyrillic, Hebrew, Arabic, CJK, Hangul
NON_LATIN_PATTERN = re.compile("[\u0400-\u04ff\u0590-\u06ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uf900-\ufaff]")
INFORMAL_PATTERN = re.compile(r"\b(du|dich|dir|dein|deine|deinen|deinem|deiner|deines)\b", re.IGNORECASE)
CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")

# Response formats v1: the same checks as rag_core/domain/answer_shapes.py, kept equal by
# tests/contracts/test_answer_shapes.py; table_cited is the evals' own.
FMT_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
FMT_SEPARATOR = re.compile(r"^[ \t]*\|[ \t:|-]*-[ \t:|-]*\|[ \t]*$")
FMT_TASK = re.compile(r"^[ \t]*[-*][ \t]+\[[ xX]\][ \t]+\S")
FMT_STEP = re.compile(r"^[ \t]*\d{1,2}[.)][ \t]+\S")
FMT_ITEM = re.compile(r"^[ \t]*[-*][ \t]+\S")
FMT_MINDMAP = re.compile(r"```mindmap[ \t]*\n(.*?)(?:\n[ \t]*```|\Z)", re.S)
FMT_CITE = re.compile(r"\[\d{1,2}(?:\s*,\s*\d{1,2})*\]")
SHAPES = ("text", "table", "checklist", "steps", "mindmap")


def _table_rows(text):
    rows, in_table = [], False
    for line in str(text or "").splitlines():
        if FMT_SEPARATOR.match(line):
            in_table = True
        elif in_table and FMT_ROW.match(line):
            rows.append(line)
        elif not FMT_ROW.match(line):
            in_table = False
    return rows


def detect_shape(text):
    """mindmap, table, checklist (≥ 2 items), steps (≥ 2 numbered lines) or text."""
    if FMT_MINDMAP.search(str(text or "")):
        return "mindmap"
    if _table_rows(text):
        return "table"
    lines = str(text or "").splitlines()
    if sum(1 for line in lines if FMT_TASK.match(line)) >= 2:
        return "checklist"
    if sum(1 for line in lines if FMT_STEP.match(line)) >= 2:
        return "steps"
    return "text"


def structure_lines(text, shape):
    if shape == "mindmap":
        found = FMT_MINDMAP.search(str(text or ""))
        return [line for line in (found.group(1) if found else "").splitlines() if FMT_ITEM.match(line)]
    if shape == "table":
        return _table_rows(text)
    if shape == "checklist":
        return [line for line in str(text or "").splitlines() if FMT_TASK.match(line)]
    if shape == "steps":
        return [line for line in str(text or "").splitlines() if FMT_STEP.match(line)]
    return [text] if str(text or "").strip() else []


def cited_share(text, shape):
    lines = structure_lines(text, shape)
    return sum(1 for line in lines if FMT_CITE.search(line)), len(lines)


def _has_text(cell):
    return bool(re.search(r"\w", FMT_CITE.sub("", cell)))


def table_cited(text):
    """cited_share of a table read with its source row (the evals' own; cited_share stays the assistant's copy): a row
    counts when it carries a citation, or when every cell after its first that holds text lies in a column a source row
    cites. A source row holds only citations after its first cell (the model labels it "Quelle", once "Link"); a source
    column puts a citation in every row and needs no rule of its own."""
    rows = [row.strip().strip("|").split("|") for row in _table_rows(text)]
    # ponytail: one set of source columns for all tables of an answer (no stored answer holds two); split per table if one does
    cols = {
        i for r in rows if len(r) > 1 and not any(map(_has_text, r[1:])) for i, c in enumerate(r) if FMT_CITE.search(c)
    }
    covered = [
        r
        for r in rows
        if any(FMT_CITE.search(c) for c in r)
        or (cols and all(i in cols for i, c in enumerate(r) if i and _has_text(c)))
    ]
    return len(covered), len(rows)


def url_matches(url, fragment, mode=None):
    """source_url_any check: the fragment anywhere in the URL (default) or, with mode 'page', the page itself — the URL
    without its '#…' anchor (kept when the fragment has one) and trailing '/' ends with the fragment, so a section path
    does not pass on its sub-pages (G31, G32, G37, G38)."""
    if mode != "page":
        return fragment in url
    base = url if "#" in fragment else url.split("#", 1)[0]
    return base.rstrip("/").endswith(fragment.rstrip("/"))


# ponytail: wording rules, not understanding. Ceilings: paraphrases and synonyms stay fails; a negation in front of the whole
# match is not seen (the verbatim test has the same gap); two-digit years (13.07.17) are not expanded; units other than % and
# € (ha, Std.) are not mapped; one extra word is the whole gap (GAP = 1, chosen by measuring 14,800 stored turns: 3 adds one);
# a number with a decimal comma is one token, never stemmed ("2,50" is not "50"); "ein…" counts as the number 1.
MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("januar", "jan"),
            ("februar", "feb"),
            ("märz", "maerz", "mär"),
            ("april", "apr"),
            ("mai",),
            ("juni", "jun"),
            ("juli", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("oktober", "okt"),
            ("november", "nov"),
            ("dezember", "dez"),
        ],
        1,
    )
    for m in names
}
MONTH_DATE = re.compile(
    r"\b(\d{1,2})\.\s*(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b\.?,?(?:\s+(\d{4}))?", re.IGNORECASE
)
NUMBER_WORDS = dict(zip("zwei drei vier fünf sechs sieben acht neun zehn elf zwölf".split(), range(2, 13)))
NUMBER_WORDS.update(dict.fromkeys("ein eine einen einem einer eines".split(), 1))
PHONE = re.compile(r"(?<![\d.,:])0\d{2,}(?:[ \t]*[/–-]?[ \t]*\d+)+")
LITERAL = re.compile(r"://|@|[a-z]\.[a-z]{2}")  # URLs, mail addresses and domains match verbatim only
TOKEN = re.compile(r"\d+(?:,\d+)+|[^\W_]+|[§%€]")
CENTS = re.compile(r"(\d),(?:00|-)(?!\d)")  # 60,00 Euro = 60,- Euro = 60 Euro
SYMBOLS = {"%": "prozent", "€": "euro", "eur": "euro"}
PERCENT = re.compile(r"\bv\.\s?h\.|\bvom hundert\b", re.IGNORECASE)
THOUSANDS = re.compile(r"(?<![\d.])(\d{1,3})\.(\d{3})(?![\d.])")
SKIP = set("der die das dem den des ihr ihre ihrem ihren ihrer ihres sein seine seinem seinen seiner seines".split())
GAP_NEGATION = re.compile(r"^(nicht|kein\w*|nie|niemal\w*)$")
SUFFIXES = ("ern", "en", "em", "er", "es", "e", "n", "s")
GAP = 1
LINK = re.compile(r"https?://[^\s)\]>\"'„“]+")


def _stem(word):
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokens(text):
    """The comparable words of a text: dates, numbers, phone numbers, currency and per cent in one spelling, articles
    dropped, inflections stemmed."""
    text = FMT_CITE.sub(" ", text.replace("**", ""))
    text = MONTH_DATE.sub(lambda m: f"{m[1]}.{MONTHS[m[2].lower()]}.{m[3] or ''}", text)
    text = PERCENT.sub(" % ", THOUSANDS.sub(r"\1\2", CENTS.sub(r"\1", text)))
    text = PHONE.sub(lambda m: re.sub(r"\D", "", m[0]), text)
    out = []
    for raw in TOKEN.findall(text.casefold()):
        if raw.isdigit():
            if raw != "00":  # minutes and cents: 15.00 Uhr = 15 Uhr, 60,00 Euro = 60 Euro
                out.append(raw.lstrip("0") or "0")
            continue
        if "," in raw:
            out.append(raw)
            continue
        if raw in SKIP:
            continue
        raw = SYMBOLS.get(raw, str(NUMBER_WORDS.get(raw, raw)))
        out.append(raw if raw.isdigit() else _stem(raw))
    return out


def _gap_ok(word):
    return not GAP_NEGATION.match(word) and not any(c.isdigit() for c in word)


def contains_phrase(answer_tokens, phrase, gap=GAP):
    """The phrase's tokens in order in the answer's, at most `gap` words between neighbours; a gap word is no negation and
    no number."""
    want = tokens(phrase)
    if not want:
        return False
    for start, word in enumerate(answer_tokens):
        if word != want[0]:
            continue
        pos, ok = start, True
        for w in want[1:]:
            for skip in range(gap + 1):
                nxt = pos + 1 + skip
                if (
                    nxt < len(answer_tokens)
                    and answer_tokens[nxt] == w
                    and all(_gap_ok(t) for t in answer_tokens[pos + 1 : nxt])
                ):
                    pos = nxt
                    break
            else:
                ok = False
                break
        if ok:
            return True
    return False


# ponytail: a link to any host whose path matches source_url_any counts; links in answers come from the sources (prompt rule 1).
def names_document(answer_tokens, answer, sources, expect):
    """find_document: the answer links a document that source_url_any accepts, or names word for word the title (two
    words at least) of a listed source it accepts."""

    def accepts(url):
        return any(url_matches(url, p, expect.get("source_url_match")) for p in expect.get("source_url_any", []))

    if any(accepts(u.rstrip(".,;:")) for u in LINK.findall(answer)):
        return True
    titles = {t for s in sources if accepts(s["url"]) and len(tokens(t := s.get("title") or "")) >= 2}
    return any(contains_phrase(answer_tokens, t, gap=0) for t in titles)


# ponytail: a phrase list, no dates. The probe says the event is past, so wording that presents it as coming is the fail;
# a new phrasing (or a negation outside its sentence) is missed until added, a date check is not worth its false alarms. A
# negation elsewhere in the same sentence hides an upcoming claim ("findet am 3. Juni statt, nicht am 2. Juni").
_IN = r"(?:[^.!?\n]|(?<=\d)\.)"  # inside a sentence: a dot after a digit ("am 23. Juli") does not end it
UPCOMING = re.compile(
    rf"\bdemnächst\b|\bbevorstehend\w*|\bnoch\b{_IN}{{0,60}}\banmelde\w*|"
    rf"\bstattfinde[nt]\b|\bfindet\b{_IN}{{0,120}}\bstatt\b|\bwird\b{_IN}{{0,120}}\bstattfinden\b",
    re.IGNORECASE,
)
# "nächste"/"kommende" count only in a sentence with a digit: "die nächsten Termine finden Sie auf der Seite" points on
COMING = re.compile(r"\b(?:nächste|kommende)[nmrs]?\b", re.IGNORECASE)
NEGATED = re.compile(r"\b(nicht|kein\w*|nie)\b", re.IGNORECASE)
# a sentence ends after a year ("am 23. Juli 2026."), never after a day: "am 23. Juli"
EVENT_SENTENCE = re.compile(r"(?:(?<=[^\d\s][.!?])|(?<=\d{4}[.!?]))\s+(?=[A-ZÄÖÜ])|\n+")


def presents_upcoming(answer):
    """True when the answer opens with "Ja" or a sentence without a negation presents an event as coming."""
    if re.match(r"\W*ja\b", answer, re.IGNORECASE):
        return True
    return any(
        (UPCOMING.search(s) or COMING.search(s) and re.search(r"\d", s)) and not NEGATED.search(s)
        for s in EVENT_SENTENCE.split(answer.replace("**", ""))
    )


POLARITY_WORDS = {"yes": r"ja|yes", "no": r"nein|no"}
SENTENCE_END = re.compile(r"(?<=[.!?])\s|\n")


def contradicts_polarity(answer, polarity):
    """True when a yes/no answer opens with the opposite word, or its first sentence holds the opposite as a short form
    ("Ja," / "Nein,") without opening with the expected word. Markdown emphasis, list markers and quote marks before the
    first word are ignored.
    ponytail: an answer that opens with context and states its verdict later passes unchecked; read the verdict
    sentence-wise if that gap matters."""
    text = re.sub(r"[*_`]", "", answer).lstrip(" \t\n>#-•\"'„“")
    first = SENTENCE_END.split(text, maxsplit=1)[0]
    same, other = POLARITY_WORDS[polarity], POLARITY_WORDS["no" if polarity == "yes" else "yes"]
    if re.match(r"(?:%s)\b" % same, first, re.IGNORECASE):
        return False
    return bool(
        re.match(r"(?:%s)\b" % other, first, re.IGNORECASE) or re.search(r"\b(?:%s)\s*," % other, first, re.IGNORECASE)
    )


# An English variant must be English. Words that are clearly one
# language; "in", "an", "will", "so" and the like are left out because both languages use them.
ENGLISH_WORDS = re.compile(
    r"\b(the|and|is|are|you|your|of|to|for|with|this|that|be|can|has|have|from|which|must|should|or)\b", re.IGNORECASE
)
GERMAN_WORDS = re.compile(
    r"\b(der|die|das|und|ist|sind|sie|ihr|ihre|nicht|mit|für|von|zu|bei|wird|werden|oder|auf|dem|den|"
    r"des|ein|eine|einer|im)\b",
    re.IGNORECASE,
)


def looks_english(text):
    """True when English words outweigh German ones. Text in round brackets (the German names the English variant gives
    after its translation), the [n] markers and table rules are left out; one English word is enough for a short answer."""
    body = re.sub(r"\([^)]*\)", " ", text or "")
    body = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]|^\s*\|[\s:|-]+\|\s*$", " ", body, flags=re.MULTILINE)
    english = len(ENGLISH_WORDS.findall(body))
    return english >= 1 and english > len(GERMAN_WORDS.findall(body))


# Scopes: the group_values check binds a value to the scope it is
# stated for. The expectation may name its own scopes ("scopes": {name: [terms]}); a term matches case-insensitively at a
# word start, so it is a prefix: "beamt" matches "Beamtinnen", never "Verbeamtung".
# Without "scopes" the status groups apply, where a bare "Beschäftigte" beside "Beamte" means Tarifbeschäftigte (the
# documents' convention).
STATUS_SCOPES = {
    "beamte": ["beamt"],
    "tarif": ["tarif", "tvöd", "arbeitnehmer", "beschäftigtenbereich"],
}
GROUP_BARE = re.compile(r"\bbeschäftigte[nr]?\b", re.IGNORECASE)
NEGATION = re.compile(r"\b(nicht|kein\w*)\b", re.IGNORECASE)
CLAUSE = re.compile(r"[;,](?!\d)\s*|\s+(?:bzw\.?|während|sowie|und für)\s+", re.IGNORECASE)
AND = re.compile(r"\s+und\s+", re.IGNORECASE)
# a sentence ends at . ! ? before a capital, never after a number ("30. April"), a single small letter ("z. B.") or a common
# abbreviation ("Hl."), but after a full date ("31.03. Für Beamte …")
ABBREVIATIONS = "".join(
    rf"(?<!\b{a}\.)"
    for a in ("Hl", "Dr", "St", "Nr", "Abs", "Art", "ca", "bzw", "ggf", "inkl", "evtl", "[a-z]", r"z\. B")
)
SENTENCE = re.compile(r"(?:(?<=[^\d][.!?])" + ABBREVIATIONS + r"|(?<=\d\d\.\d\d\.))\s+(?=[A-ZÄÖÜ*\[])")
LEAD_BOLD = re.compile(r"^\s*\*\*[^*]+\*\*\s*:?\s*$")


def _finder(scopes):
    """groups(text, context=None): the names of the scopes a text names; the patterns are compiled once per call."""
    custom = scopes is not None
    patterns = {
        name: re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + ")", re.IGNORECASE)
        for name, terms in (scopes if custom else STATUS_SCOPES).items()
    }

    def groups(text, context=None):
        found = {name for name, pattern in patterns.items() if pattern.search(text)}
        if not custom and GROUP_BARE.search(text) and patterns["beamte"].search(context or text):
            found.add("tarif")
        return found

    return groups


def _clauses(text, assigned, groups):
    """(scopes, clause) pieces of a text that names two or more scopes; a clause without a scope takes the scopes of the
    nearest clause that names one (itself, then left, then right), else the scopes assigned to the text. "und" ends a
    clause only after a number ("bis 31.03. und Beamte bis 30.04."), so "Beamte und Tarifbeschäftigte erhalten 31 Tage"
    stays one clause."""
    clauses = []
    for clause in CLAUSE.split(text):
        parts = []
        for part in AND.split(clause):
            if parts and not re.search(r"\d", parts[-1]):
                parts[-1] += " und " + part
            else:
                parts.append(part)
        clauses += parts
    named = [groups(c, text) for c in clauses]
    out = []
    for i, clause in enumerate(clauses):
        nearest = [named[j] for j in [*range(i, -1, -1), *range(i + 1, len(clauses))] if named[j]]
        out.append((nearest[0] if nearest else set(assigned), clause))
    return out


def _group_pieces(answer, groups, custom=False):
    """(scopes, text) pieces: table cells, list items and sentences, with "**" removed. A cell belongs to its column
    header's scope, else to its row label's, else to the scopes it names. A sentence belongs to the scopes it names, else
    to an earlier sentence of its line, else (list item) to the preceding non-item line, else to the section scope set by
    the last lead line (heading, bold-only line, line ending in ":"). A piece naming two or more scopes is cut into
    clauses. When the whole answer names one scope, pieces without a scope belong to it."""
    out, header, inherited, section = [], None, set(), set()
    for raw in str(answer or "").splitlines():
        line = raw.replace("**", "")
        if FMT_SEPARATOR.match(line):
            continue
        if FMT_ROW.match(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = [groups(c, line) for c in cells]
                continue
            row_groups = groups(cells[0], line)
            for i, cell in enumerate(cells):
                column = header[i] if i < len(header) else set()
                found = column if len(column) == 1 else row_groups if len(row_groups) == 1 else groups(cell, line)
                out.extend(_clauses(cell, found, groups) if len(found) >= 2 else [(found, cell)])
            continue
        header = None
        if not line.strip():
            continue
        is_item = bool(FMT_ITEM.match(line) or FMT_STEP.match(line))
        if not is_item and (raw.lstrip().startswith("#") or LEAD_BOLD.match(raw) or line.rstrip().endswith(":")):
            section = groups(line)
        earlier = set()
        for sentence in SENTENCE.split(line):
            found = groups(sentence) or earlier or (inherited if is_item else set()) or section
            earlier = groups(sentence) or earlier
            out.extend(_clauses(sentence, found, groups) if len(found) >= 2 else [(found, sentence)])
        if not is_item:
            inherited = groups(line)
    named = set().union(*(found for found, _ in out))
    single = named if len(named) == 1 and not custom else set()  # status scopes only: one custom scope claims nothing
    return [(found or single, text) for found, text in out]


def _negated(piece, value):
    """The clause that holds the value says "nicht" or "kein…" outside the value itself."""
    clause = next((c for c in CLAUSE.split(piece) if value in c), piece)
    return bool(NEGATION.search(clause.replace(value, " ")))


def group_values_failed(answer, expect):
    """Failures of a group_values expectation {"scopes": {name: [terms]}, name: {"any": [...], "none": [...]}}; "scopes"
    is optional (default: the status groups beamte and tarif), "ohne" = a piece that names no scope. any: some piece of
    the scope holds a value; none: no piece holds it unnegated."""
    # ponytail: text heuristics with four ceilings: "und" before any number joins two scopes in one clause
    # ("Beamte erhalten Familienzuschlag und Tarifbeschäftigte nicht"); "jeweils für Tarifbeschäftigte bzw. Beamte"
    # enumerations and comparisons like "(31.03. statt 30.04.)" bind a value to the wrong clause; a list of bare
    # "Beschäftigte" items under a lead-in with no group gets no group; a negation elsewhere in the value's clause hides a
    # wrong value. Each costs a false fail or a miss, never a crash; hand reading covers them.
    pieces = [
        (groups, text.lower())
        for groups, text in _group_pieces(answer, _finder(expect.get("scopes")), "scopes" in expect)
    ]
    failed = []
    for group, rule in expect.items():
        if group == "scopes":
            continue
        mine = [text for groups, text in pieces if (group in groups if group != "ohne" else not groups)]
        if rule.get("any") and not any(v.lower() in text for text in mine for v in rule["any"]):
            failed.append(f"group_values:{group}:any")
        failed += [
            f"group_values:{group}:none:{v}"
            for v in rule.get("none", [])
            if any(v.lower() in text and not _negated(text, v.lower()) for text in mine)
        ]
    return failed


def turn_expect(turn, original_shape):
    """The checks of a turn: every English variant gets the language check, and the shape "original" means the shape of
    the conversation's last question (every version is made from the original answer)."""
    expect = dict(turn.get("expect") or {})
    if turn.get("variant") == "english":
        expect["language"] = "en"
    if expect.get("shape") == "original":
        expect["shape"] = original_shape or "text"
    return expect


def evaluate_turn(answer, sources, errors, expect, meta=None):
    """Automatic checks for one turn. Returns a dict; `failed` lists every failed check."""
    expect = expect or {}
    low = answer.replace("**", "").lower()
    digits = re.sub(r"\D", "", answer)
    checks: dict[str, Any] = {
        "errors": errors,
        "empty": not answer.strip(),
        "leaks": sorted(set(LEAK_PATTERN.findall(answer))),
        "non_latin": sorted(set(NON_LATIN_PATTERN.findall(answer))),
        "emojis": len(EMOJI_PATTERN.findall(answer)),
        "informal": sorted({m.lower() for m in INFORMAL_PATTERN.findall(answer)}),
        "words": len(answer.split()),
        "n_sources": len(sources),
        "internal_sources": sum(1 for s in sources if s["internal"]),  # kept so old and new rows stay comparable
        "citations": sorted({int(n) for n in CITATION_PATTERN.findall(answer)}),
    }
    failed = []
    if errors:
        failed.append("error_event")
    if checks["empty"] and not expect.get("allow_empty"):
        failed.append("empty_answer")
    if checks["leaks"]:
        failed.append("leak")
    if checks["emojis"]:
        failed.append("emoji")
    if checks["non_latin"]:
        failed.append("non_latin")
    if checks["informal"] and not expect.get("allow_informal"):
        failed.append("informal_address")
    at = tokens(answer) if "contains_any" in expect else []
    if "contains_any" in expect and not (
        any(s.lower() in low for s in expect["contains_any"])
        or any(contains_phrase(at, s) for s in expect["contains_any"] if not LITERAL.search(s.casefold()))
        or (expect.get("names_document") and names_document(at, answer, sources, expect))
    ):
        failed.append("contains_any")
    for s in expect.get("contains_none", []):
        if s.lower() in low:
            failed.append("contains_none:" + s)
    failed += group_values_failed(answer, expect.get("group_values", {}))
    if expect.get("past_event") and presents_upcoming(answer):
        failed.append("past_event")
    if expect.get("polarity") and contradicts_polarity(answer, expect["polarity"]):
        failed.append("polarity")
    if "digits_any" in expect and not any(d in digits for d in expect["digits_any"]):
        failed.append("digits_any")
    if "source_url_any" in expect and not any(
        any(url_matches(s["url"], p, expect.get("source_url_match")) for p in expect["source_url_any"]) for s in sources
    ):
        failed.append("source_url_any")
    if expect.get("no_sources") and sources:
        failed.append("no_sources")
    if "max_words" in expect and checks["words"] > expect["max_words"]:
        failed.append("max_words")
    meta = meta or {}
    if "shape" in expect or "cited_min" in expect:
        produced = detect_shape(answer)
        cited, total = table_cited(answer) if produced == "table" else cited_share(answer, produced)
        checks["format"] = {"sent": meta.get("shape"), "produced": produced, "cited": [cited, total]}
        if "shape" in expect and (meta.get("shape") != expect["shape"] or produced != expect["shape"]):
            failed.append("shape")
        if total and cited / total < expect.get("cited_min", 0.8):
            failed.append("cited")
    if expect.get("variant_ok") and not meta.get("variant_ok"):
        failed.append("variant")
    if expect.get("language") == "en" and not looks_english(answer):
        failed.append("language")
    checks["failed"] = failed
    checks["auto_pass"] = not failed
    checks["manual_review"] = bool(expect.get("manual"))
    return checks
