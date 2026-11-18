"""The prompt's date, formatted "%A, der %d.%m.%Y %H:%M:%S"; the weekday's name comes from the profile, so no process-wide
locale is needed and the image carries none. The time zone is the container's (TZ=Europe/Berlin).
Also the dates a text writes out, the questions that ask for the next or a coming event, and the past dates an answer writes as
if they were still to come."""

import re
from collections.abc import Sequence
from datetime import date, datetime

# German date grammar: code, not profile (German administrative language, like the scope rules)
MONTHS = {
    "januar": 1, "jan.": 1, "februar": 2, "feb.": 2, "märz": 3, "mär.": 3, "april": 4, "apr.": 4, "mai": 5,
    "juni": 6, "jun.": 6, "juli": 7, "jul.": 7, "august": 8, "aug.": 8, "september": 9, "sep.": 9, "sept.": 9,
    "oktober": 10, "okt.": 10, "november": 11, "nov.": 11, "dezember": 12, "dez.": 12,
}  # fmt: skip
WEEKDAYS = ("montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag")
_WEEKDAY = r"(?:(%s)\b,?\s*(?:den\s+)?)?" % "|".join(WEEKDAYS)
_MONTH = "|".join(re.escape(m) if m.endswith(".") else m + r"\b" for m in sorted(MONTHS, key=len, reverse=True))
# 12.03.2019, 1.7.2007, 26. 6.2019, 19.11. (only after a weekday), 13.07.17 (a two-digit year only with two-digit day and month)
NUMERIC_DATE = re.compile(
    _WEEKDAY + r"(?<![\d.])(?:(\d{1,2})\.\s?(\d{1,2})\.(?:\s?(\d{4}))?|(\d{2})\.(\d{2})\.(\d{2}))(?!\d|\.\d)",
    re.IGNORECASE,
)
# 23. Juli 2026, 1. Okt. 2025, 9_ Oktober 2026 and 02.Januar 2015 (OCR), Donnerstag, 2. Juli (only after a weekday)
NAMED_DATE = re.compile(_WEEKDAY + r"(?<![\d.])(\d{1,2})[._]?\s*(" + _MONTH + r")(?:\s+(\d{4}))?(?!\d)", re.IGNORECASE)
# "ab nächster Woche krankgeschrieben": the asker's own time, not an event
_OWN_TIME = r"(?!\s+(?:woche|monat|jahr|tag)\b)"
# the next or a coming event; deadlines and validity ("bis wann", "seit wann", "ab wann") are not events
EVENT_QUESTION = re.compile(
    rf"\b(?:nächste[nmrs]?{_OWN_TIME}|demnächst|kommende[nmrs]?{_OWN_TIME}|bald|anstehende?\w*|bevorstehende?\w*|geplant\w*|"
    r"noch\b.{0,60}\b(?:anmelde\w*|teilnehme\w*|buche\w*)|"
    r"(?<!bis )(?<!seit )(?<!ab )wann\s+(?:ist|sind|findet|finden|fährt|fahren|beginnt|beginnen|startet|starten|gibt))\b",
    re.IGNORECASE,
)
# a sentence with one of these words says the date has passed
PAST_WORD = re.compile(
    r"\b(?:fand|fanden|war|waren|wurde|wurden|endete|endeten|lag|lagen|zuletzt|bereits|vergangen\w*|vorbei|abgelaufen\w*|"
    r"ehemalig\w*|damals|letzte[nmrs]?|seit|stattgefunden)\b",
    re.IGNORECASE,
)
# evals' EVENT_SENTENCE (apps/evals/src/evals/checks.py): a sentence ends after a year ("am 23. Juli 2026."), never after a day
SENTENCE = re.compile(r"(?:(?<=[^\d\s][.!?])|(?<=\d{4}[.!?]))\s+(?=[A-ZÄÖÜ])|\n+")


def format_date(now: datetime, date_format: str, day_names: Sequence[str]) -> str:
    """now in date_format, %A replaced by the name of now's weekday (day_names: Monday first; no % in a name)."""
    return now.strftime(date_format.replace("%A", day_names[now.weekday()]))


def asks_for_event(question: str) -> bool:
    """True when the question asks for the next or a coming event."""
    return bool(EVENT_QUESTION.search(question or ""))


def _weekday(year: int, month: int, day: int) -> int:
    try:
        return date(year, month, day).weekday()
    except ValueError:
        return -1


def _date(weekday: str | None, day: int, month: int, year: int | None, today: date) -> date | None:
    """The date, or None; without a year only after a weekday that falls on day.month in exactly one year of [today-1, today+1]."""
    if year is None and weekday:
        wanted = WEEKDAYS.index(weekday.lower())
        years = [y for y in range(today.year - 1, today.year + 2) if _weekday(y, month, day) == wanted]
        year = years[0] if len(years) == 1 else None
    if year is None or year < 1900:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def written_dates(text: str, today: date) -> list[tuple[int, int, date]]:
    """(start, end, date) of each date the text writes out, in order, overlaps dropped (the first wins)."""
    found: list[tuple[int, int, date]] = []
    for m in NUMERIC_DATE.finditer(text):
        if m[2]:
            d = _date(m[1], int(m[2]), int(m[3]), int(m[4]) if m[4] else None, today)
        elif m.end() == len(text):  # "12.03.20" at the end of a cut window may be the start of "12.03.2019"
            continue
        else:
            yy = int(m[7])
            d = _date(m[1], int(m[5]), int(m[6]), 2000 + yy if yy <= today.year % 100 + 1 else 1900 + yy, today)
        if d:
            found.append((m.start(), m.end(), d))
    for m in NAMED_DATE.finditer(text):
        d = _date(m[1], int(m[2]), MONTHS[m[3].lower()], int(m[4]) if m[4] else None, today)
        if d:
            found.append((m.start(), m.end(), d))
    kept: list[tuple[int, int, date]] = []
    end = -1
    for start, stop, d in sorted(found):
        if start >= end:
            kept.append((start, stop, d))
            end = stop
    return kept


def shown_as_coming(answer: str, today: date) -> list[tuple[date, str, str]]:
    """(date, its wording, sentence) of each date before today that the answer writes in a sentence without a past word, in
    order; bold markers are ignored."""
    found: list[tuple[date, str, str]] = []
    for sentence in SENTENCE.split(answer.replace("**", "")):
        if not PAST_WORD.search(sentence):
            found += [
                (d, sentence[start:end], sentence.strip())
                for start, end, d in written_dates(sentence, today)
                if d < today
            ]
    return found
