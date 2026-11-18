"""The prompt's date without a process-wide locale: the same string as with the de_DE locale and
"%A, der %d.%m.%Y %H:%M:%S", for every weekday in every month."""

import locale
from datetime import date, datetime

import pytest

from rag_core.domain.dates import (
    asks_for_event,
    format_date,
    parse_term,
    shown_as_coming,
    term_start,
    terms_written,
    written_dates,
)

DAYS = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")
FORMAT = "%A, der %d.%m.%Y %H:%M:%S"
INSTANTS = [
    datetime(2026, month, day, 14, 3, 5) for month in range(1, 13) for day in range(1, 8)
]  # each month: all 7 days


def test_the_date_names_the_weekday_in_german() -> None:
    assert format_date(datetime(2026, 10, 12, 14, 3, 5), FORMAT, DAYS) == "Montag, der 12.10.2026 14:03:05"


def test_the_date_equals_the_de_DE_strftime_for_every_weekday_and_month() -> None:
    before = locale.setlocale(locale.LC_TIME)
    try:
        locale.setlocale(locale.LC_TIME, "de_DE.UTF-8")
    except locale.Error:
        pytest.skip("no de_DE.UTF-8 locale here (the Mac has one)")
    try:
        assert [format_date(now, FORMAT, DAYS) for now in INSTANTS] == [now.strftime(FORMAT) for now in INSTANTS]
    finally:
        locale.setlocale(locale.LC_TIME, before)


# The dates a text writes out, the questions about an event, the past dates an answer writes as coming.
TODAY = date(2026, 10, 26)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("am 12.03.2019", date(2019, 3, 12)),
        ("Rechtsstand 1.7.2007", date(2007, 7, 1)),
        ("Beispielstadt, den 26. 6.2019", date(2019, 6, 26)),
        ("Rundschreiben v. 13.07.17 an alle", date(2017, 7, 13)),
        ("am 04.04.00 eröffnet", date(2000, 4, 4)),
        ("am 01.10.99 gegründet", date(1999, 10, 1)),
        ("am 23. Juli 2026", date(2026, 7, 23)),
        ("seit 1. Okt. 2025", date(2025, 10, 1)),
        ("am 9_ Oktober 2026", date(2026, 10, 9)),
        ("sowie der 02.Januar 2015", date(2015, 1, 2)),
        ("Treffen: Donnerstag, 2. Juli, 14 Uhr", date(2026, 7, 2)),
        ("Programm: Mittwoch 19.11. ab 14 Uhr", date(2025, 11, 19)),
        ("Der Ausflug war am 23.07.2026.", date(2026, 7, 23)),
    ],
)
def test_each_written_date_form_is_read(text: str, expected: date) -> None:
    assert [d for _, _, d in written_dates(text, TODAY)] == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "nach 1.2.15 der Liste",
        "Ziffer 5.3.12 gilt",
        "am 2. Juli",  # no year, no weekday
        "Termine: 27.10. und 03.11.",
        "im Juli 2026",
        "seit 2019",
        "am 31.02.2020",
        "Az. 12.03.0815",
        "Mittwoch, 26.06.",  # no year of 2025-2027 has 26 June on a Wednesday
        "Frist bis 12.03.20",  # a window cut inside "12.03.2019": a two-digit year at the end of the text is no date
    ],
)
def test_what_is_no_date(text: str) -> None:
    assert written_dates(text, TODAY) == []


@pytest.mark.parametrize(
    "question",
    [
        "Wann ist die nächste Sitzung?",
        "Wann ist die kommende Sitzung?",
        "Gibt es demnächst einen Kurs?",
        "Kann ich mich noch zum Ausflug anmelden?",
        "Wann findet das Sommerfest statt?",
        "Ist eine Schulung geplant?",
        "Welche Veranstaltungen stehen bald an?",
    ],
)
def test_questions_about_an_event(question: str) -> None:
    assert asks_for_event(question)


@pytest.mark.parametrize(
    "question",
    [
        "Bis wann muss ich den Antrag stellen?",
        "Seit wann gilt die Regel?",
        "Ab wann gibt es das Jobticket?",
        "Ich bin ab nächster Woche krank.",
        "Ich gehe nächstes Jahr in Rente.",
        "Was gilt derzeit?",
        "Wann wurde die Satzung beschlossen?",
        "Was ist eine Dienstvereinbarung?",
    ],
)
def test_questions_that_are_no_event(question: str) -> None:
    assert not asks_for_event(question)


def test_a_past_date_written_as_coming_is_found_with_its_wording_and_sentence() -> None:
    answer = "Der nächste Ausflug ist am 23.07.2026 [2]. Anmeldung bei Frau X."
    assert shown_as_coming(answer, TODAY) == [
        (date(2026, 7, 23), "23.07.2026", "Der nächste Ausflug ist am 23.07.2026 [2].")
    ]


@pytest.mark.parametrize(
    "answer",
    [
        "Der Ausflug fand am 23.07.2026 statt [2].",
        "Der letzte Termin war der 1. Oktober 2026.",
        "Der nächste Ausflug ist am 03.11.2026 [1].",  # after today
        "Heute, am 26.10.2026, ist die Sitzung.",  # today is not past
        "Es gibt keinen Termin.",
    ],
)
def test_no_date_is_found_where_none_is_shown_as_coming(answer: str) -> None:
    assert shown_as_coming(answer, TODAY) == []


def test_bold_dates_and_several_sentences_are_read_in_order() -> None:
    answer = "Die Sitzung ist am **1. Okt. 2025** [1]. Ersatztermin: Donnerstag, 2. Juli.\nDanach folgt 04.08.2026."
    assert [(d, w) for d, w, _ in shown_as_coming(answer, TODAY)] == [
        (date(2025, 10, 1), "1. Okt. 2025"),
        (date(2026, 7, 2), "Donnerstag, 2. Juli"),
        (date(2026, 8, 4), "04.08.2026"),
    ]


def test_semester_notation_parsed() -> None:
    text = "seit WiSe 2019/20, im SoSe 2027 und ab WS 26/27, nicht WiSe 2019/22, nicht SoSe 2027/28"
    assert [(t, text[a:b]) for a, b, t in terms_written(text)] == [
        ((2019, 1), "WiSe 2019/20"),
        ((2027, 0), "SoSe 2027"),
        ((2026, 1), "WS 26/27"),
    ]
    assert parse_term("Wintersemester 2023") == (2023, 1) and parse_term("Sommersemester 2024") == (2024, 0)
    assert parse_term("WiSe 2023/2024") == (2023, 1) and parse_term("kein Semester") is None
    assert term_start((2026, 1)) == date(2026, 10, 1) and term_start((2027, 0)) == date(2027, 4, 1)
