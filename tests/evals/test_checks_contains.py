"""contains_any reads the wording, not the spelling: rows are taken from the wording of stored answers. The verbatim test
always stays first, so nothing that passed can fail."""

import pytest

from evals.checks import evaluate_turn

DOC = "https://docs.example.org/circulars/2031-5/2031-05-ab-verfuegung/"
OTHER = "https://docs.example.org/circulars/2032-1/2032-01-ab-verfuegung/"


def passes(answer, phrase):
    return "contains_any" not in evaluate_turn(answer, [], [], {"contains_any": [phrase]})["failed"]


SAME = [
    ("Der Termin ist am 13. November 2026.", "13.11.2026"),
    ("Der Termin ist am 13.11.2026.", "13. November 2026"),
    ("Stichtag ist der 08.04.", "8. April"),
    ("Es sind noch vier Plätze frei.", "4 Plätze"),
    ("Beginn um 15.30 Uhr in der Mensa.", "15:30"),
    ("Beginn um 15 Uhr.", "15:00 Uhr"),
    ("Sie erreichen uns unter 0000 – 55 66 77.", "0000556677"),
    ("Telefon 0000/55 66 77", "0000 – 55 66 77"),
    ("Die Gebühr beträgt 60,- Euro.", "60 Euro"),
    ("Die Gebühr beträgt 60,00 €.", "60,- Euro"),
    ("Der Hebesatz beträgt 385 v. H.", "385 %"),
    ("Der Zuschuss beträgt 1.500 EUR.", "1500 Euro"),
    ("Wir prüfen den Fiktiven Betrieb.", "Fiktive Betriebe"),
    ("Das gilt ab frühestens eine Woche vor dem tatsächlichen Auszug.", "frühestens eine Woche vor dem Auszug"),
    ("Siehe „Notwendige Vergabeunterlagen“ im Handbuch.", "Notwendige Vergabeunterlagen"),
    ("Siehe **Nr. 47 /2031** [1, 2].", "Nr. 47/2031"),
    ("Die Straße ist gesperrt.", "Strasse"),
    ("Der Wert ist 2,7183 ha.", "2,7183 ha"),
    ("Der Wert ist 0,5 ha.", "0,5 ha"),
    ("Die Frist beträgt eine Woche.", "1 Woche"),
    ("Die Frist beträgt 1 Jahr.", "ein Jahr"),
    ("Tel. 0000 – 55 66 77", "0000556677"),
]
GUARDS = [
    ("Sie sind von der Bezuschussung nicht ausgenommen.", "von der Bezuschussung ausgenommen"),
    ("Die Frist beträgt 3 bis 6 Monate.", "3 Monate"),
    ("Der Zuschuss beträgt 5.000 €.", "5 €"),
    ("Es gilt der Hebesatz 385.", "385 %"),
    ("Unter Termine auf example.org finden Sie alles.", "events.example.org"),
    ("Ich helfe Ihnen gern weiter.", "Wir unterstützen Sie gerne"),
    ("Der Termin ist am 14.11.2026.", "13. November 2026"),
    ("Die Gebühr beträgt 2,50 €.", "50 Euro"),
    ("Es sind 3,5 %.", "5 Prozent"),
    ("Es sind 1,5 Millionen.", "1. Mai"),
    ("Die Frist beträgt fünf Jahre.", "ein Jahr"),
    ("Es dauert zwei Wochen.", "eine Woche"),
    ("Antrag niemals genehmigt", "Antrag genehmigt"),
    ("Der Zuschuss beträgt 5.000 €.", "5 Euro"),
]


@pytest.mark.parametrize("answer,phrase", SAME)
def test_same_wording_passes(answer, phrase):
    assert passes(answer, phrase)


@pytest.mark.parametrize("answer,phrase", GUARDS)
def test_guards_stay_failed(answer, phrase):
    assert not passes(answer, phrase)


def test_url_phrase_matches_verbatim():
    assert passes("Siehe https://events.example.org/calendar", "events.example.org")


def test_old_substring_test_stays_first():
    assert passes("Siehe Nr. 47 /2031 und mehr", "Nr. 47 /2031")


def test_names_document():
    expect = {
        "contains_any": ["Verfügung Nr. 5/2031"],
        "source_url_any": ["2031-5/2031-05-ab-verfuegung/"],
        "names_document": True,
    }
    title = "Verfügung der Abteilungsleitung Pilotphase Beispielverfahren"
    sources = [{"url": DOC + "?layout=file", "title": title, "internal": True}]

    def failed(answer, sources, expect=expect):
        return evaluate_turn(answer, sources, [], expect)["failed"]

    assert "contains_any" in failed("Das steht in einer Verfügung.", sources, {**expect, "names_document": False})
    assert "contains_any" not in failed("Hier: " + DOC + "?layout=file.", sources)  # a link
    assert "contains_any" not in failed(
        "Das ist die Verfügung der Abteilungsleitung Pilotphase Beispielverfahren.", sources
    )
    other = [{"url": OTHER, "title": title, "internal": True}]
    assert "contains_any" in failed("Hier: " + OTHER, other)  # a link to another document
    assert "contains_any" in failed("Das ist die Verfügung der Abteilungsleitung Pilotphase Beispielverfahren.", other)
    assert "contains_any" in failed("Das steht in einer Verfügung.", sources)  # no link, no title


def test_names_document_title_rules():
    def failed(answer, title):
        sources = [{"url": DOC, "title": title, "internal": True}]
        expect = {"contains_any": ["x y z"], "source_url_any": ["2031-05-ab-verfuegung/"], "names_document": True}
        return "contains_any" in evaluate_turn(answer, sources, [], expect)["failed"]

    assert failed("Das ist das Rundschreiben.", "Rundschreiben")  # a one-word title proves nothing
    assert not failed("Das ist die Verfügung Pilotphase Beispielverfahren.", "Verfügung Pilotphase Beispielverfahren")
    assert failed(
        "Das ist die Verfügung neue Pilotphase Beispielverfahren.", "Verfügung Pilotphase Beispielverfahren"
    )  # gap 0


def test_names_document_survives_a_source_without_title():
    sources = [{"url": DOC, "title": None, "internal": True}]
    expect = {"contains_any": ["x y z"], "source_url_any": ["2031-05-ab-verfuegung/"], "names_document": True}
    assert "contains_any" in evaluate_turn("Nichts.", sources, [], expect)["failed"]
