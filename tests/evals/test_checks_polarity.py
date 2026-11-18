"""polarity probes: a yes/no answer fails when its first sentence gives the opposite verdict; a verdict stated later
in the text is not checked."""

import pytest

from evals.checks import evaluate_turn


def failed(answer, polarity):
    return evaluate_turn(answer, [], [], {"polarity": polarity})["failed"]


FAILS = [
    ("Nein, das ist nicht erlaubt.", "yes"),
    ("**Nein.** Das ist nicht erlaubt.", "yes"),
    ("- Nein, das ist nicht erlaubt.", "yes"),
    ("\n  nein: das ist nicht erlaubt.", "yes"),
    ("Ja, das ist erlaubt.", "no"),
    ("Grundsätzlich ja, aber nur mit Genehmigung.", "no"),
    ("No, you may not.", "yes"),
    ("**Yes**, you may.", "no"),
    ("Das ist, nein, nicht erlaubt.", "yes"),
    ("Nein\nDas ist nicht erlaubt.", "yes"),
]
PASSES = [
    ("Ja, das ist erlaubt.", "yes"),
    ("**Ja.** Das ist erlaubt.", "yes"),
    ("Nein, das ist nicht erlaubt.", "no"),
    ("Yes, you may.", "yes"),
    ("No, you may not.", "no"),
    ("Nein, das ist nicht erlaubt. Ja, Ausnahmen gibt es.", "no"),  # the second sentence is not looked at
    ("Ja, nein, vielleicht.", "yes"),  # opens with the expected word
    ("Laut Dienstanweisung gilt Folgendes. Nein, das ist nicht erlaubt.", "yes"),  # verdict later: unchecked
    ("Die Regelung nennt keinen Betrag.", "yes"),
    ("Jahresurlaub wird gewährt.", "no"),  # "Ja" only as a word start
    ("Nobody knows.", "yes"),
    ("", "yes"),
]


@pytest.mark.parametrize("answer,polarity", FAILS)
def test_opposite_verdict_fails(answer, polarity):
    assert "polarity" in failed(answer, polarity)


@pytest.mark.parametrize("answer,polarity", PASSES)
def test_other_answers_pass(answer, polarity):
    assert "polarity" not in failed(answer, polarity)


def test_no_polarity_expectation_is_not_checked():
    assert "polarity" not in evaluate_turn("Nein.", [], [], {})["failed"]
