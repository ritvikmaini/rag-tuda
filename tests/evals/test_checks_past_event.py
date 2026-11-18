"""past_event probes: the probe asserts the event is over, so an answer that presents it as coming fails; rows are taken
from the wording of stored answers."""

import pytest

from evals.checks import evaluate_turn


def failed(answer, expect=None):
    return evaluate_turn(answer, [], [], {"past_event": True} if expect is None else expect)["failed"]


FAILS = [
    "Die nächste Sitzung findet am 12. Mai 2026 statt.",
    "Das Ratschbankerl findet demnächst im Sommer statt.",
    "Sie können sich noch bis Freitag anmelden.",
    "Die kommende Delegiertenversammlung wird im Herbst stattfinden.",
    "Ja, die Versammlung war am 8. April 2025.",
    "Sie findet am 23. Juli 2026 statt. Ein Termin steht fest.",  # "23. Juli" is no sentence end
    "Der Termin ist unklar.\nDie nächsten Termine am 5. Mai stehen im Kalender.",
    "Die kommende Sitzung ist am 3. Juni.",
]
PASSES = [
    "Ein neuer Termin ist nicht bekannt. Das letzte Treffen war am 16. Oktober 2024.",
    "Nein, es gibt demnächst keinen Termin.",
    "Die Veranstaltung fand am 23. Juli 2026 statt.",
    "Die nächste Sitzung findet nicht mehr statt.",
    "Die Quellen nennen nur das Treffen vom 16. Oktober 2024.",
    "Das letzte Treffen war am 16. Oktober 2024. Die nächsten Termine finden Sie auf der Seite des Beirats.",
    "Die kommenden Termine stehen im Kalender.",
]


@pytest.mark.parametrize("answer", FAILS)
def test_upcoming_is_a_fail(answer):
    assert "past_event" in failed(answer)


@pytest.mark.parametrize("answer", PASSES)
def test_past_wording_passes(answer):
    assert "past_event" not in failed(answer)


def test_only_probes_with_the_key_are_checked():
    assert "past_event" not in failed("Die nächste Sitzung findet am 12. Mai 2026 statt.", {})
