"""Which acronym groups a question names, and the question without those terms."""

from rag_core.domain.acronyms import expand

GROUPS = (
    ("PO", "Prüfungsordnung", "Studiengangsordnung"),
    ("PO-Wechsel", "Wechsel der Prüfungsordnung"),
    ("CP", "LP", "Credit Points", "Leistungspunkte"),
    ("Rücktritt", "Rücktritt von der Prüfung"),
)


def test_a_term_names_its_group_and_leaves_the_rest_of_the_question() -> None:
    rest, groups = expand("Was steht in der PO zur Anmeldung?", GROUPS)
    assert groups == [GROUPS[0]] and rest.split() == ["Was", "steht", "in", "der", "zur", "Anmeldung?"]


def test_matching_ignores_case_and_takes_a_plural_or_genitive_of_a_long_term() -> None:
    assert expand("prüfungsordnungen", GROUPS)[1] == [GROUPS[0]]
    assert expand("des Rücktritts", GROUPS)[1] == [GROUPS[3]]
    assert expand("wie viele credit   points", GROUPS)[1] == [GROUPS[2]]


def test_a_term_is_a_whole_word_never_part_of_another() -> None:
    assert expand("Post und Polizei, Apo, Cpu", GROUPS) == ("Post und Polizei, Apo, Cpu", [])
    assert expand("Pflicht-PO", GROUPS)[1] == [GROUPS[0]]  # a hyphen ends a word


def test_the_longest_term_wins_where_terms_overlap() -> None:
    rest, groups = expand("Wie klappt der PO-Wechsel?", GROUPS)
    assert groups == [GROUPS[1]] and "PO" not in rest  # not the group of "PO" besides


def test_groups_come_in_order_of_appearance_once_each() -> None:
    _, groups = expand("Wie viele LP zählt der Rücktritt von der Prüfung, die PO und die CP?", GROUPS)
    assert groups == [GROUPS[2], GROUPS[3], GROUPS[0]]  # "Rücktritt von der Prüfung" is one term, not two matches


def test_nothing_named_leaves_the_question_as_it_is() -> None:
    assert expand("Wo ist das Studienbüro?", GROUPS) == ("Wo ist das Studienbüro?", [])
    assert expand("PO", ()) == ("PO", [])
