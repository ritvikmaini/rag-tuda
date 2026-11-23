"""A document's own scope statement: which cues print, which never do. Hand-written snippets."""

from rag_core.settings.profile import ScopeVocabulary

from indexer.chunking.scope import MAX_CHARS, document_scope

VOCABULARY = ScopeVocabulary(
    groups={
        "Beamtinnen und Beamte": ("beamt",),
        "Tarifbeschäftigte (TVöD)": ("tarifbeschäftigt", "tvöd"),
        "Anwärterinnen und Anwärter": ("anwärter",),
    },
    areas=("studierendenwerk", "rechenzentrum"),
    compilations=("haushaltsplan",),
)


def scope(page1, *more, title="Rundschreiben"):
    pages = [(1, page1)] + [(n, text) for n, text in enumerate(more, 2)]
    return document_scope(title, pages, VOCABULARY)


def test_an_addressee_naming_a_group_is_the_scope():
    assert scope(
        "Rundschreiben 2025\nAn alle Beamtinnen und Beamten der Fachbereiche\nAnsparung\nSehr geehrte Damen und Herren,"
    ) == ("An alle Beamtinnen und Beamten der Fachbereiche")


def test_a_distribution_addressee_and_a_form_recipient_give_nothing():
    assert scope("Rundschreiben 2024\nAn alle Fachbereiche und Dezernate\nUrlaub 2024") == ""
    assert scope("Antrag\nAn die Universitätskasse\nHiermit beantrage ich") == ""


def test_an_addressee_over_several_lines_is_joined():
    page1 = "Rundschreiben 2026\nAn alle\nFachbereiche und Dezernate\n(Beamtinnen und Beamte des\nStudierendenwerks Muster)\nTauschtage"
    assert scope(page1) == "An alle Fachbereiche und Dezernate (Beamtinnen und Beamte des Studierendenwerks Muster)"


def test_an_exclusion_never_prints_and_the_other_cues_still_do():
    page1 = "Rundschreiben 2026\nAn alle\nFachbereiche und Dezernate\n(außer Stiftung Muster und\nStudierendenwerk Muster)\nTauschtage"
    assert scope(page1) == ""
    assert (
        scope("Rundschreiben\nAn alle Fachbereiche und Dezernate (ausgenommen Studierendenwerk Muster)\nJobticket")
        == ""
    )
    assert scope("Rundschreiben\nAn alle Fachbereiche (mit Ausnahme des Studierendenwerks Muster)\nUmfrage") == ""
    assert scope("Rundschreiben\nDie Regelung gilt nicht für das Rechenzentrum.") == ""
    assert (
        scope(
            "Leitlinie\n§ 1 Geltungsbereich\nDer Geltungsbereich umfasst alle Einrichtungen der Universität (ohne Studierendenwerk)."
        )
        == ""
    )
    assert scope("Hinweise\nDiese Dienstanweisung gilt für alle Beamten, aber keine Anwärter.") == ""
    assert scope("Hinweise für Beamte (nicht für Anwärter)\nText") == ""  # nor a label from a heading with an exclusion
    page1 = "Rundschreiben\nAn alle Fachbereiche (außer Studierendenwerk Muster)\nDie Regelung gilt für alle Beamtinnen und Beamten."
    assert scope(page1) == "Die Regelung gilt für alle Beamtinnen und Beamten."


def test_a_scope_heading_gives_its_first_sentence():
    page2 = "§ 2 Geltungsbereich\nDiese Dienstvereinbarung gilt für alle Beschäftigten des Studierendenwerks Muster. Sie regelt die Prämie."
    assert scope("Dienstvereinbarung Leistungsprämie\nPräambel", page2, title="DV Leistungsprämie") == (
        "Diese Dienstvereinbarung gilt für alle Beschäftigten des Studierendenwerks Muster."
    )


def test_a_heading_sentence_runs_across_line_breaks():
    page = (
        "§ 1 Geltungsbereich\nDiese Satzung gilt für die Kinderbetreuung des\nStudierendenwerks Muster.\n§ 2 Aufnahme"
    )
    assert (
        scope("Satzung\nPräambel", page) == "Diese Satzung gilt für die Kinderbetreuung des Studierendenwerks Muster."
    )


def test_a_sentence_about_the_document_counts_on_any_page():
    page2 = "Weitere Hinweise folgen. Vorliegende Änderungen gelten ausschließlich für Beschäftigte des TVöD. Mit freundlichen Grüßen"
    assert scope("Rundschreiben 2025\nAn alle Fachbereiche und Dezernate\nTarifrunde", page2) == (
        "Vorliegende Änderungen gelten ausschließlich für Beschäftigte des TVöD."
    )
    assert (
        scope("Rundschreiben\nDie Regelung gilt für das Rechenzentrum.") == "Die Regelung gilt für das Rechenzentrum."
    )


def test_rule_level_sentences_give_nothing():
    assert scope("Hinweise\nSatz 1 gilt nicht für Beamte, die einen eigenen Anspruch haben.") == ""
    assert scope("Hinweise\nDies gilt auch für Beamtinnen und Beamte.") == ""
    assert scope("Hinweise\nFür Teilzeitkräfte gilt das entsprechend.") == ""


def test_a_form_field_after_the_heading_gives_nothing():
    assert scope("Meldebogen\nGeltungsbereich\nName der Schule:\nBeamte") == ""


def test_a_compilation_and_a_contents_line_give_nothing():
    page = "Inhalt\n§ 1 Geltungsbereich\nDiese Satzung gilt für das Studierendenwerk Muster."
    assert scope(page, title="Haushaltsplan 2026") == ""
    assert scope("Inhalt\n4. Geltungsbereich ........ 7\n5. Diese Dienstanweisung gilt für Beamte ........ 8") == ""


def test_a_group_in_the_page1_heading_gives_its_label_the_title_alone_none():
    assert (
        scope("§ Ihr Recht - Tarifbeschäftigte (TVöD)\nThemen:", title="Ihr Recht Beschäftigte")
        == "Tarifbeschäftigte (TVöD)"
    )
    assert scope("Rundschreiben 2025\nText", title="Ansparung URL Beamte") == ""  # a file title often names a topic
    assert scope("Bereich: 0 Beamte 0 Beschäftigte\nAntrag") == ""  # a form row is no heading
    assert scope("Förmliche Verpflichtung nichtbeamteter Personen\nText") == ""  # a stem only at a word start
    assert (
        scope("Satzung\nDiese Satzung gilt für die Nutzung zum Zwecke des außerschulischen Sports.") == ""
    )  # whole words


def test_a_group_word_below_the_page1_heading_gives_nothing():
    page1 = "Rundschreiben 2025\nUrlaub\nAnwärterinnen und Anwärter im Vorbereitungsdienst sind nicht berücksichtigt."
    assert scope(page1, title="Urlaub") == ""


def test_a_label_already_in_the_addressee_is_not_repeated():
    page1 = "Rundschreiben 2025\nAn alle Beamtinnen und Beamten der Fachbereiche\nAnsparung"
    assert scope("Ansparung für Beamte\n" + page1) == "An alle Beamtinnen und Beamten der Fachbereiche"


def test_the_line_keeps_whole_cues_within_the_limit():
    long_area = "An alle Beschäftigten des Studierendenwerks Muster " + "und ihrer Betriebe " * 3
    statement = "Die Regelung gilt für Anwärter im Rechenzentrum " + "und seine Außenstellen " * 4 + "."
    joined = scope("Hinweise für Anwärter\n" + long_area.strip() + "\nBetreff\n" + statement)
    assert len(joined) <= MAX_CHARS and joined.startswith("An alle Beschäftigten") and "Rechenzentrum" not in joined
    assert joined.endswith("; Anwärterinnen und Anwärter")  # the cut statement named them, the printed line does not
    one = scope(
        "Hinweise\nDiese Dienstanweisung gilt für das Studierendenwerk Muster "
        + "mit Sitz in der Musterstraße " * 8
        + "."
    )
    assert len(one) <= MAX_CHARS and one.endswith("…") and " " not in one[-2:]
    assert (
        scope(
            "Rundschreiben\nAn alle Beschäftigten des Studierendenwerks Muster " + "mit Sitz in der Musterstraße " * 4
        )
        == ""
    )


def test_nothing_without_a_limit():
    empty = ScopeVocabulary(groups={}, areas=(), compilations=())
    assert document_scope("Gesetz", [(1, "§ 1 Geltungsbereich\nDieses Gesetz gilt für alle Behörden.")], empty) == ""
    # "nur für" names no one: a restriction alone is no limit
    assert scope("Klausel\n§ 1 Geltungsbereich\nDie Klausel gilt nur für die Stoffe im Verzeichnis.") == ""


def test_only_an_opening_scope_heading_counts():
    law = "§ 91 Anwendungsbereich\nDie §§ 84 und 85 gelten nur für Beamte im Dienst."
    assert scope("Gesetz\nText", law) == ""
    assert scope("Gesetz\nText", "§ 2 Geltungsbereich\nDie Satzung gilt nur für das Studierendenwerk Muster.") == (
        "Die Satzung gilt nur für das Studierendenwerk Muster."
    )


def test_sentences_end_after_a_year_never_after_an_abbreviation():
    assert scope("Hinweise\nDie Regelung gilt für die Hl. Musterstiftung und das Studierendenwerk Muster.") == (
        "Die Regelung gilt für die Hl. Musterstiftung und das Studierendenwerk Muster."
    )
    page = "Diese Dienstanweisung gilt für alle Beschäftigten bis 2025. Sie regelt die Prämie des Studierendenwerks Muster."
    assert scope("Hinweise\n" + page) == ""


def test_tables_following_passages_and_corresponding_rules_give_nothing():
    assert scope("Hinweise\n| Diese Dienstanweisung gilt für Beamte. | 2025 |") == ""
    assert scope("Hinweise\nNachfolgendes gilt ausschließlich für Beamte:") == ""
    assert scope("Hinweise\nDiese Regelung gilt für Beamte entsprechend.") == ""


def test_a_heading_sentence_stops_at_the_next_heading():
    page = "§ 1 Geltungsbereich\nDie Anordnung gilt für das Studierendenwerk Muster\n2. Umsetzung\nDie Umsetzung folgt."
    assert scope("Anordnung\nText", page) == "Die Anordnung gilt für das Studierendenwerk Muster"


def test_an_addressee_stops_at_the_salutation():
    page1 = "Rundschreiben\nAn alle Fachbereiche (Studierendenwerk Muster\nSehr geehrte Damen und Herren,\nText"
    assert scope(page1) == "An alle Fachbereiche (Studierendenwerk Muster"
