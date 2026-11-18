import pytest

from rag_core.domain.language import DE_STOPWORDS, EN_STOPWORDS, detect, looks_english_question, strip_foreign


def test_foreign_scripts_are_dropped_and_other_text_is_kept() -> None:
    assert strip_foreign("Antwort 中文 かな 한글 ，ok Привет äöü [1]", True) == "Antwort    ok Привет äöü [1]"
    assert strip_foreign("Nur Deutsch.", True) == "Nur Deutsch."


def test_an_english_question_has_two_english_function_words_and_no_german_one() -> None:
    assert looks_english_question("How do I apply for leave?")
    assert looks_english_question("Please summarize the service agreement on mobile work.")
    assert looks_english_question("Can you shorten that?")
    assert looks_english_question("What does 'Dienstvereinbarung' mean in the archive?")
    assert not looks_english_question("Wie beantrage ich Urlaub?")
    assert not looks_english_question("Wie funktioniert das Home Office Tool?")
    assert not looks_english_question("Urlaub")
    assert not looks_english_question("")
    assert not looks_english_question("Wo ist the Formular?")  # one English hit and a German one


def test_a_german_question_with_an_english_title_stays_german() -> None:
    for question in (
        "Was bedeutet Code of Conduct for the City?",
        'Was steht im Leitfaden "Working from home: the do and dont"?',
        "Info zu Terms and Conditions of the App",
        "Was bedeutet „Code of Conduct for the City“?",
        "Was steht im Leitfaden »How to apply for the leave«?",
    ):
        assert not looks_english_question(question), question
    assert looks_english_question('What is the "Dienstvereinbarung" about?')


DE = [
    "Was regelt § 535 BGB?",
    "Wie hoch darf die Kaution sein?",
    "Kann der Vermieter die Miete erhöhen?",
    "Wann muss ich die Nebenkostenabrechnung bezahlen?",
    "Wer zahlt die Kaution?",
    "Welche Rechte hat der Mieter bei Schimmel in der Wohnung?",
    "Müssen wir die Heizung im Sommer abschalten?",
    "ist das erlaubt",
    "Hello, wie funktioniert das?",
    "Thanks, und jetzt bitte die Zusammenfassung",
]
DE_WITHOUT_UMLAUTS = [
    "Kuendigungsfrist fuer Mieter ueber 5 Jahre?",
    "Muss der Mieter waehrend der Renovierung die Miete zahlen?",
    "Koennen Sie mir erklaeren, was eine Kappungsgrenze ist?",
]
EN = [
    "What does § 573c BGB say?",
    "How high may the deposit be?",
    "Can the landlord raise the rent?",
    "When do I have to pay the utility bill?",
    "Who pays the deposit?",
    "Is the landlord allowed to enter the flat?",
    "Does the BGB regulate rent increases after modernisation?",
    "What are the notice periods under § 573c BGB for tenants?",
    "Explain the Kappungsgrenze for me",
    "Please summarise the BetrKV",
    "§ 573c BGB: what does it mean for me?",
    "IS THIS ALLOWED",
    "What's the rent?",
    "Why is the rent so high in Munich?",
]
MIXED_MAJORITY = [
    ("Can you tell me was ein Mietvertrag ist?", "en"),  # 3 English stopwords against 2 German ones
    ("Was ist the Mietspiegel?", "de"),  # one each: a tie goes to German
]
NO_EVIDENCE = [
    "",
    "   ",
    "§ 556 BGB 12345",
    "Heizkostenverordnung § 7 Abs. 1",
    "Wohnflächenverordnung",
    "Rent increase",  # no stopword in either set: German by the tie rule
]


@pytest.mark.parametrize("text", DE + DE_WITHOUT_UMLAUTS)
def test_german(text):
    assert detect(text) == "de"


@pytest.mark.parametrize("text", EN)
def test_english(text):
    assert detect(text) == "en"


@pytest.mark.parametrize("text,expected", MIXED_MAJORITY)
def test_mixed(text, expected):
    assert detect(text) == expected


@pytest.mark.parametrize("text", NO_EVIDENCE)
def test_tie_or_empty_is_german(text):
    assert detect(text) == "de"


def test_case_and_punctuation_do_not_matter():
    assert detect("THE RENT, THE DEPOSIT!") == detect("the rent the deposit") == "en"


def test_sets_do_not_overlap():
    assert not DE_STOPWORDS & EN_STOPWORDS


def test_return_values_are_only_de_or_en():
    for text in DE + EN + [t for t, _ in MIXED_MAJORITY] + NO_EVIDENCE:
        assert detect(text) in {"de", "en"}
