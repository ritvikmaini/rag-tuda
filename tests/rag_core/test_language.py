from rag_core.domain.language import looks_english_question, strip_foreign


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
