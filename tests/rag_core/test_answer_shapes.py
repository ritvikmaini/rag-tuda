"""Response formats v1: instruction texts, shape detection, frames."""

from types import SimpleNamespace

from assistant_api.sse import frames as fr
from rag_core.domain import answer_shapes as af  # noqa: E402
from rag_core.domain.answer_shapes import turn_variant_problem
from rag_core.domain.history import LastTurn

TABLE = (
    "Es gibt zwei Regelungen [1].\n\n| Gruppe | Urlaub |\n| --- | --- |\n| Beamte | 30 Tage [1] |\n"
    "| Tarif | 30 Tage |\n\nMehr dazu [2]."
)
CHECKLIST = (
    "Vor der Elternzeit ist einiges zu tun [1].\n\n- [ ] Antrag stellen – 7 Wochen vorher – Personalamt [1]\n"
    "- [ ] Elterngeld beantragen [2]\n- [x] Arbeitgeberbescheinigung"
)
STEPS = (
    "So beantragen Sie eine Dienstreise [1].\n\n1. Antrag ausfüllen [1]\n2. Genehmigung einholen – Amtsleitung [2]\n"
    "3. Reise abrechnen"
)
MINDMAP = (
    "Ein Überblick [1].\n\n```mindmap\n# Arbeitszeit\n## Gleitzeit\n- Kernzeit 9 bis 15 Uhr [1]\n- Rahmenzeit [2]\n"
    "## Mobile Arbeit\n- Dienstvereinbarung\n```"
)


def test_texts_have_no_braces_and_restate_the_citation_rule():
    for name, text in list(af.SHAPE_TEXTS.items()) + list(af.VARIANT_TEXTS.items()):
        assert "{" not in text and "}" not in text, name
        assert "Quellennummer" in text, name
        assert text.startswith("- "), name


def test_every_variant_has_a_text_and_the_names_are_fixed():
    assert af.VARIANTS == ("table", "checklist", "steps", "mindmap", "shorter", "longer", "simple", "english")
    assert af.SHAPES == ("text", "table", "checklist", "steps", "mindmap")
    assert set(af.SHAPE_TEXTS) == set(af.SHAPE_VARIANTS)
    assert set(af.VARIANT_TEXTS) == set(af.VARIANTS) - set(af.SHAPE_VARIANTS)


def test_no_instruction_without_formats_or_for_plain_text():
    assert af.format_instruction({"answer_shape": "table"}) == ""
    assert af.format_instruction({"formats": "v2", "answer_shape": "table"}) == ""
    assert af.format_instruction({"formats": "v1", "answer_shape": "text"}) == ""
    assert af.format_instruction({"formats": "v1"}) == ""


def test_shape_then_variant_in_one_instruction():
    assert af.format_instruction({"formats": "v1", "answer_shape": "steps"}) == af.SHAPE_TEXTS["steps"]
    both = af.format_instruction({"formats": "v1", "answer_shape": "table", "variant": "english"})
    assert both == af.SHAPE_TEXTS["table"] + "\n" + af.VARIANT_TEXTS["english"]
    assert af.format_instruction({"formats": "v1", "answer_shape": "bogus", "variant": "bogus"}) == ""


def test_language_follows_the_english_variant_only():
    assert af.answer_language({"formats": "v1", "variant": "english", "language": "Deutsch"}) == "Englisch"
    assert af.answer_language({"variant": "english", "language": "Deutsch"}) == "Deutsch"
    assert af.answer_language({"formats": "v1", "variant": "simple"}) == "Deutsch"
    assert af.answer_language({"language": "Englisch"}) == "Englisch"


def test_entry_route():
    from assistant_api.graph.routes import entry_route

    assert entry_route({"variant": "table"}) == "variant"
    assert entry_route({"variant": None}) == "question"
    assert entry_route({"variant": "drop table"}) == "question"
    assert entry_route({}) == "question"


def test_detect_shape():
    assert af.detect_shape(TABLE) == "table"
    assert af.detect_shape(CHECKLIST) == "checklist"
    assert af.detect_shape(STEPS) == "steps"
    assert af.detect_shape(MINDMAP) == "mindmap"
    assert af.detect_shape("Die Kernzeit endet um 15 Uhr [1].\n1. Hinweis") == "text"
    assert af.detect_shape("| nur | eine Zeile |") == "text"
    assert af.detect_shape("") == "text"
    assert af.detect_shape(None) == "text"


def test_cited_share_counts_structure_lines():
    assert af.cited_share(TABLE, "table") == (1, 2)
    assert af.cited_share(CHECKLIST, "checklist") == (2, 3)
    assert af.cited_share(STEPS, "steps") == (2, 3)
    assert af.cited_share(MINDMAP, "mindmap") == (2, 3)
    assert af.cited_share("Text [1].", "text") == (1, 1)
    assert af.cited_share("", "text") == (0, 0)


def test_log_line():
    assert af.shape_log_line("table", TABLE) == "[Shape] requested=table produced=table cited=1/2"
    assert af.shape_log_line("text", STEPS, "steps") == "[Variant] name=steps requested=text produced=steps cited=2/3"


def test_frames():
    from assistant_api.sse.events import shape_frame

    assert fr.wire(shape_frame("checklist")) == {"type": "shape", "content": "checklist"}
    assert fr.wire(shape_frame("<b>"))["content"] == "text"
    assert fr.wire(fr.variant("table", 1)) == {  # pyright: ignore[reportArgumentType]
        "type": "variant",
        "content": {"name": "table", "ok": True},
    }


def test_with_instruction_appends_to_the_system_message_only():
    from assistant_api.graph.nodes.generate import with_instruction
    from langchain_core.messages import HumanMessage, SystemMessage

    system, human = SystemMessage(content="Regeln."), HumanMessage(content="Frage")
    out = with_instruction([system, human], "- Form: x")
    assert out[0].content == "Regeln.\n- Form: x"
    assert out[1] is human
    assert system.content == "Regeln."
    same = [system, human]
    assert with_instruction(same, "") is same
    only_human = [human]
    assert with_instruction(only_human, "- Form: x") is only_human


def test_shape_reminder_is_the_shape_line_then_a_length_line():
    for shape in af.SHAPE_VARIANTS:
        assert af.shape_reminder({"formats": "v1", "answer_shape": shape}) == af.SHAPE_TEXTS[shape]
        for variant in ("shorter", "longer"):
            assert (
                af.shape_reminder({"formats": "v1", "answer_shape": shape, "variant": variant})
                == af.SHAPE_TEXTS[shape] + "\n" + af.VARIANT_TEXTS[variant]
            )
        # the simple and English lines are not repeated
        for variant in ("simple", "english"):
            assert (
                af.shape_reminder({"formats": "v1", "answer_shape": shape, "variant": variant}) == af.SHAPE_TEXTS[shape]
            )
    assert af.shape_reminder({"formats": "v1", "answer_shape": "text"}) == ""
    assert af.shape_reminder({"formats": "v1", "variant": "shorter"}) == ""
    for shape in af.SHAPE_VARIANTS:
        assert af.shape_reminder({"answer_shape": shape}) == ""


def sent_prompt(**state):
    """The messages the generate node sends to the model (rag-generation, one source) for a state with the format keys."""
    import asyncio
    import types
    from pathlib import Path

    import yaml
    from assistant_api.graph.nodes.generate import make_generate
    from assistant_api.settings import Messages
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, HumanMessage
    from rag_core.services.prompts import PromptService, load_profile_prompts
    from rag_core.settings import load_profile
    from rag_core.settings.profile import Profile

    profile_file = Path(__file__).resolve().parents[2] / "profiles" / "example" / "profile.yaml"
    seen = []

    class Model:
        async def ainvoke(self, prompt):
            seen.append(prompt)
            return AIMessage(content="Antwort")

    class Counter:
        max_model_len, exact = 8192, True

        async def count_text(self, text):
            return len(text) // 2 + 1

        async def count_chat(self, messages):
            return sum(len(m["content"]) // 2 + 8 for m in messages) + 8

    deps = types.SimpleNamespace(
        profile=load_profile(Profile, profile_file),
        messages=Messages.model_validate(
            yaml.safe_load((profile_file.parent / "messages.de.yaml").read_text(encoding="utf-8"))
        ),
        prompts=PromptService(load_profile_prompts(profile_file.parent / "prompts")),
        answer_llm=Model(),
        counter=Counter(),
    )
    doc = Document(page_content="Quelle eins", metadata={"filename": "a.pdf", "section_id": "p1", "url": "https://x/a"})
    full = {
        "messages": [HumanMessage(content="Wie beantrage ich Urlaub?")],
        "intent": "lookup",
        "raw_docs": [doc],
        "grouped_raw_docs": [doc],
        "formatted_docs": "Quelle eins",
        "standalone_question": "Wie beantrage ich Urlaub?",
        **state,
    }
    asyncio.run(make_generate(deps)(full))  # pyright: ignore[reportArgumentType]  # a stand-in for GraphDeps and AgentState
    return seen[0]


def test_the_form_is_repeated_at_the_end_of_the_user_message():
    for shape in af.SHAPE_VARIANTS:
        system, user = sent_prompt(formats="v1", answer_shape=shape)
        assert system.content.endswith("\n" + af.SHAPE_TEXTS[shape])  # the system copy stays
        assert user.content.endswith("\n\n" + af.SHAPE_TEXTS[shape])
        assert user.content.count(af.SHAPE_TEXTS[shape]) == 1


def test_plain_text_and_no_formats_add_nothing_to_the_user_message():
    for state in ({"formats": "v1", "answer_shape": "text"}, {"answer_shape": "steps"}, {}):
        system, user = sent_prompt(**state)
        assert "- Form:" not in system.content + user.content
        assert user.content.endswith("Wie beantrage ich Urlaub?")


def test_the_english_variant_ends_with_the_form_line_then_the_english_reminder():
    system, user = sent_prompt(formats="v1", answer_shape="table", variant="english")
    assert user.content.endswith("\n\n" + af.SHAPE_TEXTS["table"] + "\n\n" + af.ENGLISH_REMINDER)
    assert user.content.count(af.SHAPE_TEXTS["table"]) == 1
    assert system.content.endswith(af.VARIANT_TEXTS["english"])
    # English with plain text: the reminder alone
    _, user = sent_prompt(formats="v1", answer_shape="text", variant="english")
    assert user.content.endswith("Wie beantrage ich Urlaub?\n\n" + af.ENGLISH_REMINDER)


def test_a_length_variant_repeats_its_length_line_after_the_form_line():
    system, user = sent_prompt(formats="v1", answer_shape="steps", variant="longer")
    assert user.content.endswith("\n\n" + af.SHAPE_TEXTS["steps"] + "\n" + af.VARIANT_TEXTS["longer"])
    assert system.content.endswith(af.SHAPE_TEXTS["steps"] + "\n" + af.VARIANT_TEXTS["longer"])
    # plain text with a length variant: the system copy alone
    _, user = sent_prompt(formats="v1", answer_shape="text", variant="longer")
    assert user.content.endswith("Wie beantrage ich Urlaub?")


# ---- grounding of steps and checklists


def test_steps_and_checklists_come_only_from_the_sources():
    # the grounding stays, the way out into sentences goes (it turned real procedures into prose)
    for name in ("steps", "checklist"):
        text = af.SHAPE_TEXTS[name]
        assert "die in den Quellen stehen" in text, name
        assert "Sätzen" not in text and "stattdessen" not in text, name


def test_every_table_row_names_its_source():
    assert "letzte Spalte Quelle" in af.SHAPE_TEXTS["table"]


def test_a_chosen_form_gets_the_same_instruction_as_an_automatic_one():
    # the "does not fit" sentence never produced its explanation and gave the model a way out of the form
    assert not hasattr(af, "CHOSEN_FORM_TEXT")
    for shape in af.SHAPE_VARIANTS:
        chosen = af.format_instruction({"formats": "v1", "answer_shape": shape, "variant": shape})
        assert chosen == af.SHAPE_TEXTS[shape]


def test_english_is_asked_for_in_so_many_words():
    assert "vollständig auf Englisch" in af.VARIANT_TEXTS["english"]
    assert af.language_reminder({"formats": "v1", "variant": "english"}) == af.ENGLISH_REMINDER
    assert "English" in af.ENGLISH_REMINDER
    assert af.language_reminder({"variant": "english"}) == ""
    assert af.language_reminder({"formats": "v1", "variant": "simple"}) == ""
    assert af.language_reminder({"formats": "v1"}) == ""
    # an English answer needs no formats: the analysis' language or an English-looking question is enough
    assert af.language_reminder({"language": "Englisch"}) == af.ENGLISH_REMINDER
    english = {"messages": [SimpleNamespace(content="How do I apply for leave?")]}
    assert af.language_reminder(english) == af.ENGLISH_REMINDER and af.answer_language(english) == "Englisch"
    german = {"messages": [SimpleNamespace(content="Wie beantrage ich Urlaub?")]}
    assert af.language_reminder(german) == "" and af.answer_language(german) == "Deutsch"
    assert af.answer_language({**english, "language": "Französisch"}) == "Französisch"


def test_with_reminder_ends_the_last_user_message():
    from assistant_api.graph.nodes.generate import with_reminder
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    system, old, answer, last = (
        SystemMessage(content="Regeln."),
        HumanMessage(content="Alt"),
        AIMessage(content="A"),
        HumanMessage(content="Frage"),
    )
    out = with_reminder([system, old, answer, last], "Answer in English.")
    assert out[-1].content == "Frage\n\nAnswer in English."
    assert out[:3] == [system, old, answer] and last.content == "Frage"
    same = [system, last]
    assert with_reminder(same, "") is same
    no_user = [system]
    assert with_reminder(no_user, "x") is no_user


def test_turn_variant_problem_keeps_its_reasons():
    t = "11111111-1111-4111-8111-111111111111"
    good = LastTurn(turn_id=t, answered=True, has_sources=True)
    assert turn_variant_problem("poster", t, good) == "unknown variant"
    assert turn_variant_problem("table", None, good) == "no turn id"
    assert turn_variant_problem("table", t, None) == "no answer to change"
    assert turn_variant_problem("table", t, LastTurn(t, False, True)) == "no answer to change"
    assert turn_variant_problem("table", "22222222-2222-4222-8222-222222222222", good) == "not the last question"
    assert turn_variant_problem("table", t, LastTurn(t, True, False)) == "no sources"
    assert turn_variant_problem("shorter", t, good) == ""
