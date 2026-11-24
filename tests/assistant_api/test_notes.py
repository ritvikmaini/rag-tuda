"""The lines the generate node appends to an answer in code: the PO versions it rests on, the visa disclaimer and the note on
an English information translation. They follow the metadata of the sources the answer cites and the answer's language."""

import asyncio
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

import yaml
from assistant_api.graph.nodes.generate import make_generate
from assistant_api.settings import Messages
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from rag_core.domain.filters import Filter
from rag_core.domain.history import TS_KEY
from rag_core.services.prompts import PromptService, load_profile_prompts
from rag_core.settings import load_profile
from rag_core.settings.card import load_card
from rag_core.settings.profile import Profile, Roles

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "profiles" / "example"
TU = ROOT / "profiles" / "tu-informatik"
BSC = {"degree": ["bachelor"], "programme": ["bsc-informatik"], "regulation": "po"}
VISA = (
    "Keine Rechtsberatung. Verbindlich entscheidet die zuständige Ausländerbehörde (Stadt Darmstadt oder Landkreis "
    "Darmstadt-Dieburg, je nach Wohnort)."
)


class Counter:
    max_model_len, exact = 8192, True

    async def count_text(self, text):
        return len(text) // 2 + 1

    async def count_chat(self, messages):
        return sum(len(m["content"]) // 2 + 8 for m in messages) + 8


def doc(section_id: str, **columns) -> Document:
    return Document(
        page_content=f"Quelle {section_id}",
        metadata={"section_id": section_id, "url": f"https://x/{section_id}", "cross_encoder_score": 0.9, **columns},
    )


def answered(
    docs, answer: str, *, applied: Filter | None = None, english=False, profile: Path = TU, earlier=(), seen=None
) -> str:
    """The answer the node sends for the given sources when the model writes `answer`; `earlier` is the stored conversation
    before the question, `seen` collects the prompts the model gets."""

    class Model:
        async def ainvoke(self, prompt):
            if seen is not None:
                seen.append(prompt)
            return AIMessage(content=answer)

    def texts(language: str) -> Messages | None:
        path = profile / f"messages.{language}.yaml"
        return Messages.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))) if path.exists() else None

    deps = types.SimpleNamespace(
        profile=load_profile(Profile, EXAMPLE / "profile.yaml"),
        card=load_card(ROOT / "deploy" / "env" / "ci-fake-card.toml"),
        messages=texts("de"),
        messages_en=texts("en"),
        prompts=PromptService(load_profile_prompts(EXAMPLE / "prompts")),
        answer_llm=Model(),
        counter=Counter(),
        roles=load_profile(Roles, TU / "roles.yaml").model_dump(),
    )
    question = "Wie lange dauert die Thesis?"
    state = {
        "messages": [*earlier, HumanMessage(content=question)],
        "intent": "lookup",
        "raw_docs": docs,
        "grouped_raw_docs": docs,
        "formatted_docs": "",
        "standalone_question": question,
        "answer_shape": "text",
        "filter": applied,
        "language": "en" if english else "de",
    }
    result = asyncio.run(make_generate(deps)(state, {}))  # pyright: ignore[reportArgumentType]  # stand-ins for GraphDeps and AgentState
    return result["generation"]


def test_po_line_lists_each_cited_version():
    docs = [doc("a", **BSC, po_version=[2023]), doc("b", **BSC, po_version=[2015]), doc("c", **BSC, po_version=[2020])]
    one = answered(docs, "Sechs Monate [1].")
    assert one.endswith("Sechs Monate [1].\n\nGrundlage: Prüfungsordnung 2023 des B.Sc. Informatik")
    # one entry per cited version, newest first; the version of a source the answer does not cite is left out
    both = answered(docs, "Sechs Monate [1], in der PO 2015 vier [2].")
    assert both.endswith("\n\nGrundlage: PO 2023 und PO 2015 des B.Sc. Informatik")
    # a programme valid for fewer versions than the source names keeps only its own
    shared = doc("d", programme=["msc-computer-science", "msc-informatik"], regulation="po", po_version=[2015, 2023])
    assert answered([shared], "Antwort [1].").endswith(
        "Grundlage: Prüfungsordnung 2023 des M.Sc. Computer Science; PO 2023 und PO 2015 des M.Sc. Informatik"
    )
    # English twin, in the answer's language
    english = answered(docs, "Six months [1], four in the PO 2015 [2].", english=True)
    assert english.endswith("\n\nBased on: PO 2023 and PO 2015 of the B.Sc. Informatik")
    # a PO the filter derived from a start term is a default: the line says "as a rule"
    term = Filter(po_version=(2023,), po_source="term")
    assert answered(docs, "Sechs Monate [1].", applied=term).endswith(
        "Grundlage (in der Regel): Prüfungsordnung 2023 des B.Sc. Informatik"
    )
    chip = Filter(po_version=(2023,), po_source="chip")
    assert "in der Regel" not in answered(docs, "Sechs Monate [1].", applied=chip)


def test_no_po_line_without_version_bound_source():
    plain = doc("a", regulation="apb")  # NULL scope: applies to all
    bound = doc("b", **BSC, po_version=[2023])
    assert answered([plain, bound], "Antwort [1].") == "Antwort [1]."
    assert answered([plain, bound], "Antwort ohne Quelle.") == "Antwort ohne Quelle."  # nothing cited
    assert answered([plain, bound], "Antwort [1].", english=True) == "Antwort [1]."
    # the example profile words no lines
    assert answered([bound], "Antwort [1].", profile=EXAMPLE) == "Antwort [1]."


def test_visa_disclaimer_on_residence_source():
    residence = doc("a", regulation="residence")
    other = doc("b", regulation="info")
    assert answered([residence, other], "Antrag beim Amt [1].").endswith("Antrag beim Amt [1].\n\n" + VISA)
    assert answered([residence, other], "Sprechzeiten [2].") == "Sprechzeiten [2]."  # the residence source is not cited
    english = answered([residence], "Apply at the office [1].", english=True)
    assert english.endswith(
        "\n\nNo legal advice. The competent immigration authority (City of Darmstadt or District of "
        "Darmstadt-Dieburg, depending on your place of residence) has the final say."
    )
    # beside the PO line: the version first, the disclaimer last
    both = answered([residence, doc("b", **BSC, po_version=[2023])], "Visum [1], Frist [2].")
    assert both.endswith("Grundlage: Prüfungsordnung 2023 des B.Sc. Informatik\n\n" + VISA)


def test_non_binding_note_on_english_translation():
    translation = doc("a", regulation="po", lang="en", programme=["bsc-informatik"], po_version=[2023])
    original = doc("b", regulation="po", lang="de", programme=["bsc-informatik"], po_version=[2023])
    guide = doc("c", regulation="info", lang="en")  # an English page that is no translation of a binding text
    english = answered([translation], "Six months [1].", english=True)
    assert "\n\nNot legally binding: the cited source is an English information translation." in english
    german = answered([translation], "Sechs Monate [1].")  # also for a German answer, in German
    assert "\n\nNicht rechtsverbindlich: Die zitierte Quelle ist eine englische Informationsübersetzung." in german
    assert "rechtsverbindlich" not in answered([original], "Sechs Monate [1].")
    assert "rechtsverbindlich" not in answered([guide], "Sechs Monate [1].")
    assert "rechtsverbindlich" not in answered(
        [translation, original], "Sechs Monate [2]."
    )  # cites the German original only


def test_a_line_the_answer_already_has_is_not_added_again():
    docs = [doc("a", **BSC, po_version=[2023]), doc("b", regulation="residence")]
    written = (
        "Sechs Monate [1], Antrag beim Amt [2].\n\nGrundlage:  Prüfungsordnung 2023\ndes B.Sc. Informatik\n\n" + VISA
    )
    assert answered(docs, written) == written
    # only the line the model left out is added
    partly = "Sechs Monate [1], Antrag beim Amt [2].\n\n" + VISA
    assert answered(docs, partly) == partly + "\n\nGrundlage: Prüfungsordnung 2023 des B.Sc. Informatik"


def test_the_history_the_model_gets_has_the_appended_lines_removed():
    asked = datetime.now(timezone.utc).isoformat()
    german = "Sechs Monate [1].\n\nGrundlage: PO 2023 und PO 2015 des B.Sc. Informatik\n\n" + VISA
    english = "Six months [1].\n\nBased on (as a rule): Examination regulations 2023 of the B.Sc. Informatik"
    earlier = [
        HumanMessage(content="Erste Frage", additional_kwargs={TS_KEY: asked}),
        AIMessage(content=german),
        HumanMessage(content="Second question", additional_kwargs={TS_KEY: asked}),
        AIMessage(content=english),
    ]
    seen: list = []
    answered([doc("a", regulation="info")], "Antwort [1].", earlier=earlier, seen=seen)
    sent = "\n".join(str(m.content) for m in seen[0])
    assert "Sechs Monate" in sent and "Six months" in sent  # the answers stay in the history
    assert "Grundlage" not in sent and "Based on" not in sent and "Keine Rechtsberatung" not in sent
    # a paragraph in the middle of an answer, or one the profile does not word as a line, stays
    kept = AIMessage(content="Grundlage: das steht im Text.\n\nWeiter geht es [1].")
    seen.clear()
    answered([doc("a", regulation="info")], "Antwort [1].", earlier=[earlier[0], kept], seen=seen)
    assert "Grundlage: das steht im Text." in "\n".join(str(m.content) for m in seen[0])


def test_a_failing_metadata_lookup_costs_the_lines_not_the_answer(monkeypatch: pytest.MonkeyPatch):
    from assistant_api.graph.nodes import generate

    class Source(dict):
        def get(self, *args):
            raise RuntimeError("broken")

    real = generate.fit_context

    async def broken(*args, **kwargs):
        fitted = await real(*args, **kwargs)
        return {**fitted, "sources": [Source(s) for s in fitted["sources"]]}

    monkeypatch.setattr(generate, "fit_context", broken)
    assert answered([doc("a", **BSC, po_version=[2023])], "Antwort [1].") == "Antwort [1]."


def test_grouped_citations_get_the_po_line_and_the_visa_disclaimer():
    bound = doc("a", **BSC, po_version=[2023])
    residence = doc("b", regulation="residence")
    other = doc("c", regulation="info")
    for written in ("[1, 2]", "[1,2]", "[1][2]"):
        answer = answered([bound, residence, other], f"Frist und Antrag {written}.")
        assert answer.endswith("Grundlage: Prüfungsordnung 2023 des B.Sc. Informatik\n\n" + VISA), written
    # a group that holds only an uncited source adds nothing
    assert answered([bound, residence, other], "Antwort [3, 3].") == "Antwort [3, 3]."
