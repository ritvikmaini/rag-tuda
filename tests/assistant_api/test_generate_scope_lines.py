"""The generate node's status-group lines: they follow the sources' typed columns and the card's version-table gate."""

import asyncio
import types
from pathlib import Path

import yaml
from assistant_api.graph.nodes.generate import make_generate
from assistant_api.settings import Messages
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from rag_core.domain.filters import Filter
from rag_core.services.prompts import PromptService, load_profile_prompts
from rag_core.settings import load_profile
from rag_core.settings.card import Gated, load_card
from rag_core.settings.profile import Profile, Roles, StatusGroups

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "profiles" / "example"
TU = ROOT / "profiles" / "tu-informatik"
QUESTION = "Wie viele Credits hat die Thesis?"


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


def sent_user_message(
    docs,
    *,
    gated: bool,
    question: str = QUESTION,
    applied: Filter | None = None,
    shape="text",
    english=False,
    whole=False,
    formats="",
) -> str:
    """The user message the node sends to the model for sources with the given typed columns."""
    seen = []

    class Model:
        async def ainvoke(self, prompt):
            seen.append(prompt)
            return AIMessage(content="Antwort")

    card = load_card(ROOT / "deploy" / "env" / "ci-fake-card.toml")
    deps = types.SimpleNamespace(
        profile=load_profile(Profile, EXAMPLE / "profile.yaml").model_copy(
            update={"status_groups": StatusGroups(compare=True, reminder="differ")}
        ),
        card=card.model_copy(update={"gated": Gated(iterative_retrieval=False, version_table=gated)}),
        messages=Messages.model_validate(yaml.safe_load((TU / "messages.de.yaml").read_text(encoding="utf-8"))),
        messages_en=Messages.model_validate(yaml.safe_load((TU / "messages.en.yaml").read_text(encoding="utf-8")))
        if english
        else None,
        prompts=PromptService(load_profile_prompts(EXAMPLE / "prompts")),
        answer_llm=Model(),
        counter=Counter(),
        roles=load_profile(Roles, TU / "roles.yaml").model_dump(),
    )
    state = {
        "messages": [HumanMessage(content=question)],
        "intent": "lookup",
        "raw_docs": docs,
        "grouped_raw_docs": docs,
        "formatted_docs": "",
        "standalone_question": question,
        "answer_shape": shape,
        "filter": applied,
        "language": "en" if english else "de",
        "formats": formats,
    }
    asyncio.run(make_generate(deps)(state, {}))  # pyright: ignore[reportArgumentType]  # stand-ins for GraphDeps and AgentState
    return "\n\n".join(str(m.content) for m in seen[0]) if whole else seen[0][-1].content


def test_the_version_table_line_follows_the_card_gate_and_the_cited_versions():
    two = [
        doc("a", degree=["bachelor"], programme=["bsc-informatik"], po_version=[2023]),
        doc("b", degree=["bachelor"], programme=["bsc-informatik"], po_version=[2015]),
    ]
    table = "eine Zeile je Version"
    assert table in sent_user_message(two, gated=True)
    assert table not in sent_user_message(two, gated=False)
    assert table not in sent_user_message(two[:1], gated=True)  # one version in the context
    assert table not in sent_user_message(two, gated=True, shape="steps")  # a shaped answer keeps its own form


def test_the_differ_and_compare_lines_run_whatever_the_gate():
    versions = [doc("a", po_version=[2023]), doc("b", po_version=[2015])]
    message = sent_user_message(versions, gated=False)
    assert "PO 2023 sowie PO 2015" in message
    assert "Zeile je Version" not in message
    degrees = [doc("a", degree=["bachelor"]), doc("b", degree=["master"])]
    assert "für wen sie gilt" in sent_user_message(degrees, gated=False)
    named = sent_user_message(degrees[:1], gated=False, applied=Filter(po_version=(2015, 2023)))
    assert "PO 2023 sowie PO 2015" in named
    compare = sent_user_message(degrees, gated=False, question="Was ist der Unterschied zwischen Bachelor und Master?")
    assert "Bachelor-Studierende sowie Master-Studierende" in compare


def test_the_compare_line_agrees_with_the_applied_filter():
    docs = [doc("a", degree=["master"], po_version=[2023])]
    question = "Gilt das auch für Master-Studierende?"
    # the question reads as a comparison of the master alone; the filter holds nothing else, so nothing else is compared
    assert "vergleicht" not in sent_user_message(
        docs, gated=False, question=question, applied=Filter(degree=("master",))
    )
    # a programme filter carries its degree
    msc = Filter(programme=("msc-informatik",))
    assert "vergleicht" not in sent_user_message(docs, gated=False, question=question, applied=msc)
    # no filter: the master is contrasted with the bachelor
    assert "vergleicht Master-Studierende sowie Bachelor-Studierende" in sent_user_message(
        docs, gated=False, question=question
    )


def test_every_shape_names_the_versions_of_its_sources():
    versions = [doc("a", po_version=[2023]), doc("b", po_version=[2015])]
    message = sent_user_message(versions, gated=True, shape="steps")
    assert "PO 2023 sowie PO 2015" in message
    assert "Zeile je Version" not in message


def test_an_english_answer_gets_the_english_lines_and_labels():
    degrees = [doc("a", degree=["bachelor"], po_version=[2023]), doc("b", degree=["master"], po_version=[2015])]
    message = sent_user_message(
        degrees, gated=True, question="What is the difference between bachelor and master?", english=True
    )
    assert "The question compares Bachelor students and Master students." in message
    assert "PO 2023 and PO 2015" in message
    assert "one row per version" in message


def test_an_english_answer_frames_its_sources_and_asks_for_its_form_in_english():
    sources = [doc("a", degree=["master"], po_version=[2023], title="Thesis rules", doc_year=2023)]
    english = sent_user_message(
        sources,
        gated=False,
        question="How many credits does the thesis give?",
        english=True,
        shape="steps",
        whole=True,
        formats="v1",
    )
    assert "There are 1 sources:" in english and "As of: 2023" in english
    assert "Es liegen" not in english and "Stand:" not in english
    assert "- Form: Begin with one sentence. Then describe the procedure as numbered steps" in english
    assert "Beginne mit einem Satz" not in english
    german = sent_user_message(sources, gated=False, shape="steps", whole=True, formats="v1")
    assert "Es liegen 1 Quellen vor:" in german and "Stand: 2023" in german and "Beginne mit einem Satz" in german
