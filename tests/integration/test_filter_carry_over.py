"""The applied filter over turns: a follow-up that names nothing keeps the previous turn's filter, also when its analysis
failed; a variant runs on the filter of the turn it changes; an empty filtered result says which filter was applied; the
"understood" step shows it. TU leg: the PO 2015 and PO 2023 sections of the mini-site's B.Sc. hub and the PO 2023 PDF tell the
filtered from the unfiltered turns, the university-wide regulation and the office pages (no PO) pass every filter.
Black box: HTTP, SQL as rag_app and the fake model's request log."""

import uuid
from typing import Any

import psycopg
from rag_testing import harness
from rag_testing.minisite import pages

HUB = "https://www.informatik.tu-darmstadt.de" + pages.BSC_HUB + ".de.jsp"
PO_2015, PO_2023 = f"{HUB}#pruefungsordnung_2015", f"{HUB}#pruefungsordnung_2023"
PO_2023_PDF = "https://www.informatik.tu-darmstadt.de" + pages.PO_2023_PDF

THESIS_2015 = "Wie lange dauert die Bearbeitung der Bachelor-Thesis nach der PO 2015?"
FOLLOW_UP = "Und wie lange dauert die Bearbeitung der Bachelor-Thesis genau?"
CHIPS = {"degree": "bachelor", "programme": "bsc-informatik", "po": "2015"}
FAIL_ANALYSIS = {"purpose": "analysis", "match": FOLLOW_UP, "fail": 500}
# a filter no document of the mini-site passes: the Modulhandbuch has no document there, and every document has a regulation
MODULE_HANDBOOK = "Wie viele Credit Points hat die Abschlussarbeit laut Modulhandbuch?"
MASTER_2015 = {"degree": "master", "programme": "msc-informatik", "po": "2015"}
SHORTER = "shorter"


def new_id() -> str:
    return str(uuid.uuid4())


def urls(reply: harness.Reply) -> set[str]:
    """The urls of the documents of the docs frame."""
    return {d["metadata"]["url"] for d in reply.first("docs")}


def applied(reply: harness.Reply) -> dict[str, Any]:
    return next(f["content"] for f in reply.frames if f["type"] == "status" and f["content"]["step"] == "understood")[
        "applied_filter"
    ]


def stored(db: psycopg.Connection[Any], user: str) -> list[Any]:
    return [
        r[0]
        for r in db.execute(
            "SELECT applied_filter FROM chat.turns WHERE conversation_id = %s ORDER BY created_at", (user,)
        )
    ]


def test_followup_after_failed_analysis_keeps_filter(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(FAIL_ANALYSIS)
    first = assistant.ask(THESIS_2015, user, turn_id=new_id())  # the PO is named in the text: no chips needed
    assert PO_2015 in urls(first) and not {PO_2023, PO_2023_PDF} & urls(first)
    second = assistant.ask(FOLLOW_UP, user, turn_id=new_id())
    asked = llm.calls("analysis")[-1]["question"]
    assert asked == FOLLOW_UP  # the analysis was asked and failed (fallback), not skipped
    assert second.path == harness.LOOKUP
    assert applied(second)["po_version"] == [2015] and applied(second)["po_source"] == "carried"
    assert not {PO_2023, PO_2023_PDF} & urls(second)
    # without the earlier turn the same question reaches the PO 2023 sources: the carry-over is what narrowed it
    control = assistant.ask(FOLLOW_UP, "harness-" + new_id()[:12])
    assert {PO_2023, PO_2023_PDF} & urls(control)


def test_variant_keeps_filter_and_po_line(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str, db: psycopg.Connection[Any]
) -> None:
    turn = new_id()
    llm.script(
        {"purpose": "answer", "reply": "Die Dauer steht in den Quellen " + " ".join(f"[{n}]" for n in range(1, 10))}
    )
    original = assistant.ask(THESIS_2015, user, turn_id=turn, formats="v1")
    assert original.path == harness.LOOKUP_FORMATS
    before = stored(db, user)
    assert before[0]["po_version"] == [2015]
    variant = assistant.ask(THESIS_2015, user, formats="v1", variant=SHORTER, variant_of=turn)
    assert variant.path == harness.VARIANT and variant.first("variant") == {"name": SHORTER, "ok": True}
    # the variant answers from the stored sources of the turn, all of them PO 2015 or without a PO; the turn keeps its filter
    assert {c["url"] for c in variant.first("citations")} <= urls(original) and PO_2015 in urls(original)
    assert not {PO_2023, PO_2023_PDF} & urls(original)
    assert stored(db, user) == before
    # the docs frame hands the page the typed values, and the answer and its variants end with the line naming the version
    assert {"regulation", "degree", "programme", "po_version", "lang"} <= next(
        d["metadata"] for d in original.first("docs") if d["metadata"]["url"] == PO_2015
    ).keys()
    assert original.text.endswith("Grundlage: Prüfungsordnung 2015 des B.Sc. Informatik")
    assert variant.text.endswith("Grundlage: Prüfungsordnung 2015 des B.Sc. Informatik")
    translated = assistant.ask(THESIS_2015, user, formats="v1", variant="english", variant_of=turn)
    assert translated.first("variant") == {"name": "english", "ok": True}
    assert translated.text.endswith("Based on: Examination regulations 2015 of the B.Sc. Informatik")
    # and the next question still starts from that filter
    llm.script(FAIL_ANALYSIS)
    follow_up = assistant.ask(FOLLOW_UP, user, turn_id=new_id(), formats="v1")
    assert applied(follow_up)["po_version"] == [2015]


def test_empty_filter_result_names_the_filter(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    german = assistant.ask(MODULE_HANDBOOK, user, **MASTER_2015)
    assert not {"docs", "citations"} & set(german.types)  # the search found nothing
    assert german.text.endswith("Für den M.Sc. Informatik, PO 2015, habe ich nichts gefunden.")
    english = assistant.ask("How many credit points does the thesis give in the MHB?", user, **MASTER_2015)
    assert english.text.endswith("I found nothing for the M.Sc. Informatik, PO 2015.")
    assert [c["purpose"] for c in llm.calls()][-1] == "no_docs"  # the model still wrote the answer; the line follows it


def test_status_frame_carries_applied_filter(ready: None, assistant: harness.Assistant, user: str) -> None:
    chipped = assistant.ask(harness.OFFICE_HOURS, user, **CHIPS)
    assert applied(chipped) == {
        "degree": ["bachelor"],
        "programme": ["bsc-informatik"],
        "po_version": [2015],
        "regulation": None,
        "date": None,
        "include_archived": True,  # a PO filter keeps the archived versions in
        "po_source": "chip",
        "from_chips": ["degree", "po_version", "programme"],  # not carried once the chips are cleared
    }
    plain = assistant.ask(harness.OFFICE_HOURS, "harness-" + new_id()[:12])
    assert applied(plain) == {
        "degree": None,
        "programme": None,
        "po_version": None,
        "regulation": None,
        "date": None,
        "include_archived": False,
        "po_source": None,
        "from_chips": [],
    }
