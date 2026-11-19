"""The TU leg's scripted conversations, the page's inputs: each test asks the real backend, over the index the build made from the
mini-site, the questions of one case (a lookup, a follow-up, the chips, an English question, the iterative round, a direct
reply, an empty filtered result, each answer shape, the versions of an answer, a busy server) and checks the frames it gets.

HARNESS_RECORD=1 keeps what the page needs of them: the stream of every turn in the native form (tests/contracts/fixtures/
transcripts_v1/rec-<plan>_<conversation>_t<n>.sse), the legacy view of it with a sidecar and a plan file in data/transcripts/
(`python -m rag_testing.redact data/transcripts/raw data/transcripts/plans tests/contracts/fixtures/transcripts` writes the
committed, redacted copies) and the profile the chips read (web/page/tests/fixtures/profile.json). A run without it records
nothing. Black box: HTTP and the fake model's script."""

import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
from rag_testing import harness

ROOT = Path(__file__).resolve().parents[3]
RECORD = os.environ.get("HARNESS_RECORD") == "1"
LOCAL = ROOT / "data" / "transcripts"
NATIVE = ROOT / "tests" / "contracts" / "fixtures" / "transcripts_v1"
PROFILE_JSON = ROOT / "web" / "page" / "tests" / "fixtures" / "profile.json"
plans: dict[str, dict[str, list[str]]] = {}  # plan -> conversation -> the path of each of its turns

THESIS = harness.THESIS
THESIS_2015 = "Wie lange dauert die Bearbeitung der Bachelor-Thesis nach der PO 2015?"
FOLLOW_UP = "Und wie lange dauert die Bearbeitung der Bachelor-Thesis genau?"
COMPARE = "Was unterscheidet die PO 2015 von der PO 2023 bei der Bearbeitung der Bachelor-Thesis?"
SIGN_UP = "Wie melde ich mich zu einer Prüfung an?"
FAILED = "Ich bin in einer Klausur zum zweiten Mal durchgefallen."
RETAKE_EN = "How many times can I retake a failed exam?"
MODULE_HANDBOOK = "Wie viele Credit Points hat die Abschlussarbeit laut Modulhandbuch?"
CHIPS = {"degree": "bachelor", "programme": "bsc-informatik", "po": "2015"}
MASTER_2015 = {"degree": "master", "programme": "msc-informatik", "po": "2015"}

THESIS_ANSWER = "Die Bearbeitungszeit der Bachelor-Thesis steht in der Prüfungsordnung [1] und im Prüfungsplan [2]."
TABLE = (
    "Die Fristen im Überblick [1].\n\n| Regelung | Frist |\n| --- | --- |\n| Bearbeitung der Thesis | sechs Monate [1] |\n"
    "| Verlängerung | drei Monate [2] |"
)
STEPS = "So melden Sie sich an [1].\n\n1. Im Campus-Portal anmelden [1]\n2. Prüfung auswählen [2]\n3. Anmeldung bestätigen [2]"
CHECKLIST = (
    "Nach einem zweiten Fehlversuch ist einiges zu klären [1].\n\n- [ ] Bescheid des Prüfungsausschusses abwarten [1]\n\n"
    "- [ ] Antrag auf Nachteilsausgleich prüfen [2]\n\n- [ ] Beratung im Studienbüro vereinbaren [2]"
)
MINDMAP = (
    "Ein Überblick über die Thesis [1].\n\n```mindmap\n# Bachelor-Thesis\n## Bearbeitung\n- sechs Monate [1]\n"
    "## Verlängerung\n- drei Monate [2]\n```"
)
SHORTER = "Sechs Monate [1]."
ENGLISH = "The thesis takes six months [1]."


def save(plan: str, cid: str, reply: harness.Reply, path: str, **params: Any) -> None:
    """Keep one turn (see the module text). `path` names the case for the plan file; `params` are the request's."""
    turns = plans.setdefault(plan, {}).setdefault(cid, [])
    turns.append(path)
    if not RECORD:
        return
    stem = f"{cid}_t{len(turns)}"
    NATIVE.mkdir(parents=True, exist_ok=True)
    (NATIVE / f"rec-{plan}_{stem}.sse").write_text(reply.raw, encoding="utf-8")
    raw = LOCAL / "raw" / plan
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{stem}.raw").write_text(harness.serialise(reply.frames), encoding="utf-8")
    (raw / f"{stem}.json").write_text(
        json.dumps({"status": reply.status, "params": params}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (LOCAL / "plans").mkdir(parents=True, exist_ok=True)
    for name, conversations in plans.items():
        spec = {"conversations": [{"id": c, "turns": [{"path": p} for p in t]} for c, t in conversations.items()]}
        (LOCAL / "plans" / f"plan_{name}.json").write_text(json.dumps(spec), encoding="utf-8")


def ask(
    assistant: harness.Assistant, plan: str, cid: str, question: str, user: str, path: str = "lookup", **params: str
) -> harness.Reply:
    """One question of conversation `cid`, in the form the page asks (answer formats on), kept. It must complete."""
    reply = assistant.ask(question, user, formats="v1", **params)
    assert reply.status == 200 and reply.stream[-1]["content"]["outcome"] == "completed", reply.raw
    save(plan, cid, reply, path, question=question, formats="v1", **params)
    return reply


def understood(reply: harness.Reply) -> dict[str, Any]:
    return next(f["content"] for f in reply.frames if f["type"] == "status" and f["content"]["step"] == "understood")


def test_the_profile_the_chips_read(tokens: dict[str, str]) -> None:
    auth = {"Authorization": "Bearer " + tokens["BACKEND_REFRESH_TOKEN"]}
    with httpx.Client(base_url=harness.BACKEND, timeout=30) as client:
        body = client.get("/api/profile", headers=auth).json()
    assert body["degrees"] and body["programmes"] and body["answer"]["remote"] is False
    if RECORD:
        PROFILE_JSON.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_JSON.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def test_a_lookup_ends_with_the_line_naming_the_regulation(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script({"purpose": "answer", "reply": THESIS_ANSWER})
    reply = ask(assistant, "plain", "thesis", THESIS, user)
    assert reply.path == harness.LOOKUP_FORMATS and reply.first("shape") == "text"
    assert reply.text.startswith(THESIS_ANSWER) and "\n\nGrundlage: " in reply.text
    assert understood(reply)["applied_filter"]["degree"] == ["bachelor"]


def test_a_follow_up_keeps_the_filter_of_the_question_before_it(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script({"purpose": "answer", "reply": THESIS_ANSWER})
    first = ask(assistant, "plain", "carry", THESIS_2015, user)
    second = ask(assistant, "plain", "carry", FOLLOW_UP, user)
    assert understood(first)["applied_filter"]["po_source"] == "text"
    assert understood(second)["applied_filter"]["po_version"] == [2015]
    assert understood(second)["applied_filter"]["po_source"] == "carried"


def test_the_chips_narrow_the_search_and_are_shown_as_the_applied_filter(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script({"purpose": "answer", "reply": THESIS_ANSWER})
    reply = ask(assistant, "plain", "chips", THESIS, user, **CHIPS)
    applied = understood(reply)["applied_filter"]
    assert (applied["po_version"], applied["po_source"]) == ([2015], "chip")
    assert applied["from_chips"] == ["po_version", "programme"]  # the degree is in the question


def test_an_english_question_gets_the_english_lines(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    analysis = {"language": "en", "search_queries": ["Wiederholung nicht bestandene Prüfung"]}
    llm.script(
        {"purpose": "analysis", "match": RETAKE_EN, "reply": analysis},
        {"purpose": "answer", "reply": "You may retake a failed exam twice [1]."},
    )
    reply = ask(assistant, "plain", "english", RETAKE_EN, user, lang="en")
    assert reply.path == harness.LOOKUP_FORMATS and reply.text.startswith("You may retake a failed exam twice [1].")


def test_the_iterative_round_runs_once_for_a_question_that_needs_more_documents(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(
        {"purpose": "analysis", "match": COMPARE, "reply": {"needs_more": True}},
        {"purpose": "follow_up", "reply": {"queries": ["Verlängerung der Bearbeitungszeit"], "done": False}},
        {"purpose": "answer", "reply": THESIS_ANSWER},
    )
    reply = ask(assistant, "plain", "iterative", COMPARE, user)
    steps = [f["content"]["step"] for f in reply.frames if f["type"] == "status"]
    assert steps == ["analysing", "understood", "searching", "searching_more", "found", "writing"]
    assert [c["purpose"] for c in llm.calls()] == ["analysis", "follow_up", "answer"]
    assert understood(reply)["applied_filter"]["po_version"] == [2015, 2023]


def test_a_direct_reply_has_no_search(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(harness.OUT_OF_SCOPE)
    reply = ask(assistant, "plain", "direct", harness.JOKE, user, "direct")
    assert reply.path == harness.DIRECT_FORMATS


def test_an_empty_filtered_result_names_the_filter(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    reply = ask(assistant, "plain", "nodocs", MODULE_HANDBOOK, user, "no_docs", **MASTER_2015)
    assert not {"docs", "citations"} & set(reply.types)
    assert reply.text.endswith("Für den M.Sc. Informatik, PO 2015, habe ich nichts gefunden.")


def test_a_question_over_the_limit_is_answered_with_the_hint(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    question = "x" * 2001
    reply = assistant.ask(question, user, formats="v1")
    assert reply.stream[-1]["content"]["outcome"] == "too_long" and reply.text == harness.TOO_LONG
    save("plain", "toolong", reply, "too_long", question=question, formats="v1")


def test_a_table_steps_and_a_checklist_each_get_their_shape_and_the_line_after_it(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(
        {"purpose": "analysis", "match": THESIS, "reply": {"shape": "table"}},
        {"purpose": "analysis", "match": SIGN_UP, "reply": {"shape": "steps"}},
        {"purpose": "analysis", "match": FAILED, "reply": {"intent": "situation"}},
        {"purpose": "answer", "match": THESIS, "reply": TABLE},
        {"purpose": "answer", "match": SIGN_UP, "reply": STEPS},
        {"purpose": "answer", "match": FAILED, "reply": CHECKLIST},
    )
    for cid, question, shape, answer in (
        ("table", THESIS, "table", TABLE),
        ("steps", SIGN_UP, "steps", STEPS),
        ("checklist", FAILED, "checklist", CHECKLIST),
    ):
        reply = ask(assistant, "formats", cid, question, user + cid)
        assert reply.path == harness.LOOKUP_FORMATS and reply.first("shape") == shape, cid
        body, _, line = reply.text.rpartition("\n\n")
        assert body == answer and line.startswith("Grundlage: "), cid


def test_the_versions_of_an_answer_and_a_refused_variant(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(
        {"purpose": "answer", "system": "```mindmap", "reply": MINDMAP},
        {"purpose": "answer", "system": "Kürze die Antwort auf das Wichtigste", "reply": SHORTER},
        {"purpose": "answer", "match": "Answer in English only", "reply": ENGLISH},
        {"purpose": "answer", "match": THESIS, "reply": THESIS_ANSWER},
    )
    turn = str(uuid.uuid4())
    original = ask(assistant, "formats", "variants", THESIS, user, turn_id=turn)
    assert original.text.startswith(THESIS_ANSWER)

    def make(name: str, key: str, ok: bool, text: str | None, shape: str | None) -> harness.Reply:
        """The variant (or the switch to a stored version) `name` of the original, kept as the next turn."""
        reply = assistant.ask(THESIS, user, formats="v1", variant_of=turn, **{key: name})
        assert reply.stream[-2]["content"] == {"name": name, "ok": ok}, reply.raw
        assert text is None or reply.text.startswith(text) and reply.first("shape") == shape, name
        save("formats", "variants", reply, key, question=THESIS, formats="v1", variant_of="<turn>", **{key: name})
        return reply

    make("shorter", "variant", True, SHORTER, "text")
    make("english", "variant", True, ENGLISH, "text")
    make("mindmap", "variant", True, MINDMAP, "mindmap")
    make("original", "variant_select", True, None, None)
    lost = assistant.ask(THESIS, user, formats="v1", variant_of=turn, variant_select="table")  # never made
    assert lost.stream[-1]["content"] == {"outcome": "refused"} and lost.stream[1]["code"] == "no_version"
    save(
        "formats",
        "variants",
        lost,
        "version_missing",
        question=THESIS,
        formats="v1",
        variant_select="table",
        variant_of="<turn>",
    )

    refused_turn = str(uuid.uuid4())
    ask(assistant, "formats", "refused", THESIS, user + "-r", turn_id=refused_turn)
    off = assistant.ask(THESIS, user + "-r", variant="shorter", variant_of=refused_turn)  # without formats: no variants
    assert off.stream[-1]["content"] == {"outcome": "refused"} and harness.REFUSED in off.text
    save("formats", "refused", off, "refused", question=THESIS, variant="shorter", variant_of="<turn>")


def test_a_busy_server_answers_the_second_asker_at_once(
    ready: None, assistant: harness.Assistant, llm: harness.FakeLLM, user: str
) -> None:
    llm.script(
        {"purpose": "answer", "match": THESIS, "reply": THESIS_ANSWER + " eins zwei drei.", "token_delay_s": 0.8}
    )
    thread, out = harness.in_thread(lambda: assistant.ask(THESIS, user, formats="v1"))
    harness.wait_until(lambda: bool(llm.calls("answer")))
    busy = assistant.ask(harness.OFFICE_HOURS, user + "-b", formats="v1")
    thread.join()
    assert busy.stream[-1]["content"]["outcome"] == "busy" and harness.MSG_BUSY in busy.text
    answered = out["result"]
    assert answered.stream[-1]["content"]["outcome"] == "completed"
    save("busy", "busy-1", answered, "lookup", question=THESIS, formats="v1")
    save("busy", "busy-2", busy, "busy", question=harness.OFFICE_HOURS, formats="v1")
