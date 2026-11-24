"""The recorded streams and the fixtures the page reads beside them.

tests/contracts/fixtures/transcripts_v1/ holds the streams exactly as the backend wrote them (`just harness tu`, see
tests/integration/tu/test_conversations.py and test_stream_v1.py): the named ones carry a sidecar for the page's replay, the
`rec-*` ones are every scripted turn. fixtures/transcripts/ holds the same turns in the legacy view with their document text
masked (rag_testing.redact). web/page/tests/fixtures/ holds what the page's own tests read: the profile the chips offer, the
shared answer shapes, the backend's codes and the frames an independent reader finds in every native stream. UPDATE_GOLDEN=1
rewrites that last file."""

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from assistant_api.sse import frames as fr
from evals import checks
from rag_core.domain.source_meta import CITATION_GROUP
from evals.sse import parse_sse
from pydantic import TypeAdapter
from rag_testing import harness, redact

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "contracts" / "fixtures"
NATIVE, REDACTED = FIXTURES / "transcripts_v1", FIXTURES / "transcripts"
PAGE = ROOT / "web" / "page" / "tests" / "fixtures"
TU = ROOT / "profiles" / "tu-informatik"
FRAME = TypeAdapter(fr.Frame)
NATIVE_FILES = sorted(NATIVE.glob("*.sse"))
SIDECARS = sorted(NATIVE.glob("*.json"))
REDACTED_NAMES = sorted(p.stem for p in REDACTED.glob("*.sse"))


def stream(path: Path) -> list[dict[str, Any]]:
    events, rest = parse_sse(path.read_text(encoding="utf-8"))
    assert not rest.strip(), path.name  # a complete frame ends each recording (a client stop may stop between frames)
    return events


def test_the_recordings_exist() -> None:
    assert len(NATIVE_FILES) >= 30 and len(SIDECARS) == 10 and len(REDACTED_NAMES) >= 20


@pytest.mark.parametrize("file", NATIVE_FILES, ids=lambda p: p.stem)
def test_a_native_recording_is_a_format_1_stream(file: Path) -> None:
    text = file.read_text(encoding="utf-8")
    events = stream(file)
    for event in events:
        FRAME.validate_python(event)
    assert events[0]["type"] == "meta" and events[0]["content"]["deadline_s"] >= events[0]["content"]["idle_s"] > 0
    # every line is a frame or blank, or the comment the guard writes while the pipeline works without output
    assert all(re.fullmatch(r"data: \{.*\}|: progress|", line) for line in text.split("\n")), file.name
    stopped = file.stem == "client_stop"
    assert stopped or (text.endswith("\n\n") and events[-1]["type"] == "done"), file.name
    assert [e["type"] for e in events].count("meta") == 1


@pytest.mark.parametrize("name", REDACTED_NAMES)
def test_a_redacted_transcript_is_the_legacy_view_of_its_turn_with_no_unmasked_text(name: str) -> None:
    text = (REDACTED / (name + ".sse")).read_text(encoding="utf-8")
    side = json.loads((REDACTED / (name + ".json")).read_text(encoding="utf-8"))
    items, end = harness.frames(text)
    assert end == len(text) and harness.serialise(items) == text, name  # the written form, nothing else
    assert side["plan"] + "_" + side["id"] + "_t" + str(side["turn"]) == name
    assert not {"meta", "done"} & {f["type"] for f in items}
    assert [p for f in items for p in redact.leaks(f, follow_up=side["turn"] > 1)] == [], name
    assert side["params"].get("question") and side["status"] == 200
    # the request holds a question and its parameters, never the token
    assert not {k for k in side["params"] if "token" in k.lower() or "auth" in k.lower()}


def test_no_recording_holds_a_token() -> None:
    tokens = re.compile(r"Bearer |BACKEND_REFRESH_TOKEN")
    for folder in (NATIVE, REDACTED, PAGE):
        for file in folder.glob("*.*"):
            assert not tokens.search(file.read_text(encoding="utf-8")), file.name


def test_the_redaction_keeps_the_patterns_the_evals_checks_read() -> None:
    assert redact.FMT_CITE.pattern == checks.FMT_CITE.pattern == CITATION_GROUP.pattern
    assert redact.FMT_STEP.pattern == checks.FMT_STEP.pattern


def test_the_expected_frames_are_what_the_evals_reader_finds_in_every_native_stream() -> None:
    expected = {file.name: stream(file) for file in NATIVE_FILES}
    file = PAGE / "expected_frames.json"
    if os.environ.get("UPDATE_GOLDEN") == "1":
        file.write_text(json.dumps(expected, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    assert json.loads(file.read_text(encoding="utf-8")) == expected


def test_the_contract_constants_are_what_the_backend_writes() -> None:
    constants = json.loads((PAGE / "contract_constants.json").read_text(encoding="utf-8"))
    de = yaml.safe_load((TU / "messages.de.yaml").read_text(encoding="utf-8"))["guard"]
    assert constants["ERROR_CODES"] == {
        de["busy"]: "busy",
        de["not_ready"]: "not_ready",
        de["error"]: "pipeline_error",
        "timeout": "timeout",
        "empty answer": "empty_answer",
    }
    # the outcome and the variant code of each recorded path that does not simply complete
    outcomes: dict[str, str] = {}
    codes: dict[str, str] = {}
    errors: dict[str, str] = {}
    for side in SIDECARS:
        path = json.loads(side.read_text(encoding="utf-8"))["path"]
        events = stream(side.with_suffix(".sse"))
        if events[-1]["type"] == "done" and events[-1]["content"]["outcome"] != "completed":
            outcomes[path] = events[-1]["content"]["outcome"]
        codes.update({path: e["code"] for e in events if e["type"] == "variant" and "code" in e})
        errors.update({e["content"]: e["code"] for e in events if e["type"] == "error"})
    assert {k: v for k, v in outcomes.items() if k != "not_ready"} == constants["OUTCOMES"]
    assert codes == constants["VARIANT_CODES"]
    assert errors.items() <= constants["ERROR_CODES"].items() and len(errors) == 2  # busy and not ready
    guard = yaml.safe_load((TU / "profile.yaml").read_text(encoding="utf-8"))["guard"]
    assert constants["QUESTION_LIMIT"] == guard["max_question_chars"]


def test_the_page_reads_the_answer_shapes_of_the_backend() -> None:
    assert (PAGE / "answer_shapes.json").read_text(encoding="utf-8") == (FIXTURES / "answer_shapes.json").read_text(
        encoding="utf-8"
    )


def test_the_profile_the_chips_read_is_the_profiles() -> None:
    roles = yaml.safe_load((TU / "roles.yaml").read_text(encoding="utf-8"))
    served = json.loads((PAGE / "profile.json").read_text(encoding="utf-8"))
    programmes = {
        name: (p["degree"], p["label_de"], [v["year"] for v in p["po_versions"]])
        for name, p in roles["programmes"].items()
    }
    assert {p["id"]: (p["degree"], p["label_de"], p["po_versions"]) for p in served["programmes"]} == programmes
    assert served["answer"]["remote"] is False
