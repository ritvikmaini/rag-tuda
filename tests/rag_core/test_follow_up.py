"""The iterative round's pure parts and its model call."""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator

from rag_core.domain.follow_up import follow_up_schema, parse_follow_up, sections_digest
from rag_core.services.follow_up import follow_up_queries
from rag_core.services.prompts import PromptService, load_profile_prompts
from rag_core.settings import load_profile
from rag_core.settings.profile import Profile

TU = Path(__file__).resolve().parents[2] / "profiles" / "tu-informatik"


def section(title, path, child):
    return SimpleNamespace(
        page_content="Der ganze Abschnitt",
        metadata={"title": title, "section_path": path, "matched_child_text": child},
    )


def test_the_schema_is_in_the_strict_shape_and_carries_no_length_keywords() -> None:
    schema = follow_up_schema()
    assert list(schema["properties"]) == schema["required"] == ["reasoning", "queries", "done"]
    assert schema["additionalProperties"] is False
    assert "maxItems" not in json.dumps(schema) and "minItems" not in json.dumps(schema)
    valid = Draft202012Validator(schema).is_valid
    assert valid({"reasoning": "", "queries": ["a"], "done": False}) and not valid({"queries": []})


def test_the_digest_holds_headers_and_matched_children_within_its_budget() -> None:
    sections = [section("PO 2023", "§ 22 Prüfungen", "Anmeldung " * 100), section("APB", "§ 5", "Rücktritt")]
    digest = sections_digest(sections, 10_000)
    assert digest.splitlines()[0] == "- PO 2023 > § 22 Prüfungen"
    assert "Der ganze Abschnitt" not in digest and "- APB > § 5\n  Rücktritt" in digest
    assert len(digest.splitlines()[1].strip()) <= 300  # a child is cut
    assert sections_digest(sections, 0) == ""
    one = sections_digest(sections, len(digest.split("\n- ")[0]) + 5)  # room for the first entry only
    assert one.startswith("- PO 2023") and "APB" not in one
    bare = SimpleNamespace(page_content="x", metadata={})
    assert sections_digest([bare], 100) == "- Dokument"


def test_the_parser_takes_at_most_two_new_queries_and_never_raises() -> None:
    reply = json.dumps({"reasoning": "r", "queries": ["Eins", "zwei", "drei"], "done": False})
    assert parse_follow_up(reply, []) == ["Eins", "zwei"]
    assert parse_follow_up(reply, ["eins?"]) == ["zwei", "drei"]  # asked before (case and a final "?" do not matter)
    fenced = '<think>x</think>```json\n{"queries": ["Eins"], "done": false}\n```'
    assert parse_follow_up(fenced, []) == ["Eins"]
    for content in (
        json.dumps({"queries": ["Eins"], "done": True}),
        "keine Antwort",
        None,
        "[1, 2]",
        '{"queries": "Eins"}',
        '{"queries": [1, null, ""]}',
    ):
        assert parse_follow_up(content, []) == [], content


class Model:
    def __init__(self, reply=None, error=None, wait=0.0) -> None:
        self.reply, self.error, self.wait, self.messages = reply, error, wait, []

    async def complete(self, messages):
        self.messages = list(messages)
        await asyncio.sleep(self.wait)
        if self.error:
            raise self.error
        return self.reply


def ask(model, timeout_s=5.0):
    profile = load_profile(Profile, TU / "profile.yaml")
    return asyncio.run(
        follow_up_queries(
            question="Was gilt beim Rücktritt?",
            searched=["Rücktritt"],
            sections="- PO 2023 > § 22",
            identity=profile.identity,
            prompts=PromptService(load_profile_prompts(TU / "prompts")),
            model=model,
            timeout_s=timeout_s,
        )
    )


def test_the_call_renders_the_follow_up_prompt_and_parses_the_reply(caplog) -> None:
    caplog.set_level(logging.INFO)
    model = Model(json.dumps({"reasoning": "r", "queries": ["APB Rücktritt"], "done": False}))
    assert ask(model) == ["APB Rücktritt"]
    system, user = model.messages
    assert system[0] == "system" and "{{" not in system[1] + user[1]
    assert "Frage: Was gilt beim Rücktritt?" in user[1] and "- Rücktritt" in user[1] and "- PO 2023 > § 22" in user[1]
    assert "[FollowUp] queries=1" in caplog.text and "Rücktritt" not in caplog.text  # counts only


def test_a_failing_or_late_call_gives_no_queries(caplog) -> None:
    assert ask(Model(error=RuntimeError("boom"))) == []
    assert ask(Model(reply='{"queries": ["x"]}', wait=0.5), timeout_s=0.05) == []
    assert ask(Model(reply="kein JSON")) == []
    assert caplog.text.count("[FollowUp] call failed") == 2
