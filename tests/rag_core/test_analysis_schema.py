"""The analysis schema generated from the roles: strict-compatible, enums from the profile, and in step with the TU
rewriter prompt."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from rag_core.domain import query_analysis as qa
from rag_core.domain.filters import REGULATIONS
from rag_core.services.prompts import load_profile_prompts
from rag_core.settings import load_profile
from rag_core.settings.profile import Roles

TU = Path(__file__).resolve().parents[2] / "profiles" / "tu-informatik"
ROLES = load_profile(Roles, TU / "roles.yaml").model_dump()
FIELDS = [
    "reasoning",
    "language",
    "standalone_question",
    "intent",
    "queries_de",
    "queries_en",
    "degree",
    "programme",
    "po_version",
    "regulation",
    "enrolment_term",
    "date_range",
    "shape",
    "needs_more",
]


def objects(node: Any):
    """Every object schema inside a schema."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from objects(value)


def reply(**fields: Any) -> dict[str, Any]:
    """A complete answer: a field not given is empty."""
    empty = {
        "reasoning": "r",
        "language": "de",
        "standalone_question": "Frage?",
        "intent": "lookup",
        "queries_de": ["Frage"],
        "queries_en": [],
        "degree": None,
        "programme": None,
        "po_version": None,
        "regulation": None,
        "enrolment_term": None,
        "date_range": None,
        "shape": "text",
        "needs_more": False,
    }
    return empty | fields


def test_schema_strict_compatible() -> None:
    schema = qa.analysis_schema(ROLES)
    assert list(schema["properties"]) == FIELDS  # the reasoning comes first, so the model thinks before it answers
    for node in objects(schema):  # every property required, no extra keys
        assert set(node["required"]) == set(node["properties"]) and node["additionalProperties"] is False
    for name in (
        "degree",
        "programme",
        "po_version",
        "regulation",
        "date_range",
    ):  # an optional value is a union with null
        assert {"type": "null"} in schema["properties"][name]["anyOf"], name
    assert schema["properties"]["enrolment_term"]["type"] == ["string", "null"]
    assert schema["properties"]["language"]["enum"] == ["de", "en"]
    assert not re.search(
        r'"(minItems|maxItems|minimum|maximum|pattern|format)":', json.dumps(schema)
    )  # keywords strict refuses
    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(reply())


def test_enums_come_from_the_roles() -> None:
    props = qa.analysis_schema(ROLES)["properties"]
    assert props["degree"]["anyOf"][0]["items"]["enum"] == ["bachelor", "master", "promotion"]
    assert props["programme"]["anyOf"][0]["items"]["enum"] == [
        "bsc-informatik",
        "msc-computer-science",
        "msc-informatik",
    ]
    assert props["po_version"]["anyOf"][0]["items"]["enum"] == [2015, 2023]
    assert props["regulation"]["anyOf"][0]["items"]["enum"] == list(REGULATIONS)
    assert props["intent"]["enum"] == list(qa.INTENTS) and props["shape"]["enum"] == list(qa.ANSWER_SHAPES)
    one = {"programmes": {"x-y": {"degree": "master", "po_versions": [{"year": 2020}]}}, "groups": {}}
    narrow = qa.analysis_schema(one)["properties"]
    assert narrow["degree"]["anyOf"][0]["items"]["enum"] == ["master"]
    assert narrow["po_version"]["anyOf"][0]["items"]["enum"] == [2020]


def test_arrays_for_two_versions() -> None:
    valid = Draft202012Validator(qa.analysis_schema(ROLES)).is_valid
    assert valid(reply(po_version=[2015, 2023], programme=["msc-informatik"], degree=["master"]))
    assert valid(reply(po_version=None, regulation=["po", "mhb"]))
    assert not valid(reply(po_version=[2019]))  # a year no programme has a PO for
    assert not valid(reply(po_version=2015))  # one version is still an array
    assert not valid(reply(programme=["bsc-physik"]))
    assert not valid({k: v for k, v in reply().items() if k != "needs_more"})  # every property is required
    assert valid(reply(date_range={"from": "2026-11-19", "to": None}))
    assert not valid(reply(date_range={"from": "2026-11-19"}))


def test_without_roles_the_schema_is_the_base_one() -> None:
    schema = qa.analysis_schema(None)
    assert list(schema["properties"]) == [
        "standalone_question",
        "intent",
        "search_queries",
        "language",
        "date_from",
        "date_to",
        "answer_shape",
    ]
    assert set(schema["required"]) == set(schema["properties"]) and schema["additionalProperties"] is False


@pytest.mark.parametrize("overlay", ["small", "large"])
def test_the_tu_prompt_asks_for_the_schemas_keys(overlay: str) -> None:
    system = load_profile_prompts(TU / "prompts", overlay)["question-rewriting"].messages[0][1]
    shown = re.search(r"# Format der Antwort\n(.*?)\n\n", system, re.S)
    assert shown and re.findall(r'^ {2}"(\w+)":', shown[1], re.M) == FIELDS
    # the prompt names every value the enums allow
    props = qa.analysis_schema(ROLES)["properties"]
    for name in ("degree", "programme", "po_version", "regulation"):
        for value in props[name]["anyOf"][0]["items"]["enum"]:
            assert f'"{value}"' in system or str(value) in system, (name, value)
    # each example answer is one the schema accepts and the parser reads
    examples = [json.loads(line) for line in system.splitlines() if line.startswith('{"reasoning"')]
    assert len(examples) >= 8
    validator = Draft202012Validator(qa.analysis_schema(ROLES))
    for example in examples:
        validator.validate(example)
        parsed = qa.parse_query_analysis(json.dumps(example), "x")
        assert parsed.parsed == "json" and parsed.search_queries == example["queries_de"]
    assert {e["language"] for e in examples} == {"de", "en"}
    assert any(e["queries_en"] for e in examples) and any(e["po_version"] == [2015, 2023] for e in examples)
    assert any(e["needs_more"] for e in examples)
    assert any(e["date_range"] for e in examples)
    # a document type is set only where the message names it (the analysis value filters when nothing else names one)
    assert {tuple(e["regulation"]) for e in examples if e["regulation"]} == {("po",), ("mhb",)}
    longest = max(len(json.dumps(e, ensure_ascii=False)) for e in examples)
    assert longest < 1500, longest  # about 500 tokens at three characters each: inside the 800 of analysis_max_tokens


@pytest.mark.parametrize("roles", [ROLES, None])
def test_the_schema_is_plain_json(roles) -> None:
    assert json.loads(json.dumps(qa.analysis_schema(roles))) == qa.analysis_schema(roles)
