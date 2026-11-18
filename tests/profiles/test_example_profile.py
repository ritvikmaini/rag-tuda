"""The example profile: it loads with the indexer's and the assistant's schema, keeps the prompts' variables and the fake
LLM's markers, and holds ten undated documents."""

import re
from pathlib import Path

import yaml
from rag_core.settings import load_profile
from rag_core.settings.profile import Profile
from rag_testing import llm

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "profiles" / "example"
PROMPT_OF = {
    "analysis": "question-rewriting",
    "direct": "direct-reply",
    "no_docs": "no-docs-found",
    "answer": "rag-generation",
}
VARIABLES = {  # the variables the assistant fills in for each prompt
    "question-rewriting": {"bot_name", "current_date", "history", "question", "today_iso"},
    "direct-reply": {"bot_name", "conversation_state", "current_date", "intent", "language", "question"},
    "no-docs-found": {"bot_name", "current_date", "language", "question"},
    "rag-generation": {"bot_name", "context", "current_date", "language", "question"},
}
VARIABLE = re.compile(r"\{\{\s*(\w+)\s*\}\}")
YEAR = re.compile(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)")


def test_the_profile_loads_and_names_one_public_collection_and_one_source():
    profile = load_profile(Profile, EXAMPLE / "profile.yaml")
    assert [(c.name, c.audiences) for c in profile.collections] == [("rag", ["public"])]
    assert list(profile.ingest.collections) == ["rag"]
    ((name, source),) = profile.ingest.sources.items()
    assert (name, source.connector, source.audience, source.min_documents) == (
        "gesetze",
        "example.fixture",
        "public",
        10,
    )
    assert source.model_extra == {"base_url": "https://www.gesetze-im-internet.de/"}


def test_the_messages_load_and_label_the_law_texts():
    from assistant_api.settings import Messages

    messages = Messages.model_validate(yaml.safe_load((EXAMPLE / "messages.de.yaml").read_text(encoding="utf-8")))
    assert messages.labels.origins == {"website": "Gesetze im Internet"}
    assert messages.labels.source_types == {} and messages.labels.hosts == {}
    # no "internen": the example's audience is public
    assert messages.docs.none == "Hinweis: Für die Antwort wurden keine Quellen benutzt!"


def test_each_prompt_is_version_1_of_this_profile():
    for name in PROMPT_OF.values():
        assert yaml.safe_load((EXAMPLE / "prompts" / f"{name}.yaml").read_text(encoding="utf-8"))["version"] == 1, name


def test_each_prompt_keeps_its_variables_and_the_fake_llms_marker():
    for name in PROMPT_OF.values():
        mine = (EXAMPLE / "prompts" / f"{name}.yaml").read_text(encoding="utf-8")
        assert set(VARIABLE.findall(mine)) == VARIABLES[name], name
    for purpose, marker in llm.MARKERS:
        holders = [
            n for n in PROMPT_OF.values() if marker in (EXAMPLE / "prompts" / f"{n}.yaml").read_text(encoding="utf-8")
        ]
        assert holders == [PROMPT_OF[purpose]], purpose
    for mark in llm.QUESTION_MARKS.values():
        assert any(
            mark.strip() in (EXAMPLE / "prompts" / f"{n}.yaml").read_text(encoding="utf-8") for n in PROMPT_OF.values()
        )


def test_ten_documents_each_titled_and_without_a_year_in_title_or_name():
    docs = sorted((EXAMPLE / "documents").glob("*.md"))
    assert len(docs) == 10
    for path in docs:
        first = path.read_text(encoding="utf-8").split("\n", 1)[0]
        assert first.startswith("# ") and not YEAR.search(first) and not YEAR.search(path.stem), path.name
    assert sum(p.stat().st_size for p in docs) < 600_000  # the CI leg stays small


def test_the_ci_overlay_points_the_source_at_the_mounted_documents(tmp_path):
    from indexer.settings import load_settings

    env = {
        "RAG_PROFILE": str(EXAMPLE / "profile.yaml"),
        "RAG_OVERLAY": str(ROOT / "deploy" / "env" / "ci-example.yaml"),
    }
    settings = load_settings(
        env, secrets_dir=str(tmp_path / "secrets")
    )  # no secret file needed, as tests/indexer/conftest.py
    assert settings.collection == "rag"
    assert {n: e.dir for n, e in settings.endpoints.items()} == {"gesetze": "/app/example"}
