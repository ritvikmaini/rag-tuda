"""The scope-score tool: one JSON line per conversation with its intent and the reranker's scores, never a question, a query,
a standalone text or a document; panels skipped; a failure becomes the error class."""

import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from langchain_core.documents import Document

from assistant_api import scope_scores
from assistant_api.graph.state import GraphDeps
from rag_core.services.prompts import PromptService, load_profile_prompts
from rag_core.settings import load_profile
from rag_core.settings.card import load_card
from rag_core.settings.profile import Profile

PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "example" / "profile.yaml"
# a parent may carry None: it counts as 0.0
SCORES = {"Kanarienvogel": (0.91, 0.7, 0.2, 0.1), "Tintenfisch": (), "Zaunkoenig": (0.4, None)}


class Analysis:
    """The analysis model: reads the question's canary word, answers with a standalone text and queries that repeat it."""

    async def complete(self, messages):
        text = " ".join(str(part) for message in messages for part in message)  # (role, text) pairs
        intent = "out_of_scope" if "Kanarienvogel" in text else "lookup"
        return json.dumps(
            {"intent": intent, "standalone_question": "Standalone Kuckuck", "search_queries": ["Suchwort Kuckuck"]}
        )


class Retrieval:
    def __init__(self):
        self.calls = []

    async def retrieve(self, queries, *, question="", date_from=None, date_to=None, principal=None):
        self.calls.append((list(queries), question))
        text = " ".join(queries) + question
        if "Fehlerwurm" in text:
            raise ConnectionError("down: Fehlerwurm")
        word = next((w for w in SCORES if w in text), None)
        return [
            Document(page_content="Dokumenttext Biber", metadata={"cross_encoder_score": s})
            for s in SCORES.get(word or "", ())
        ]


def probe(cid, question, **turn):
    return {"id": cid, "turns": [{"q": question, **turn}, {"q": "Zweite Frage Pinguin"}]}


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    files = {
        "a": [probe("A1", "Wo ist der Kanarienvogel?"), probe("A2", "Wo ist der Tintenfisch?")],
        "b": [
            probe("B1", "Wo ist der Zaunkoenig?"),
            probe("B2", "Der Fehlerwurm?"),
            probe("B3", "Lang", repeat={"text": "x" * 3000, "times": 1}),
        ],
        "p": [probe("Q1", "Panelfrage Walross")],
    }
    (tmp_path / "probes").mkdir()
    for name, convs in files.items():
        (tmp_path / "probes" / f"{name}.json").write_text(json.dumps({"conversations": convs}), encoding="utf-8")
    gate = {
        "sets": {
            "set_a": {"file": "probes/a.json"},
            "set_b": {"file": "probes/b.json", "formats": "v1"},
            "panel": {"file": "probes/p.json", "ids": ["Q1"]},
        }
    }
    (tmp_path / "gate.yaml").write_text(yaml.safe_dump(gate), encoding="utf-8")
    return tmp_path


def make_deps() -> GraphDeps:
    profile = load_profile(Profile, PROFILE)
    return GraphDeps(
        profile=profile,
        card=load_card(PROFILE.parents[2] / "deploy" / "env" / "ci-fake-card.toml"),
        messages=SimpleNamespace(),  # type: ignore[arg-type]
        prompts=PromptService(load_profile_prompts(PROFILE.parent / "prompts")),
        analysis=Analysis(),  # type: ignore[arg-type]
        answer_llm=None,
        counter=None,  # type: ignore[arg-type]
        retrieval=Retrieval(),
    )


def run(profile_dir: Path, names=None, limit=None, deps=None):
    out = io.StringIO()
    code = asyncio.run(scope_scores.score(deps or make_deps(), profile_dir, names, limit, out))
    return code, [json.loads(line) for line in out.getvalue().splitlines()], out.getvalue()


def test_a_row_per_conversation_with_the_keys_in_order_and_the_panel_skipped(profile_dir: Path) -> None:
    code, rows, _ = run(profile_dir)
    assert code == 0 and [(r["set"], r["id"]) for r in rows] == [
        ("set_a", "A1"),
        ("set_a", "A2"),
        ("set_b", "B1"),
        ("set_b", "B2"),
        ("set_b", "B3"),
    ]
    assert all(list(r) == ["set", "id", "intent", "top1", "top3", "hits", "error"] for r in rows)
    assert rows[0] == {
        "set": "set_a", "id": "A1", "intent": "out_of_scope", "top1": 0.91, "top3": [0.91, 0.7, 0.2], "hits": 4, "error": None,
    }  # fmt: skip
    assert (rows[1]["top1"], rows[1]["top3"], rows[1]["hits"]) == (None, [], 0)  # no hits: null, not zero
    assert (rows[2]["intent"], rows[2]["top1"], rows[2]["top3"]) == ("lookup", 0.4, [0.4, 0.0])


def test_only_the_first_turn_is_asked_and_the_retrieval_gets_the_analysis_queries(profile_dir: Path) -> None:
    deps = make_deps()
    run(profile_dir, ["set_a"], deps=deps)
    assert [c[1] for c in deps.retrieval.calls] == ["Standalone Kuckuck"] * 2  # type: ignore[attr-defined]
    assert "Suchwort Kuckuck" in deps.retrieval.calls[0][0]  # type: ignore[attr-defined]


def test_no_question_query_standalone_text_or_document_is_in_the_output(profile_dir: Path) -> None:
    _, _, text = run(profile_dir)
    for word in (
        "Kanarienvogel",
        "Tintenfisch",
        "Zaunkoenig",
        "Fehlerwurm",
        "Kuckuck",
        "Biber",
        "Pinguin",
        "Walross",
        "xxx",
    ):
        assert word not in text, word


def test_a_failure_becomes_the_class_name_and_the_run_goes_on(profile_dir: Path) -> None:
    _, rows, text = run(profile_dir, ["set_b"])
    assert [r["error"] for r in rows] == [None, "ConnectionError", "too_long"]  # the message (a question) is never kept
    assert (rows[1]["intent"], rows[1]["top1"], rows[1]["hits"]) == ("lookup", None, None)  # what was found stays
    assert (rows[2]["intent"], rows[2]["hits"]) == (None, None)  # too long for the analysis: not asked
    assert "Fehlerwurm" not in text


def test_sets_and_limit_choose_what_is_scored(profile_dir: Path) -> None:
    _, rows, _ = run(profile_dir, ["set_b", "set_a"], limit=1)
    assert [(r["set"], r["id"]) for r in rows] == [("set_b", "B1"), ("set_a", "A1")]  # --sets order, one per set


def test_an_unknown_set_or_a_panel_is_refused_and_nothing_scored_exits_1(profile_dir: Path) -> None:
    for names in (["nope"], ["panel"]):
        code, rows, _ = run(profile_dir, names)
        assert code == 1 and rows == []
    code, rows, _ = run(profile_dir, ["set_b"], limit=2)  # B1 scores, B2 fails: it ran
    assert code == 0 and len(rows) == 2
    deps = make_deps()
    deps.retrieval.retrieve = _down  # type: ignore[attr-defined]
    code, rows, _ = run(profile_dir, ["set_a"], deps=deps)
    assert code == 1 and [r["error"] for r in rows] == ["ConnectionError", "ConnectionError"]


async def _down(*args, **kwargs):
    raise ConnectionError("db down: Kanarienvogel")


def test_a_run_that_cannot_start_exits_1_and_logs_only_the_error_class(tmp_path: Path, capsys, caplog) -> None:
    assert scope_scores.main(["--sets", "set_a,set_b", "--limit", "1"], environ={}, secrets_dir=tmp_path) == 1
    assert "could not run: SettingsError" in caplog.text
    assert capsys.readouterr().out == ""


def test_the_secrets_folder_is_left_to_the_settings_rule(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(scope_scores, "load_settings", lambda environ, secrets_dir: seen.append(secrets_dir) or 1 / 0)
    assert scope_scores.main(["--sets", "set_a"], environ={}) == 1
    assert seen == [None]  # RAG_SECRETS_DIR, else /run/secrets, is the settings' rule


def test_a_missing_setting_is_logged_with_its_message(tmp_path: Path, caplog) -> None:
    assert scope_scores.main(["--sets", "set_a"], environ={}, secrets_dir=tmp_path) == 1
    assert "SettingsError: RAG_PROFILE: not set" in caplog.text  # a setting error names a key, never a value
