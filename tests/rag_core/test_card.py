"""Model cards: the validated settings model, the window rule, the class defaults and the quirks."""

import tomllib
from pathlib import Path

import pytest

from rag_core.domain.language import strip_foreign
from rag_core.settings import SettingsError
from rag_core.settings.card import Card, effective_window, load_card, with_local_url
from rag_core.settings.profile import Generation

ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / "models" / "cards"
SHIPPED = (
    "s-qwen3.5-4b",
    "s-qwen3-4b-instruct-2507",
    "s-gemma4-e4b",
    "m-gemma4-31b",
    "m-qwen3.8-27b",
    "m-qwen3.6-35b-a3b",
    "l-groq-gpt-oss-120b",
    "l-groq-qwen3.8-27b",
    "l-groq-gpt-oss-20b",
    "xl-vllm-example",
)

LOCAL = {"provider": "ollama", "url": "http://127.0.0.1:11434/v1", "model": "test:4b"}


def card(size: str = "M", *, answer: dict | None = None, **sections: dict) -> Card:
    answer = {**LOCAL, "context_window": 262144, **(answer or {})}
    data = {
        "name": "test",
        "class": size,
        "structured_output": "json_schema",
        "chars_per_token": 3.3,
        "prompt_overlay": "large",
        "slots": 1,
        "answer": answer,
        **sections,
    }
    return Card.model_validate(data)


def test_window_is_min_of_budget_model_and_tpm() -> None:
    local = card(budget={"window_tokens": 16000, "answer_tokens": 1200}, limits={"tpm": 0})
    assert effective_window(local) == 16000
    groq = card(
        "L",
        answer={
            "provider": "groq",
            "url": "https://api.groq.com/openai/v1",
            "model": "g/120b",
            "context_window": 131072,
        },
        analysis={**LOCAL, "context_window": 32768},
        budget={"window_tokens": 16000, "answer_tokens": 1000},
        limits={"tpm": 8000},
    )
    assert effective_window(groq) == 8000
    small_model = card(answer={"context_window": 4096}, budget={"window_tokens": 16000, "answer_tokens": 700})
    assert effective_window(small_model) == 4096


def test_remote_answer_requires_analysis_block() -> None:
    remote = {"provider": "groq", "url": "https://api.groq.com/openai/v1", "model": "g/120b", "context_window": 131072}
    with pytest.raises(ValueError, match="analysis"):
        card("L", answer=remote)
    with_block = card("L", answer=remote, analysis={**LOCAL, "context_window": 32768})
    assert with_block.analysis_endpoint.model == "test:4b" and with_block.answer.model == "g/120b"
    assert (
        card().analysis_endpoint is card().answer or card().analysis_endpoint == card().answer
    )  # omitted: the answer's


def test_class_defaults_fill_missing_budget() -> None:
    s = card("S", budget={"window_tokens": 8000})
    assert (s.budget.final_k, s.budget.full_sources, s.budget.history_turns, s.budget.history_tokens) == (8, 3, 2, 1200)
    assert (s.budget.max_queries, s.budget.pre_rerank_k, s.budget.rerank_queries, s.budget.answer_tokens) == (
        3,
        20,
        1,
        700,
    )
    assert not s.gated.iterative_retrieval and not s.gated.version_table
    m = card("M")
    assert (m.budget.window_tokens, m.budget.final_k, m.budget.full_sources, m.budget.answer_tokens) == (
        16000,
        14,
        6,
        1200,
    )
    assert m.gated.iterative_retrieval and m.gated.version_table
    # a key the card names wins over the class default
    assert card("M", budget={"final_k": 9}).budget.final_k == 9
    xl = card("XL")
    assert (xl.budget.window_tokens, xl.budget.final_k, xl.budget.history_turns, xl.budget.answer_tokens) == (
        64000,
        25,
        8,
        2000,
    )
    assert card("L").budget.window_tokens == 8000
    with pytest.raises(ValueError):
        card("XXL")


def test_cjk_strip_only_when_quirk_on() -> None:
    text = "Antwort 中文 かな 한글 ok"
    assert card().quirks.strip_cjk is False
    assert strip_foreign(text, card().quirks.strip_cjk) == text
    on = card(quirks={"strip_cjk": True})
    assert on.quirks.strip_cjk is True
    assert strip_foreign(text, on.quirks.strip_cjk) == "Antwort    ok"


def test_logit_bias_only_from_card_quirks() -> None:
    assert card().quirks.logit_bias == {} and card().quirks.bad_words == ()
    biased = card(quirks={"logit_bias": {3845: -5.0}, "bad_words": ["Mitarbeiter"]})
    assert biased.quirks.logit_bias == {3845: -5.0} and biased.quirks.bad_words == ("Mitarbeiter",)
    with pytest.raises(ValueError, match="1024"):
        card(quirks={"logit_bias": {i: -100.0 for i in range(1025)}})
    assert len(card(quirks={"logit_bias": {i: -100.0 for i in range(1024)}}).quirks.logit_bias) == 1024
    # the profile no longer carries any of them, nor the answer's sampling and tokens (the card's)
    base = {"analysis_temperature": 0, "analysis_max_tokens": 400}
    assert Generation.model_validate(base)
    for key, value in (
        ("model", "m"),
        ("analysis_model", "m"),
        ("disable_thinking", True),
        ("logit_bias", {1: 1.0}),
        ("bad_words", ["x"]),
        ("max_tokens", 900),
        ("temperature", 0.1),
        ("sampling", {"top_k": 50}),
    ):
        with pytest.raises(ValueError):
            Generation.model_validate({**base, key: value})


def test_the_ten_shipped_cards_load_with_tomllib_and_the_model() -> None:
    assert sorted(p.stem for p in CARDS.glob("*.toml")) == sorted(SHIPPED)
    for name in SHIPPED:
        raw = tomllib.loads((CARDS / f"{name}.toml").read_text(encoding="utf-8"))
        loaded = load_card(name, CARDS)
        assert raw["name"] == loaded.name == name and loaded.size_class.lower() == name.split("-")[0]
        assert load_card(CARDS / f"{name}.toml") == loaded
        assert 0 < effective_window(loaded) <= loaded.answer.context_window
    groq = load_card("l-groq-gpt-oss-120b", CARDS)
    assert groq.limits.tpm == 8000 and effective_window(groq) == 8000 and groq.analysis is not None
    assert groq.reasoning.answer == "low" and groq.answer.model == "openai/gpt-oss-120b"
    assert load_card("m-qwen3.6-35b-a3b", CARDS).answer.model == "qwen3.6:35b-a3b"


def test_a_bad_card_names_the_file_and_the_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('name = "bad"\nclass = "M"\nbogus = 1\n', encoding="utf-8")
    with pytest.raises(SettingsError, match=r"bogus.*bad\.toml|bad\.toml.*bogus"):
        load_card(path)
    with pytest.raises(SettingsError, match="cannot be read"):
        load_card("missing", tmp_path)


def test_a_section_that_is_no_table_names_the_file_and_the_key(tmp_path: Path) -> None:
    path = tmp_path / "flat.toml"
    path.write_text(
        'name = "flat"\nclass = "M"\nstructured_output = "json_object"\nchars_per_token = 3\nprompt_overlay = "large"\n'
        'slots = 1\nbudget = 5\n[answer]\nprovider = "ollama"\nurl = "http://127.0.0.1:11434/v1"\nmodel = "m"\n'
        "context_window = 8192\n[gated]\niterative_retrieval = true\nversion_table = true\n",
        encoding="utf-8",
    )
    with pytest.raises(SettingsError, match=r"budget.*flat\.toml"):
        load_card(path)


def test_llm_url_replaces_the_local_endpoints_only() -> None:
    local = card(analysis={**LOCAL, "model": "analysis:1b", "context_window": 32768})
    moved = with_local_url(local, "http://ollama:11434/v1")
    assert moved.answer.url == moved.analysis_endpoint.url == "http://ollama:11434/v1"
    assert moved.answer.model == "test:4b" and moved.analysis_endpoint.model == "analysis:1b"
    groq = {"provider": "groq", "url": "https://api.groq.com/openai/v1", "model": "g/120b", "context_window": 131072}
    mixed = with_local_url(
        card("L", answer=groq, analysis={**LOCAL, "context_window": 32768}), "http://ollama:11434/v1"
    )
    assert mixed.answer.url == "https://api.groq.com/openai/v1"  # a remote endpoint keeps its own
    assert mixed.analysis_endpoint.url == "http://ollama:11434/v1"


def test_llm_url_must_be_an_http_url_without_a_trailing_slash() -> None:
    for bad in ("ollama:11434/v1", "http://ollama:11434/v1/"):
        with pytest.raises(SettingsError, match="RAG__LLM_URL"):
            with_local_url(card(), bad)
