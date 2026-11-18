from pathlib import Path

import pytest

from rag_tuda import cards, classes

CARDS = Path(__file__).parent / "fixtures" / "cards"
FILES = sorted(CARDS.glob("*.toml"))


def test_default_card_s_below_24gb(scenario):
    assert cards.default_for(scenario("m2_16")) == "s-qwen3-4b-instruct-2507"  # 16 GB unified: 10.4 usable
    assert cards.default_for(scenario("m3pro36")) == "s-qwen3-4b-instruct-2507"  # 36 GB unified: 23.4 usable
    assert cards.default_for(scenario("cpu32")) == "s-qwen3-4b-instruct-2507"  # no GPU: nothing usable
    assert classes.usable_gb(scenario("m3pro36")) < 24


def test_default_card_m_on_m1max64(scenario):
    assert classes.usable_gb(scenario("m1max64")) == pytest.approx(41.6)
    assert cards.default_for(scenario("m1max64")) == "m-qwen3.6-35b-a3b"
    assert cards.default_for(scenario("rtx4090")) == "m-qwen3.6-35b-a3b"  # 24.5 GB of VRAM


def test_unknown_model_gets_class_s_with_warning():
    assert cards.class_of_model("test:35b-a3b", CARDS) == ("M", None)
    name, warning = cards.class_of_model("some-lab/mystery-7b", CARDS)
    assert name == "S"
    assert warning and "some-lab/mystery-7b" in warning and "class S" in warning


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_cards_parse_with_tomllib(path):
    card = cards.load(path)
    assert card["name"] == path.stem
    assert card["class"] in {c.name for c in classes.CLASSES}
    assert card["answer"]["model"] and card["answer"]["context_window"] > 0
    assert 0 < card["budget"]["window_tokens"] <= card["answer"]["context_window"]
    assert path.stem.split("-")[0] == card["class"].lower()


def test_fixture_cards_cover_each_remote_and_local_shape():
    loaded = {p.stem: cards.load(p) for p in FILES}
    assert loaded["l-test-groq"]["limits"]["tpm"] == 8000 and "analysis" in loaded["l-test-groq"]
    assert loaded["m-test-35b"]["gated"] == {"iterative_retrieval": True, "version_table": True}


def test_a_malformed_card_names_its_file(tmp_path):
    (tmp_path / "bad.toml").write_text("name = ", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.toml"):
        cards.class_of_model("x", tmp_path)


def test_window_is_the_smallest_of_budget_model_and_tokens_per_minute():
    card = cards.load(CARDS / "l-test-groq.toml")
    assert cards.window(card) == 8000
    assert cards.window({"budget": {"window_tokens": 16000}, "answer": {"context_window": 4096}}) == 4096


def test_a_card_without_a_window_gets_its_classs():
    answer = {"context_window": 262144}
    assert cards.window({"class": "M", "answer": answer}) == 16000
    assert cards.window({"class": "S", "budget": {"answer_tokens": 700}, "answer": answer}) == 8000


def test_a_named_card_wins_and_a_model_finds_the_card_that_serves_it(scenario):
    m1 = scenario("m1max64")
    assert cards.choose("s-gemma4-e4b", None, m1) == (cards.named("s-gemma4-e4b"), None)
    assert cards.choose(None, None, m1)[0]["name"] == "m-qwen3.6-35b-a3b"
    assert cards.choose(None, "gemma4:e4b", m1) == (cards.named("s-gemma4-e4b"), None)


def test_an_unknown_model_runs_the_class_s_card_with_a_warning(scenario):
    card, warning = cards.choose(None, "some-lab/mystery-7b", scenario("m1max64"))
    assert card["name"] == "s-qwen3-4b-instruct-2507" and warning and "class S" in warning


def test_a_card_that_is_missing_is_an_error_naming_it(scenario):
    with pytest.raises(ValueError, match="no-such-card"):
        cards.choose("no-such-card", None, scenario("m1max64"))


def test_the_cards_that_fit_follow_the_memory(scenario):
    names = lambda f: [c["name"] for c in cards.fitting(f)]  # noqa: E731
    small, big = names(scenario("m2_16")), names(scenario("m1max64"))
    assert "s-qwen3-4b-instruct-2507" in small and "m-qwen3.6-35b-a3b" not in small
    assert "m-qwen3.6-35b-a3b" in big and "l-groq-gpt-oss-120b" in small and "xl-vllm-example" not in big
    assert "xl-vllm-example" in names(scenario("h100x4"))


def test_a_missing_cards_folder_is_an_error_naming_it(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        cards.all_cards(tmp_path / "nowhere")
