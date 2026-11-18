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


def test_the_command_line_computes_the_window_the_backend_does():
    from rag_core.settings.card import effective_window, load_card

    shipped = sorted(cards.CARDS_DIR.glob("*.toml"))
    assert len(shipped) == 10
    for path in shipped:
        assert cards.window(cards.load(path)) == effective_window(load_card(path))
