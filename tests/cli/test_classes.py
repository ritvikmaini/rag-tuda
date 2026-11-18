import dataclasses
import re

import pytest

from rag_tuda import classes
from rag_tuda.doctor import Facts, Docker, Ollama


def facts(nvidia=(), apple=False, unified=0.0, system="linux", machine="x86_64"):
    return Facts(
        os=system,
        arch=machine,
        is_wsl=False,
        cpu_name="x",
        cores=8,
        ram_gb=unified or 32.0,
        apple_chip="Apple M2" if apple else "",
        unified_gb=unified,
        nvidia=list(nvidia),
        docker=Docker(),
        ollama=Ollama(),
        disk_free_gb=100.0,
        brew=False,
    )


@pytest.mark.parametrize(
    "gb,name",
    [(0, "S"), (11.9, "S"), (23.9, "S"), (24, "M"), (40, "M"), (200, "M")],
)
def test_nvidia_boundaries(gb, name):
    f = facts(nvidia=[("GPU", gb)] if gb else [])
    assert classes.pick_class(f).name == name


def test_nvidia_vram_is_summed():
    assert classes.pick_class(facts(nvidia=[("a", 12.0), ("b", 12.0)])).name == "M"


def test_apple_uses_65_percent():
    assert classes.usable_gb(facts(apple=True, unified=64, system="darwin", machine="arm64")) == pytest.approx(41.6)
    assert classes.pick_class(facts(apple=True, unified=16, system="darwin", machine="arm64")).name == "S"
    assert classes.pick_class(facts(apple=True, unified=36, system="darwin", machine="arm64")).name == "S"  # 23.4
    assert classes.pick_class(facts(apple=True, unified=38, system="darwin", machine="arm64")).name == "M"  # 24.7
    assert classes.pick_class(facts(apple=True, unified=64, system="darwin", machine="arm64")).name == "M"


def test_nvidia_wins_over_apple_and_intel_mac_is_zero():
    assert classes.usable_gb(facts(nvidia=[("g", 8.0)], apple=True, unified=64)) == 8.0
    assert classes.usable_gb(facts(unified=64, system="darwin", machine="x86_64")) == 0
    assert classes.pick_class(facts(unified=64, system="darwin", machine="x86_64")).name == "S"


def test_hardware_only_picks_a_local_class():
    assert classes.pick_class(facts(nvidia=[("h", 80.0)] * 8)).name == "M"


def test_class_table_data():
    c = {x.name: x for x in classes.CLASSES}
    assert list(c) == ["S", "M", "L", "XL"]
    assert c["S"].ollama_tag == "qwen3:4b-instruct-2507-q4_K_M" and c["S"].download_gb == 2.5
    assert c["M"].ollama_tag == "qwen3.6:35b-a3b" and c["M"].download_gb == 22.6
    assert (c["S"].context_k, c["M"].context_k, c["L"].context_k, c["XL"].context_k) == (8, 16, 8, 64)
    assert c["L"].ollama_tag is None and c["XL"].ollama_tag is None
    assert c["XL"].vllm_model == "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" and "untested" in c["XL"].use_case
    for x in classes.CLASSES:
        assert x.use_case


def test_default_cards_are_named_in_one_place():
    assert classes.CLASSES[0].default_card == "s-qwen3-4b-instruct-2507"
    assert classes.CLASSES[1].default_card == "m-qwen3.6-35b-a3b"
    assert classes.CLASSES[2].default_card is None and classes.CLASSES[3].default_card is None


def test_no_thinking_tags_anywhere():
    bare = re.compile(r"qwen3:(4b|30b)(?![-\w])")
    for x in classes.CLASSES:
        assert not bare.search(repr(dataclasses.asdict(x)))


def test_fixed_models_and_total():
    names = [m.name for m in classes.FIXED_MODELS]
    assert names == [
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "BAAI/bge-reranker-v2-m3",
        "docling models",
    ]
    assert [m.gb for m in classes.FIXED_MODELS] == [1.2, 2.3, 0.5]
    assert classes.download_total_gb(classes.CLASSES[0]) == pytest.approx(6.5)
    assert classes.download_total_gb(classes.CLASSES[1]) == pytest.approx(26.6)
    assert classes.download_total_gb(classes.CLASSES[3]) == pytest.approx(4.0)
