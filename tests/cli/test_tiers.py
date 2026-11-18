import dataclasses
import re

import pytest

from rag_tuda import tiers
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
    [(0, "T1"), (11.9, "T1"), (12, "T2"), (39.9, "T2"), (40, "T3"), (200, "T3")],
)
def test_nvidia_boundaries(gb, name):
    f = facts(nvidia=[("GPU", gb)] if gb else [])
    assert tiers.pick_tier(f).name == name


def test_nvidia_vram_is_summed():
    assert tiers.pick_tier(facts(nvidia=[("a", 24.0), ("b", 24.0)])).name == "T3"


def test_apple_uses_65_percent():
    assert tiers.usable_gb(facts(apple=True, unified=64, system="darwin", machine="arm64")) == pytest.approx(41.6)
    assert tiers.pick_tier(facts(apple=True, unified=16, system="darwin", machine="arm64")).name == "T1"
    assert tiers.pick_tier(facts(apple=True, unified=36, system="darwin", machine="arm64")).name == "T2"
    assert tiers.pick_tier(facts(apple=True, unified=64, system="darwin", machine="arm64")).name == "T3"


def test_nvidia_wins_over_apple_and_intel_mac_is_zero():
    assert tiers.usable_gb(facts(nvidia=[("g", 8.0)], apple=True, unified=64)) == 8.0
    assert tiers.usable_gb(facts(unified=64, system="darwin", machine="x86_64")) == 0
    assert tiers.pick_tier(facts(unified=64, system="darwin", machine="x86_64")).name == "T1"


def test_t3_plus_needs_four_big_gpus():
    assert tiers.pick_tier(facts(nvidia=[("h", 80.0)] * 4)).name == "T3+"
    assert tiers.pick_tier(facts(nvidia=[("h", 80.0)] * 3)).name == "T3"
    assert tiers.pick_tier(facts(nvidia=[("h", 80.0)] * 3 + [("s", 48.0)])).name == "T3"
    assert tiers.pick_tier(facts(nvidia=[("h", 48.0)] * 8)).name == "T3"


def test_tier_table_data():
    t = {x.name: x for x in tiers.TIERS}
    assert t["T1"].ollama_tag == "qwen3:4b-instruct-2507-q4_K_M" and t["T1"].download_gb == 2.5
    assert t["T2"].ollama_tag == "qwen3:14b" and t["T2"].download_gb == 9.3
    assert t["T3"].ollama_tag == "qwen3:30b-a3b-instruct-2507-q4_K_M" and t["T3"].download_gb == 19
    assert (t["T1"].context_k, t["T2"].context_k, t["T3"].context_k, t["T3+"].context_k) == (8, 8, 32, 32)
    assert t["T3+"].ollama_tag is None
    assert t["T2"].vllm_model == "Qwen/Qwen3-14B-AWQ"
    for x in tiers.TIERS:
        assert x.use_case


def test_no_thinking_tags_anywhere():
    bare = re.compile(r"qwen3:(4b|30b)(?![-\w])")
    for x in tiers.TIERS:
        assert not bare.search(repr(dataclasses.asdict(x)))


def test_fixed_models_and_total():
    names = [m.name for m in tiers.FIXED_MODELS]
    assert names == [
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "BAAI/bge-reranker-v2-m3",
        "docling models",
    ]
    assert [m.gb for m in tiers.FIXED_MODELS] == [1.2, 2.3, 0.5]
    t1 = tiers.TIERS[0]
    assert tiers.download_total_gb(t1) == pytest.approx(6.5)
    assert tiers.download_total_gb(tiers.TIERS[3]) == pytest.approx(4.0)
