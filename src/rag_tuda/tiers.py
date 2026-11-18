"""Hardware tiers and the models each tier runs."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_tuda.doctor import Facts

APPLE_USABLE_SHARE = 0.65
T3_PLUS_GPUS = 4
T3_PLUS_GPU_GB = 80.0


@dataclass(frozen=True)
class Tier:
    name: str
    rule: str
    ollama_tag: str | None
    download_gb: float | None
    context_k: int
    vllm_model: str
    use_case: str


@dataclass(frozen=True)
class Model:
    name: str
    gb: float


TIERS = (
    Tier(
        "T1",
        "< 12 GB, or no GPU / AMD / Intel",
        "qwen3:4b-instruct-2507-q4_K_M",
        2.5,
        8,
        "Qwen/Qwen3-4B-Instruct-2507",
        "personal document Q&A on a laptop, short answers with citations, small collections (thousands of pages)",
    ),
    Tier(
        "T2",
        "12-40 GB",
        "qwen3:14b",
        9.3,
        8,
        "Qwen/Qwen3-14B-AWQ",
        "the full experience: summaries, tables, charts, multi-turn, larger collections",
    ),
    Tier(
        "T3",
        ">= 40 GB",
        "qwen3:30b-a3b-instruct-2507-q4_K_M",
        19,
        32,
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "long documents in one context (32k), stronger reasoning over figures, team use on one workstation",
    ),
    Tier(
        "T3+",
        ">= 4 x 80 GB NVIDIA",
        None,
        None,
        32,
        "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "the same at multi-user scale (untested here)",
    ),
)

FIXED_MODELS = (
    Model("Snowflake/snowflake-arctic-embed-m-v2.0", 1.2),
    Model("BAAI/bge-reranker-v2-m3", 2.3),
    Model("docling models", 0.5),
)


def download_total_gb(tier: Tier) -> float:
    return (tier.download_gb or 0) + sum(m.gb for m in FIXED_MODELS)


def usable_gb(f: "Facts") -> float:
    """NVIDIA VRAM if present, else a share of Apple unified memory, else 0."""
    if f.nvidia:
        return sum(v for _, v in f.nvidia)
    if f.os == "darwin" and f.arch == "arm64":
        return f.unified_gb * APPLE_USABLE_SHARE
    return 0.0


def pick_tier(f: "Facts") -> Tier:
    t1, t2, t3, t3p = TIERS
    if len([1 for _, v in f.nvidia if v >= T3_PLUS_GPU_GB]) >= T3_PLUS_GPUS:
        return t3p
    gb = usable_gb(f)
    return t3 if gb >= 40 else t2 if gb >= 12 else t1
