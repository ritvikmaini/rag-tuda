"""Model classes S, M, L and XL, the hardware facts that pick a local class, and the models every class downloads."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_tuda.doctor import Facts

APPLE_USABLE_SHARE = 0.65
M_FROM_GB = 24.0  # usable memory below this runs class S, from this up class M


@dataclass(frozen=True)
class SizeClass:
    name: str
    rule: str
    default_card: str | None  # the card `rag up` runs when the hardware picks this class
    ollama_tag: str | None
    download_gb: float | None
    context_k: int
    vllm_model: str
    use_case: str


@dataclass(frozen=True)
class Model:
    name: str
    gb: float


CLASSES = (
    SizeClass(
        "S",
        f"< {M_FROM_GB:g} GB usable memory, or no GPU / AMD / Intel",
        "s-qwen3-4b-instruct-2507",
        "qwen3:4b-instruct-2507-q4_K_M",
        2.5,
        8,
        "Qwen/Qwen3-4B-Instruct-2507",
        "short cited answers on a laptop: a small model, an 8K window, no iterative retrieval",
    ),
    SizeClass(
        "M",
        f">= {M_FROM_GB:g} GB usable memory",
        "m-qwen3.6-35b-a3b",
        "qwen3.6:35b-a3b",
        22.6,
        16,
        "Qwen/Qwen3.6-35B-A3B",
        "the full experience: a 16K window, iterative retrieval, the version table",
    ),
    SizeClass(
        "L",
        "a hosted open model (Groq free tier), chosen with --card",
        None,
        None,
        None,
        8,
        "openai/gpt-oss-120b",
        "a stronger model at the provider's 8K window, with a local analysis model",
    ),
    SizeClass(
        "XL",
        "large open models on vLLM, chosen with --card",
        None,
        None,
        None,
        64,
        "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "long windows and every gated step on a GPU server (documented, untested)",
    ),
)

FIXED_MODELS = (
    Model("Snowflake/snowflake-arctic-embed-m-v2.0", 1.2),
    Model("BAAI/bge-reranker-v2-m3", 2.3),
    Model("docling models", 0.5),
)


def download_total_gb(cls: SizeClass) -> float:
    return (cls.download_gb or 0) + sum(m.gb for m in FIXED_MODELS)


def usable_gb(f: "Facts") -> float:
    """NVIDIA VRAM if present, else a share of Apple unified memory, else 0."""
    if f.nvidia:
        return sum(v for _, v in f.nvidia)
    if f.os == "darwin" and f.arch == "arm64":
        return f.unified_gb * APPLE_USABLE_SHARE
    return 0.0


def pick_class(f: "Facts") -> SizeClass:
    """The local class the hardware fits: L and XL are never picked, they come with an explicit card."""
    s, m, _, _ = CLASSES
    return m if usable_gb(f) >= M_FROM_GB else s
