"""Model classes S, M, L and XL, the hardware facts that pick a local class, and the models every class downloads."""

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from rag_tuda import cards

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
    window_tokens: int
    vllm_model: str
    use_case: str


@dataclass(frozen=True)
class Model:
    name: str
    gb: float


def size_class(name: str, rule: str, card_name: str, use_case: str, *, local: bool, hf_model: str = "") -> SizeClass:
    """A class described by its representative card: the Ollama tag, the download size, the window and the served model name
    come from the card. Only a local Ollama card has no vLLM name of its own, so the Hugging Face id of its model is given."""
    card = cards.named(card_name)
    answer = card["answer"]
    return SizeClass(
        name,
        rule,
        card_name if local else None,
        answer["model"] if answer["provider"] == "ollama" else None,
        card.get("download_gb"),
        cards.window(card),
        hf_model or answer["model"],
        use_case,
    )


@cache
def _classes() -> tuple[SizeClass, SizeClass, SizeClass, SizeClass]:
    """The four classes, built from the cards on first use so that importing this module needs no `models/` folder."""
    return (
        size_class(
            "S",
            f"< {M_FROM_GB:g} GB usable memory, or no GPU / AMD / Intel",
            "s-qwen3-4b-instruct-2507",
            "short cited answers on a laptop: a small model, an 8K window, no iterative retrieval",
            local=True,
            hf_model="Qwen/Qwen3-4B-Instruct-2507",
        ),
        size_class(
            "M",
            f">= {M_FROM_GB:g} GB usable memory",
            "m-qwen3.6-35b-a3b",
            "the full experience: a 16K window, iterative retrieval, the version table",
            local=True,
            hf_model="Qwen/Qwen3.6-35B-A3B",
        ),
        size_class(
            "L",
            "a hosted open model (Groq free tier), chosen with --card",
            "l-groq-gpt-oss-120b",
            "a stronger model at the provider's 8K window, with a local analysis model",
            local=False,
        ),
        size_class(
            "XL",
            "large open models on vLLM, chosen with --card",
            "xl-vllm-example",
            "long windows and every gated step on a GPU server (documented, untested)",
            local=False,
        ),
    )


def __getattr__(name: str) -> tuple[SizeClass, ...]:
    if name == "CLASSES":  # classes.CLASSES, read on first use
        return _classes()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


FIXED_MODELS = (
    Model("Snowflake/snowflake-arctic-embed-m-v2.0", 1.2),
    Model("BAAI/bge-reranker-v2-m3", 2.3),
    Model("docling models", 0.5),
)


# what a crawl of the core and the full source set fetches (estimates of the sources' size; the full set takes hours)
CRAWL_SETS = {
    "core": "about 300 pages and 120-150 PDFs",
    "full": "about 1,500-2,000 pages and 250-350 PDFs, which takes hours",
}


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
    s, m, _, _ = _classes()
    return m if usable_gb(f) >= M_FROM_GB else s
