"""Model cards: TOML files read with the standard library, and the default card for a machine. The backend validates the same
files with rag_core.settings.card; this side reads the few keys the command line needs."""

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_tuda.doctor import Facts

CARDS_DIR = Path(__file__).resolve().parents[2] / "models" / "cards"


def load(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"card {path}: cannot be read ({exc})") from None


def named(name: str, cards_dir: Path = CARDS_DIR) -> dict:
    return load(cards_dir / f"{name}.toml")


def window(card: dict) -> int:
    """min(budget.window_tokens, the model's window, tokens per minute when limited): rag_core's effective_window."""
    sizes = [card["budget"]["window_tokens"], card["answer"]["context_window"]]
    if tpm := card.get("limits", {}).get("tpm"):
        sizes.append(tpm)
    return min(sizes)


def default_for(facts: "Facts") -> str:
    """The card of the local class the machine's memory fits (S below 24 GB usable, M above)."""
    from rag_tuda import classes  # classes reads the cards

    name = classes.pick_class(facts).default_card
    if name is None:
        raise ValueError("a local class always names its default card")
    return name


def class_of_model(model: str, cards_dir: Path = CARDS_DIR) -> tuple[str, str | None]:
    """The class of a model named with --model: its card's class, or S with a warning when no card serves it."""
    for path in sorted(cards_dir.glob("*.toml")):
        card = load(path)
        if card.get("answer", {}).get("model") == model:
            return card["class"], None
    return "S", f"unknown model {model!r}: no card serves it, so it runs with class S values"
