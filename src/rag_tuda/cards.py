"""Model cards: TOML files read with the standard library, and the default card for a machine. The backend validates the same
files with rag_core.settings.card; this side reads the few keys the command line needs."""

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_tuda.doctor import Facts

CARDS_DIR = Path(__file__).resolve().parents[2] / "models" / "cards"
# the request window a card gets from its class when it leaves budget.window_tokens out (rag_core's class defaults)
CLASS_WINDOW_TOKENS = {"S": 8000, "M": 16000, "L": 8000, "XL": 64000}


def load(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"card {path}: cannot be read ({exc})") from None


def named(name: str, cards_dir: Path = CARDS_DIR) -> dict:
    return load(cards_dir / f"{name}.toml")


def window_tokens(card: dict) -> int:
    """The card's request window: its own budget.window_tokens, else its class's."""
    return card.get("budget", {}).get("window_tokens") or CLASS_WINDOW_TOKENS[card["class"]]


def window(card: dict) -> int:
    """min(budget.window_tokens, the model's window, tokens per minute when limited): rag_core's effective_window."""
    sizes = [window_tokens(card), card["answer"]["context_window"]]
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


def all_cards(cards_dir: Path = CARDS_DIR) -> list[dict]:
    """Every card of the folder, by file name; a missing folder is an error naming it."""
    if not cards_dir.is_dir():
        raise ValueError(
            f"model cards: the folder {cards_dir} does not exist (run rag from a checkout of the repository)"
        )
    return [load(path) for path in sorted(cards_dir.glob("*.toml"))]


def fitting(facts: "Facts", cards_dir: Path = CARDS_DIR) -> list[dict]:
    """The cards this machine can run: S always, M when the memory fits, L (hosted) always, XL with an NVIDIA GPU."""
    from rag_tuda import classes  # classes reads the cards

    local_limit = classes.pick_class(facts).name  # S below the memory line, M from it
    fit = {"S": True, "M": local_limit == "M", "L": True, "XL": bool(facts.nvidia)}
    return [card for card in all_cards(cards_dir) if fit.get(card["class"], False)]


def choose(card: str | None, model: str | None, facts: "Facts") -> tuple[dict, str | None]:
    """The card `rag up` runs and a warning, if any: the card named with --card; else the card that serves --model (an unknown
    model gets class S, with a warning); else the default card of the hardware."""
    from rag_tuda import classes

    if card:
        return named(card), None
    if model:
        for candidate in all_cards():
            if candidate.get("answer", {}).get("model") == model:
                return candidate, None
        return named(classes.CLASSES[0].default_card or ""), class_of_model(model)[1]
    return named(default_for(facts)), None
