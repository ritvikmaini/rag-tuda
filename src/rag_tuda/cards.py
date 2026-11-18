"""Model cards: TOML files read with the standard library, and the default card for a machine."""

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from rag_tuda import classes

if TYPE_CHECKING:
    from rag_tuda.doctor import Facts


def load(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def default_for(facts: "Facts") -> str:
    """The card of the local class the machine's memory fits (S below 24 GB usable, M above)."""
    name = classes.pick_class(facts).default_card
    assert name  # the local classes always name one
    return name


def class_of_model(model: str, cards_dir: Path) -> tuple[str, str | None]:
    """The class of a model named with --model: its card's class, or S with a warning when no card serves it."""
    for path in sorted(cards_dir.glob("*.toml")):
        card = load(path)
        if card.get("answer", {}).get("model") == model:
            return card["class"], None
    return "S", f"unknown model {model!r}: no card serves it, so it runs with class S values"
