"""Prompts: the profile's four chat prompts and its optional follow-up prompt, loaded from its prompts folder, and one renderer, {{variable}} substitution
(every other brace untouched, so the analysis prompt may carry JSON examples). A prompt file may hold a base text with an
{{overlay}} slot and two overlays, small and large; the card's prompt_overlay picks the one that fills the slot."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag_core.domain.query_analysis import VAR_PATTERN, fill_template

PROMPT_NAMES = ("rag-generation", "no-docs-found", "question-rewriting", "direct-reply")
FOLLOW_UP_PROMPT = "follow-up-queries"  # a profile without the file has no iterative round
OVERLAYS = ("small", "large")
OVERLAY_SLOT = "{{overlay}}"


@dataclass(frozen=True)
class PromptText:
    name: str
    version: int  # the version the profile file declares; a turn records it
    messages: tuple[tuple[str, str], ...]  # (role, template)
    variables: tuple[str, ...]


def load_profile_prompt(folder: Path, name: str, overlay: str = "large") -> PromptText:
    """One chat prompt of a prompts folder, its slot filled with the overlay; a file whose variables differ from the {{names}}
    its text uses (the overlay's included), or whose overlays and slot do not go together, is refused."""
    if overlay not in OVERLAYS:
        raise ValueError(f"overlay {overlay!r}: expected one of {', '.join(OVERLAYS)}")
    data = yaml.safe_load((folder / f"{name}.yaml").read_text(encoding="utf-8"))
    if data.get("name") != name or data.get("type") != "chat":
        raise ValueError(f"{folder / name}.yaml: expected name {name!r} and type chat")
    messages = tuple((str(m["role"]), str(m["content"])) for m in data["messages"])
    overlays = data.get("overlays")
    slotted = any(OVERLAY_SLOT in text for _, text in messages)
    if overlays is None and slotted:
        raise ValueError(f"{name}: the text has an {OVERLAY_SLOT} slot but the file has no overlays")
    if overlays is not None:
        if sorted(overlays) != sorted(OVERLAYS):
            raise ValueError(f"{name}: overlays: small and large expected, got {sorted(overlays)}")
        if not slotted:
            raise ValueError(f"{name}: the file has overlays but the text has no {OVERLAY_SLOT} slot")
        messages = tuple((role, text.replace(OVERLAY_SLOT, str(overlays[overlay]))) for role, text in messages)
    used = sorted(set(VAR_PATTERN.findall("".join(text for _, text in messages))))
    if used != sorted(data["variables"]):
        raise ValueError(f"{name}: variables {sorted(data['variables'])}, but the text uses {used}")
    return PromptText(name, int(data["version"]), messages, tuple(data["variables"]))


def load_profile_prompts(folder: Path, overlay: str = "large") -> dict[str, PromptText]:
    """The profile's four chat prompts, and the follow-up prompt when the folder has it, each with the overlay of the card's class."""
    names = (*PROMPT_NAMES, FOLLOW_UP_PROMPT) if (folder / f"{FOLLOW_UP_PROMPT}.yaml").exists() else PROMPT_NAMES
    return {name: load_profile_prompt(folder, name, overlay) for name in names}


class PromptService:
    def __init__(self, prompts: Mapping[str, PromptText]) -> None:
        self.prompts = prompts

    def versions(self) -> dict[str, int]:
        """The version of each prompt, by name."""
        return {name: prompt.version for name, prompt in self.prompts.items()}

    def render(self, name: str, variables: Mapping[str, object]) -> list[tuple[str, str]]:
        """(role, text) messages with the variables filled in; a variable that is None renders empty."""
        if name not in self.prompts:
            raise ValueError(f"No prompt defined for '{name}'")
        values = {key: "" if value is None else value for key, value in variables.items()}
        return [(role, fill_template(text, values)) for role, text in self.prompts[name].messages]
