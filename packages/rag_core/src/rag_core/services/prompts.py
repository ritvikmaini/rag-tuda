"""Prompts: the profile's four chat prompts, loaded from its prompts folder, and one renderer, {{variable}} substitution
(every other brace untouched, so the analysis prompt may carry JSON examples)."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag_core.domain.query_analysis import VAR_PATTERN, fill_template

PROMPT_NAMES = ("rag-generation", "no-docs-found", "question-rewriting", "direct-reply")


@dataclass(frozen=True)
class PromptText:
    name: str
    version: int  # the version the profile file declares; a turn records it
    messages: tuple[tuple[str, str], ...]  # (role, template)
    variables: tuple[str, ...]


def load_profile_prompt(folder: Path, name: str) -> PromptText:
    """One chat prompt of a prompts folder; a file whose variables differ from the {{names}} its text uses is refused."""
    data = yaml.safe_load((folder / f"{name}.yaml").read_text(encoding="utf-8"))
    if data.get("name") != name or data.get("type") != "chat":
        raise ValueError(f"{folder / name}.yaml: expected name {name!r} and type chat")
    messages = tuple((str(m["role"]), str(m["content"])) for m in data["messages"])
    used = sorted(set(VAR_PATTERN.findall("".join(text for _, text in messages))))
    if used != sorted(data["variables"]):
        raise ValueError(f"{name}: variables {sorted(data['variables'])}, but the text uses {used}")
    return PromptText(name, int(data["version"]), messages, tuple(data["variables"]))


def load_profile_prompts(folder: Path) -> dict[str, PromptText]:
    """The profile's four chat prompts."""
    return {name: load_profile_prompt(folder, name) for name in PROMPT_NAMES}


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
