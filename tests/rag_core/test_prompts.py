"""The profile's prompts: loaded from its prompts folder, rendered with their variables, no prompt store behind them."""

import subprocess
import sys
from pathlib import Path

import pytest

from rag_core.services.prompts import PROMPT_NAMES, PromptService, load_profile_prompts

FOLDER = Path(__file__).resolve().parents[2] / "profiles" / "example" / "prompts"


def test_prompts_load_from_profile_without_langfuse() -> None:
    prompts = load_profile_prompts(FOLDER)
    service = PromptService(prompts)
    assert tuple(prompts) == PROMPT_NAMES
    assert service.versions() == {name: prompt.version for name, prompt in prompts.items()}
    variables = {name: f"<{name}>" for name in prompts["no-docs-found"].variables} | {"question": None}
    rendered = service.render("no-docs-found", variables)
    text = "\n".join(content for _, content in rendered)
    assert [role for role, _ in rendered] == [role for role, _ in prompts["no-docs-found"].messages]
    assert "<bot_name>" in text and "{{" not in text  # every variable filled, None as empty
    imported = subprocess.run(
        [sys.executable, "-c", "import sys, rag_core.services.prompts; print('langfuse' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert imported == "False"


def test_an_unknown_prompt_is_refused() -> None:
    with pytest.raises(ValueError, match="No prompt defined for 'nope'"):
        PromptService(load_profile_prompts(FOLDER)).render("nope", {})
