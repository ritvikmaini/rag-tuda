"""The profile's prompts: loaded from its prompts folder, rendered with their variables, no prompt store behind them."""

import subprocess
import sys
from pathlib import Path

import pytest

from rag_core.services.prompts import (
    FOLLOW_UP_PROMPT,
    PROMPT_NAMES,
    PromptService,
    load_profile_prompt,
    load_profile_prompts,
)

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


def test_the_follow_up_prompt_loads_when_the_folder_has_it() -> None:
    tu = Path(__file__).resolve().parents[2] / "profiles" / "tu-informatik" / "prompts"
    assert tuple(load_profile_prompts(tu)) == (*PROMPT_NAMES, FOLLOW_UP_PROMPT)
    assert FOLLOW_UP_PROMPT not in load_profile_prompts(FOLDER)  # the example profile has no iterative round


def test_an_unknown_prompt_is_refused() -> None:
    with pytest.raises(ValueError, match="No prompt defined for 'nope'"):
        PromptService(load_profile_prompts(FOLDER)).render("nope", {})


def _prompt_file(folder: Path, overlays: str | None, slot: bool = True) -> None:
    body = "Rolle.\n{{overlay}}\nFrage: {{question}}" if slot else "Rolle.\nFrage: {{question}}"
    (folder / "no-docs-found.yaml").write_text(
        "name: no-docs-found\ntype: chat\nversion: 1\nvariables: [question]\nmessages:\n"
        f"- role: system\n  content: |-\n    {body.replace(chr(10), chr(10) + '    ')}\n" + (overlays or ""),
        encoding="utf-8",
    )


def test_the_overlay_fills_the_slot_of_the_base_text(tmp_path: Path) -> None:
    _prompt_file(tmp_path, "overlays:\n  small: kurz {{question}}\n  large: |-\n    lang\n    ausführlich\n")
    small = load_profile_prompt(tmp_path, "no-docs-found", "small")
    large = load_profile_prompt(tmp_path, "no-docs-found", "large")
    assert PromptService({"no-docs-found": small}).render("no-docs-found", {"question": "Q"}) == [
        ("system", "Rolle.\nkurz Q\nFrage: Q")
    ]
    assert large.messages == (("system", "Rolle.\nlang\nausführlich\nFrage: {{question}}"),)
    assert load_profile_prompt(tmp_path, "no-docs-found") == large  # large unless a card says small


def test_a_prompt_file_declares_both_overlays_and_the_slot_or_neither(tmp_path: Path) -> None:
    _prompt_file(tmp_path, "overlays:\n  small: a\n")
    with pytest.raises(ValueError, match="overlays: small and large"):
        load_profile_prompt(tmp_path, "no-docs-found")
    _prompt_file(tmp_path, "overlays:\n  small: a\n  large: b\n", slot=False)
    with pytest.raises(ValueError, match="no {{overlay}} slot"):
        load_profile_prompt(tmp_path, "no-docs-found")
    _prompt_file(tmp_path, None)  # a slot without overlays
    with pytest.raises(ValueError, match="no overlays"):
        load_profile_prompt(tmp_path, "no-docs-found")
    _prompt_file(tmp_path, "overlays:\n  small: a\n  large: b\n")
    with pytest.raises(ValueError, match="overlay 'medium'"):
        load_profile_prompt(tmp_path, "no-docs-found", "medium")


def test_a_prompt_without_overlays_and_without_a_slot_is_used_as_written(tmp_path: Path) -> None:
    _prompt_file(tmp_path, None, slot=False)
    assert load_profile_prompt(tmp_path, "no-docs-found", "small") == load_profile_prompt(
        tmp_path, "no-docs-found", "large"
    )
