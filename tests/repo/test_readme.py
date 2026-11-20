"""Every `rag <command>` and `just <recipe>` the README shows in code exists: the command line's subcommands, the justfile's
recipes. A renamed or removed command fails here instead of staying in the README."""

import re
from pathlib import Path

import pytest

from rag_tuda import cli

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CODE = re.compile(r"```.*?```|`[^`\n]+`", re.S)


def subcommands(capsys) -> set[str]:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    return set(re.findall(r"[{,]([a-z]+)(?=[,}])", capsys.readouterr().out))


def recipes() -> set[str]:
    return set(re.findall(r"^([a-z][\w-]*)(?: [^:\n]*)?:(?!=)", (ROOT / "justfile").read_text(encoding="utf-8"), re.M))


def mentioned(program: str) -> set[str]:
    code = "\n".join(CODE.findall(README))
    return set(re.findall(rf"(?<![\w./-]){program} ([a-z][\w-]*)", code))


def test_every_readme_command_exists(capsys):
    rag, just = mentioned("rag"), mentioned("just")
    # the README shows both families, so a parser change that finds nothing cannot pass
    assert {"doctor", "up", "reindex", "rollback", "eval", "add", "down"} <= rag
    assert {"ci", "harness"} <= just
    assert rag - subcommands(capsys) == set()
    assert just - recipes() == set()
