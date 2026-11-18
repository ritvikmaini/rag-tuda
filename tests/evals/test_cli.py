"""The console script `evals`: dispatch, and usage errors exit 64 (2 means "re-runs needed" for the gate)."""

import pytest

from evals import cli


def test_an_unknown_or_missing_command_is_a_usage_error(capsys):
    assert cli.main([]) == 64
    assert cli.main(["nonsense"]) == 64
    assert cli.main(["--help"]) == 0
    assert "run,gate,stats,compare,rescore,smoke" in capsys.readouterr().out


def test_a_command_s_usage_error_exits_64(env):
    assert cli.main(["run", "gold"]) == 64  # --label missing
    assert cli.main(["run", "--label", "t"]) == 64  # neither sets nor --probes


def test_help_of_a_command_exits_0():
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["gate", "--help"])
    assert exit_info.value.code == 0


def test_the_commands_import_no_model_or_database_library():
    import subprocess
    import sys

    code = (
        "import sys; from evals import cli; "
        "print(sorted(m for m in sys.modules if m.startswith(('langchain', "
        "'langfuse', 'openai', 'opentelemetry', 'psycopg_pool', 'numpy', 'scipy'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
