"""No indexer module reads a config file at import: every module is
imported in a fresh interpreter whose working directory has no config/."""

import pkgutil
import subprocess
import sys

import indexer


def _modules():
    for info in pkgutil.walk_packages(indexer.__path__, indexer.__name__ + "."):
        yield info.name


def test_every_indexer_module_imports_without_a_config_file(tmp_path):
    names = sorted(_modules())
    assert {"indexer.cli", "indexer.index.writer"} <= set(names)
    code = "import importlib, sys\nfor n in sys.argv[1:]:\n    importlib.import_module(n)\n"
    run = subprocess.run([sys.executable, "-c", code, *names], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
