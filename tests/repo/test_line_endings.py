"""Line endings: LF in the index for every text file, and in every checkout; the page's vendored libraries are byte-exact."""

import os
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, ".git")),
    reason="needs the git repository (a git archive export has no index)",
)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def test_no_text_file_is_stored_with_crlf():
    """The vendored libraries (`-text`) are exempt: their sha384 is the page's SRI value, so they are stored as released."""
    stored = [line.split("\t", 1) for line in git("ls-files", "--eol").splitlines()]
    assert [path for info, path in stored if info.startswith(("i/crlf", "i/mixed")) and "attr/-text" not in info] == []


def test_text_checks_out_with_lf_and_the_vendored_libraries_untouched():
    out = git(
        "check-attr",
        "eol",
        "text",
        "--",
        "apps/assistant_api/src/assistant_api/app.py",
        "web/page/vendor/showdown-2.1.0.min.js",
    )
    assert out.splitlines() == [
        "apps/assistant_api/src/assistant_api/app.py: eol: lf",
        "apps/assistant_api/src/assistant_api/app.py: text: auto",
        "web/page/vendor/showdown-2.1.0.min.js: eol: lf",
        "web/page/vendor/showdown-2.1.0.min.js: text: unset",
    ]
