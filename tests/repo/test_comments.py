"""Comments and docstrings describe the code as it is: no decision ids, unit or phase names, task ids or dated notes.

Scanned: the comments and docstrings of every tracked Python file, and the comments of the other tracked text files (`#` in
TOML, YAML, justfile, Dockerfiles and the like, `--` in SQL, `//` in JS, `/* */` in CSS, `<!-- -->` in HTML). Strings and
test data are not comments. An exception goes into ALLOWED with its reason; a stale entry fails the test."""

import ast
import io
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# English month names only: the German ones ("Juli 2026") appear in the date parsers' own examples
MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
TRACES = {
    "decision id": re.compile(r"\bD\d{1,3}\b"),
    "unit or phase name": re.compile(r"\b(?:unit[ /]?\d+[a-z]?|phase[ -]?\d)\b", re.I),
    "phase or task id": re.compile(r"\b(?:P[0-8]|T\d\.\d{1,2}[a-z]?)\b"),
    "dated note": re.compile(
        rf"\b20\d\d-\d\d-\d\d\b|\b\d{{1,2}} {MONTH} 20\d\d\b|\b{MONTH},? (?:\d{{1,2}},? )?20\d\d\b", re.I
    ),
    "old setup wording": re.compile(r"\bthe box\b|\bbox (?:image|recording)s?\b", re.I),
    "planning folder": re.compile("super" + r"powers|restructure" + " program", re.I),
}
# (path, the matched text): a comment that must keep a hit, with the reason
ALLOWED: dict[tuple[str, str], str] = {}

NOT_SCANNED = re.compile(r"^(uv\.lock|.*/uv\.lock|web/page/vendor/|tests/contracts/fixtures/|tests/cli/fixtures/)")
HASH_NAMES = {"justfile", ".gitignore", ".gitattributes", ".env_example", ".dockerignore", ".python-version"}
HASH_SUFFIXES = {".toml", ".yaml", ".yml", ".conf", ".sh", ".cfg", ".txt"}
SLASH_SUFFIXES = {".js", ".mjs"}


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.split("\0") if p and (ROOT / p).is_file() and not NOT_SCANNED.match(p)]


def python_comments(source: str) -> list[tuple[int, str]]:
    found = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            found.append((tok.start[0], tok.string))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and node.body and isinstance(node.body[0], ast.Expr):
                found.append((node.body[0].lineno, doc))
    return found


def line_comments(text: str, marker: str) -> list[tuple[int, str]]:
    """The text after `marker` where it starts a line or follows whitespace (so `https://` is not a comment)."""
    pattern = re.compile(rf"(?:^|\s){re.escape(marker)}(.*)$")
    return [(n, m.group(1)) for n, line in enumerate(text.splitlines(), 1) if (m := pattern.search(line))]


def block_comments(text: str, start: str, end: str) -> list[tuple[int, str]]:
    return [
        (text.count("\n", 0, m.start()) + 1, m.group(1))
        for m in re.finditer(re.escape(start) + r"(.*?)" + re.escape(end), text, re.S)
    ]


def comments(path: str) -> list[tuple[int, str]]:
    file = Path(path)
    text = (ROOT / path).read_text(encoding="utf-8")
    if file.suffix == ".py":
        return python_comments(text)
    if file.suffix in SLASH_SUFFIXES:
        return line_comments(text, "//") + block_comments(text, "/*", "*/")
    if file.suffix == ".css":
        return block_comments(text, "/*", "*/")
    if file.suffix == ".html":
        return block_comments(text, "<!--", "-->")
    if file.suffix == ".sql":
        return line_comments(text, "--")
    if file.name in HASH_NAMES or file.name.startswith("Dockerfile") or file.suffix in HASH_SUFFIXES:
        return line_comments(text, "#")
    return []


def traces() -> list[tuple[str, int, str, str]]:
    found = []
    for path in tracked():
        try:
            scanned = comments(path)
        except (UnicodeDecodeError, SyntaxError):
            continue
        for line, comment in scanned:
            for kind, pattern in TRACES.items():
                found += [(path, line, kind, m.group(0)) for m in pattern.finditer(comment)]
    return found


def test_no_comment_or_docstring_carries_a_process_trace() -> None:
    found = [t for t in traces() if (t[0], t[3]) not in ALLOWED]
    assert found == [], "\n".join(f"{p}:{n}: {kind}: {text}" for p, n, kind, text in found)


def test_every_allowed_exception_is_still_needed() -> None:
    hits = {(path, text) for path, _, _, text in traces()}
    assert [key for key in ALLOWED if key not in hits] == []


def test_the_scan_finds_what_it_is_meant_to() -> None:
    # the sample is assembled so that this file carries no trace itself
    parts = ["D" + "7", "D" + "59", "unit " + "5d", "phase " + "4", "T2" + ".12", "12 " + "Oct 2026", "2026" + "-10-12"]
    month_only = "# since " + "Oct 2026"
    sample = "x = 1  # decided in " + ", ".join(parts) + "\n"
    kinds = {kind for kind, pattern in TRACES.items() if pattern.search(python_comments(sample)[0][1])}
    assert TRACES["decision id"].findall(sample) == [parts[0], parts[1]]
    assert TRACES["dated note"].findall(month_only) == ["Oct 2026"]
    assert kinds == {"decision id", "unit or phase name", "phase or task id", "dated note"}
    assert python_comments('x = "' + parts[0] + ' in a string"\n') == []
