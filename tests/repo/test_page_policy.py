"""The page and its Content-Security-Policy: the browser suite's server (web/page/tests/e2e/conftest.py) sends the backend's
policy text, and the page's own sources (not vendor/) name no host: no off-origin URL in the CSS and no http(s) literal in the
scripts and the HTML beyond the two listed, which are never fetched."""

import ast
import re
from pathlib import Path

from assistant_api.api.middleware import CONTENT_SECURITY_POLICY

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "web" / "page"
# an XML namespace name (never fetched) and the base that resolves a source link into a URL object
LITERALS = {"http://www.w3.org/2000/svg", "https://example.org/"}


def sources(*suffixes: str) -> list[Path]:
    skipped = {"vendor", "tests", "node_modules"}
    return [p for p in PAGE.rglob("*") if p.suffix in suffixes and not skipped & set(p.relative_to(PAGE).parts)]


def test_the_browser_suite_serves_the_policy_the_backend_sends() -> None:
    tree = ast.parse((PAGE / "tests" / "e2e" / "conftest.py").read_text(encoding="utf-8"))
    literal = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["CONTENT_SECURITY_POLICY"]
    )
    assert literal == CONTENT_SECURITY_POLICY


def test_the_css_loads_nothing_from_another_origin() -> None:
    css = sources(".css")
    assert css
    for path in css:
        text = path.read_text(encoding="utf-8")
        assert "@import" not in text, path
        for url in re.findall(r"url\(\s*['\"]?([^'\")\s]+)", text):
            assert url.startswith("data:image/") or not re.match(r"([a-z]+:|//)", url), (path, url)


def test_no_script_or_page_names_a_host_but_the_listed_literals() -> None:
    for path in sources(".js", ".html"):
        found = set(re.findall(r"https?://[^\s'\"`)]+", path.read_text(encoding="utf-8")))
        assert found <= LITERALS, (path, found)
