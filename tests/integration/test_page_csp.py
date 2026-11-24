"""The page on the real backend under its Content-Security-Policy: the header is on the page and on an API reply, and
everything the page loads (scripts, styles, icons) comes from its own origin or is a data: image, so the policy cannot
break it. The browser check of the same page is web/page/tests/e2e/live_csp.py. Black box: HTTP only."""

from html.parser import HTMLParser
from urllib.parse import urlsplit

import httpx
from rag_testing import harness

POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'"
)


class References(HTMLParser):
    """The URL of every script, stylesheet, icon and image an HTML page references, and its inline scripts, styles and
    event-handler attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.inline: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = dict(attrs)
        if tag == "script" and not found.get("src"):
            self.inline.append("script")
        if tag == "style":
            self.inline.append("style")
        self.inline += [name for name in found if name.startswith("on") or name == "style"]
        if url := found.get("src") or (found.get("href") if tag == "link" else None):
            self.urls.append(url)


def test_the_page_and_an_api_reply_carry_the_policy() -> None:
    with httpx.Client(base_url=harness.BACKEND, timeout=30) as client:
        assert client.get("/").headers["Content-Security-Policy"] == POLICY
        assert client.get("/api/profile").headers["Content-Security-Policy"] == POLICY  # also on a 401


def test_every_script_style_and_image_the_page_references_is_its_own() -> None:
    with httpx.Client(base_url=harness.BACKEND, timeout=30) as client:
        page = client.get("/")
        found = References()
        found.feed(page.text)
        assert found.urls and found.inline == []  # the page's HTML itself has no inline script, handler or style
        for url in found.urls:
            parts = urlsplit(url)
            if parts.scheme == "data":
                assert parts.path.startswith("image/"), url  # img-src 'self' data:
            else:
                assert not parts.scheme and not parts.netloc, url  # same origin
                assert client.get("/" + url.lstrip("/")).status_code == 200, url
