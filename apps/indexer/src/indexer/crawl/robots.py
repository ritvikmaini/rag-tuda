"""A robots.txt matcher after RFC 9309.

Rules are grouped by user-agent product token (matched without regard to case; a host's `*` group applies to an agent
with no group of its own). A rule is a path pattern with `*` wildcards and an optional `$` end anchor, matched from the
start of the path plus query string. The longest matching pattern decides, and Allow wins a tie. Paths are compared in
their percent-encoding normal form, with dot-segments removed from the path. A robots.txt that is empty or has no rule
for the agent allows everything. `urllib.robotparser` does neither wildcards nor longest match, so it cannot stand in.
Rules are split at their `*` once at parse time and matched by a greedy left-to-right scan, so a hostile robots.txt
cannot make matching slow."""

import re
from dataclasses import dataclass, field, replace
from urllib.parse import quote, urlsplit

__all__ = ["Robots"]

_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_PERCENT = re.compile(r"%([0-9A-Fa-f]{2})")


def _normalise(text: str) -> str:
    """Decode percent-encoded unreserved characters, upper-case the other escapes, encode what is not ASCII or not
    allowed in a URI; `*` and `$` pass through."""

    def unescape(m: re.Match[str]) -> str:
        char = chr(int(m[1], 16))
        return char if char in _UNRESERVED else "%" + m[1].upper()

    return quote(_PERCENT.sub(unescape, text), safe="%:/?#[]@!$&'()*+,;=-._~")


def _token(agent: str) -> str:
    """The product token of a user agent string or a `User-agent` value: its leading name, lower-cased."""
    m = re.match(r"[A-Za-z0-9_-]+", agent.strip())
    return m[0].lower() if m else ""


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 section 5.2.4 for an absolute path."""
    out: list[str] = []
    segments = path.split("/")[1:]
    for segment in segments:
        if segment == "..":
            if out:
                out.pop()
        elif segment != ".":
            out.append(segment)
    if segments and segments[-1] in (".", ".."):
        out.append("")  # the path ends in a directory
    return "/" + "/".join(out)


@dataclass(frozen=True)
class _Rule:
    allow: bool
    length: int  # of the normalised pattern; the longest match decides
    parts: tuple[str, ...]  # the pattern split at `*`
    anchored: bool  # the pattern ends in `$`

    @classmethod
    def of(cls, allow: bool, pattern: str) -> "_Rule":
        pattern = _normalise(pattern)
        anchored = pattern.endswith("$")
        return cls(allow, len(pattern), tuple(pattern.removesuffix("$").split("*")), anchored)

    def matches(self, target: str) -> bool:
        if len(self.parts) == 1:
            return target == self.parts[0] if self.anchored else target.startswith(self.parts[0])
        first, *middle, last = self.parts
        if not target.startswith(first):
            return False
        pos = len(first)
        for part in middle:  # leftmost occurrence leaves the most room for the rest
            found = target.find(part, pos)
            if found < 0:
                return False
            pos = found + len(part)
        if self.anchored:
            return len(target) - len(last) >= pos and target.endswith(last)
        return target.find(last, pos) >= 0


@dataclass(frozen=True)
class Robots:
    """The parsed rules of one host, per lower-cased user-agent token (or `*`)."""

    groups: dict[str, list[_Rule]] = field(default_factory=dict)
    sitemaps: list[str] = field(default_factory=list)  # the `Sitemap:` URLs, for discovery
    unreadable: str = ""  # why robots.txt could not be read (the host is down), when it could not

    @classmethod
    def parse(cls, text: str) -> "Robots":
        groups: dict[str, list[_Rule]] = {}
        sitemaps: list[str] = []
        current: list[str] = []  # the agents of the group being read
        has_rules = False  # a user-agent line after a rule starts a new group
        for line in text.removeprefix("﻿").splitlines():
            name, sep, value = line.split("#", 1)[0].partition(":")
            name, value = name.strip().lower(), value.strip()
            if not sep:
                continue
            if name == "user-agent":
                if has_rules:
                    current, has_rules = [], False
                key = value if value == "*" else _token(value)
                current.append(key)
                groups.setdefault(key, [])
            elif name in ("allow", "disallow") and current:
                has_rules = True
                if value:  # an empty Disallow allows everything: no rule
                    for agent in current:
                        groups[agent].append(_Rule.of(name == "allow", value))
            elif name == "sitemap" and value:
                sitemaps.append(value)
        return cls(groups, sitemaps)

    @classmethod
    def disallow_all(cls, unreadable: str) -> "Robots":
        """The rules for a host whose robots.txt could not be read (a server error, no answer; `unreadable` says what):
        nothing may be fetched."""
        return replace(cls.parse("User-agent: *\nDisallow: /\n"), unreadable=unreadable)

    def allowed(self, agent: str, url: str) -> bool:
        """Whether `agent` (a product token, or a full user agent string whose leading name counts) may fetch `url`."""
        token = _token(agent)
        rules = self.groups.get(token) if token else None
        if rules is None:
            rules = self.groups.get("*", [])
        parts = urlsplit(url)
        target = _remove_dot_segments(_normalise(parts.path or "/"))
        if parts.query:
            target += _normalise("?" + parts.query)
        best_len, best_allow = -1, True
        for rule in rules:
            if rule.matches(target) and (rule.length, rule.allow) > (best_len, best_allow):
                best_len, best_allow = rule.length, rule.allow
        return best_allow
