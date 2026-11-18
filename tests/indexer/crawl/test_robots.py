"""The robots matcher: RFC 9309 groups by product token, `*` and `$` in rules, rules matched against path plus query,
the longest rule winning with Allow ahead on a tie, and percent-encoding compared in its normal form."""

import time

import pytest

from indexer.crawl.robots import Robots

AGENT = "rag-tuda/1.0 (student research prototype)"
HOST = "https://www.example.org"

# the rules of the FB20 site
FB20 = """\
User-agent: *
Disallow: *neueurl=
Disallow: *view=
Disallow: *action=
Disallow: *.store
Disallow: *.query
Disallow: *.delete
Disallow: *.login
Disallow: *.logout
Disallow: *.search
Disallow: *.form
Disallow: *.captcha
Sitemap: https://www.example.org/sitemap.xml
"""


def test_query_rule_blocks_view_param():
    robots = Robots.parse("User-agent: *\nDisallow: /*view=\n")
    assert not robots.allowed(AGENT, f"{HOST}/a?view=print")
    assert not robots.allowed(AGENT, f"{HOST}/a?x=1&view=print")
    assert robots.allowed(AGENT, f"{HOST}/a")
    assert robots.allowed(AGENT, f"{HOST}/a?page=2")


def test_wildcard_suffix_blocks_login():
    robots = Robots.parse(FB20)
    assert not robots.allowed(AGENT, f"{HOST}/fb20/en/index.login")
    assert not robots.allowed(AGENT, f"{HOST}/x.captcha")
    assert not robots.allowed(AGENT, f"{HOST}/x?neueurl=/y")
    assert not robots.allowed(AGENT, f"{HOST}/x?action=edit")
    assert robots.allowed(AGENT, f"{HOST}/fb20/en/index.en.jsp")
    assert robots.allowed(AGENT, f"{HOST}/x.pdf")
    assert robots.sitemaps == ["https://www.example.org/sitemap.xml"]


def test_longest_match_allow_wins():
    robots = Robots.parse("User-agent: *\nDisallow: /docs/\nAllow: /docs/public/\nDisallow: /docs/public/draft\n")
    assert not robots.allowed(AGENT, f"{HOST}/docs/a.html")
    assert robots.allowed(AGENT, f"{HOST}/docs/public/a.html")
    assert not robots.allowed(AGENT, f"{HOST}/docs/public/draft.html")
    # equal length: Allow beats Disallow, whatever the order
    for rules in ("Allow: /p\nDisallow: /p\n", "Disallow: /p\nAllow: /p\n"):
        assert Robots.parse("User-agent: *\n" + rules).allowed(AGENT, f"{HOST}/p")


def test_dollar_anchor():
    robots = Robots.parse("User-agent: *\nDisallow: /*.pdf$\nDisallow: /exact$\n")
    assert not robots.allowed(AGENT, f"{HOST}/a/b.pdf")
    assert robots.allowed(AGENT, f"{HOST}/a/b.pdf?x=1")
    assert robots.allowed(AGENT, f"{HOST}/a/b.pdf.html")
    assert not robots.allowed(AGENT, f"{HOST}/exact")
    assert robots.allowed(AGENT, f"{HOST}/exact/more")


def test_agent_group_falls_back_to_star():
    text = "User-agent: *\nDisallow: /private\n\nUser-agent: OtherBot\nDisallow: /\n\nUser-agent: Rag-TUDA\nDisallow: /own\n"
    robots = Robots.parse(text)
    # the product token matches without regard to case and ignores the version and comment
    assert not robots.allowed(AGENT, f"{HOST}/own/x")
    assert robots.allowed(AGENT, f"{HOST}/private")  # its own group replaces the `*` group
    assert not robots.allowed("OtherBot/2", f"{HOST}/anything")
    assert not robots.allowed("somebot", f"{HOST}/private/x")  # no group of its own: `*`
    assert robots.allowed("somebot", f"{HOST}/open")


def test_missing_robots_allows():
    for text in ("", "   \n\n", "# only a comment\n", "User-agent: *\nDisallow:\n"):
        assert Robots.parse(text).allowed(AGENT, f"{HOST}/a?view=print")


def test_agents_share_a_group_and_a_split_group_merges():
    text = "User-agent: a\nUser-agent: rag-tuda\nDisallow: /x\n\nUser-agent: rag-tuda\nDisallow: /y\n"
    robots = Robots.parse(text)
    assert not robots.allowed(AGENT, f"{HOST}/x")
    assert not robots.allowed(AGENT, f"{HOST}/y")
    assert not robots.allowed("a", f"{HOST}/x")
    assert robots.allowed("a", f"{HOST}/y")


@pytest.mark.parametrize(
    ("rule", "path", "blocked"),
    [
        ("/caf%C3%A9", "/café", True),  # a rule written encoded meets a path written raw
        ("/café", "/caf%c3%a9", True),  # and the other way round, hex case ignored
        ("/a%2Fb", "/a/b", False),  # an encoded reserved character stays distinct
        ("/a%2fb", "/a%2Fb", True),
        ("/%7Euser", "/~user", True),  # an encoded unreserved character is decoded
    ],
)
def test_percent_encoding_is_normalised(rule, path, blocked):
    robots = Robots.parse(f"User-agent: *\nDisallow: {rule}\n")
    assert robots.allowed(AGENT, HOST + path) is not blocked


def test_comments_crlf_and_unknown_fields_are_tolerated():
    robots = Robots.parse("﻿USER-AGENT: *  # everyone\r\nCrawl-delay: 5\r\ndisallow: /a # no\r\nnonsense\r\n")
    assert not robots.allowed(AGENT, f"{HOST}/a")
    assert robots.allowed(AGENT, f"{HOST}/b")


def test_many_wildcards_match_in_linear_time():
    robots = Robots.parse("User-agent: *\nDisallow: /*a*a*a*a*a*a*a*a*a*a*a*a*b\nDisallow: /*" + "x*" * 60 + "x$\n")
    started = time.perf_counter()
    assert robots.allowed(AGENT, HOST + "/" + "a" * 2000)
    assert robots.allowed(AGENT, HOST + "/" + "x" * 2000 + "y")
    assert not robots.allowed(AGENT, HOST + "/" + "a" * 20 + "b")
    assert time.perf_counter() - started < 0.1


@pytest.mark.parametrize(
    ("rule", "path", "blocked"),
    [
        ("/*a*b*c$", "/1a2b3c", True),
        ("/*a*b*c$", "/1a2b3c4", False),
        ("/*b*b$", "/b", False),  # the two b's cannot share one character
        ("/*b*b$", "/bb", True),
        ("/a*a$", "/a", False),
        ("/a*a$", "/aa", True),
    ],
)
def test_wildcard_sequences(rule, path, blocked):
    robots = Robots.parse(f"User-agent: *\nDisallow: {rule}\n")
    assert robots.allowed(AGENT, HOST + path) is not blocked


def test_group_token_ignores_a_version_suffix():
    robots = Robots.parse("User-agent: *\nDisallow: /\n\nUser-agent: rag-tuda/1.0\nDisallow: /own\n")
    assert robots.allowed(AGENT, f"{HOST}/other")
    assert not robots.allowed(AGENT, f"{HOST}/own")


@pytest.mark.parametrize(
    ("path", "blocked"),
    [
        ("/x/../private", True),
        ("/x/%2E%2E/private/y", True),
        ("/private/./a", True),
        ("/private/..", False),
        ("/a/./b", False),
    ],
)
def test_dot_segments_are_resolved(path, blocked):
    robots = Robots.parse("User-agent: *\nDisallow: /private\n")
    assert robots.allowed(AGENT, HOST + path) is not blocked


def test_disallow_all_blocks_every_url():
    robots = Robots.disallow_all()
    assert not robots.allowed(AGENT, f"{HOST}/")
    assert not robots.allowed("anybot", f"{HOST}/a?b=c")
