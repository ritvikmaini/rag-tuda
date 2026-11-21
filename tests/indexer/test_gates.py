"""The build gates."""

from indexer import gates


def test_a_source_passes_at_its_minimum_and_fails_below():
    assert gates.source_gate("web", 200, 200, "documents")["passed"]
    assert not gates.source_gate("web", 199, 200, "documents")["passed"]
    assert gates.source_gate("pdfs", 9, 9, "pdfs") == {
        "source": "pdfs",
        "kind": "pdfs",
        "count": 9,
        "minimum": 9,
        "passed": True,
    }


def test_no_stats_file_means_no_size_gate(tmp_path):
    assert gates.size_gate(1, gates.previous_stats(str(tmp_path)), 0.7)["passed"]
    assert (
        gates.size_gate(70, {"parents": 100}, 0.7)["passed"]
        and not gates.size_gate(69, {"parents": 100}, 0.7)["passed"]
    )


def test_stats_are_a_reference_only_for_the_same_profile_and_source_set():
    core = gates.source_set({"sources": "core"}, ["web", "docs"])
    assert core == {"set": "core", "names": ["docs", "web"]} == gates.source_set(None, ("docs", "web"))
    stats = {"parents": 100, "profile": "tu", "sources": core}
    assert gates.comparable(stats, "tu", core) == stats
    assert gates.comparable(stats, "example", core) == {}
    assert gates.comparable(stats, "tu", gates.source_set({"sources": "full"}, ["web", "docs"])) == {}
    assert gates.comparable({"parents": 100}, "tu", core) == {}  # written before the stats named them


def test_the_first_failed_gate_names_itself_and_says_swap_refused():
    web = gates.source_gate("web", 5, 200, "documents")
    pdfs = gates.source_gate("pdfs", 1, 9, "pdfs")
    size = gates.size_gate(10, {"parents": 100}, 0.7)
    assert gates.refusal({"web": gates.source_gate("web", 200, 200, "documents")}) is None
    assert gates.refusal({"web": web, "pdfs": pdfs, "size": size}) == (
        "only 5 web documents (minimum 200); swap refused"
    )
    assert gates.refusal({"pdfs": pdfs, "size": size}) == (
        "only 1 pdfs PDFs produced documents (minimum 9); swap refused"
    )
    hosts = gates.hosts_gate({"web": {"hosts_down": [{"host": "a.test", "reason": "http 503"}]}, "pdfs": {}})
    assert hosts == {"down": ["a.test"], "passed": False} and gates.hosts_gate({"web": {}})["passed"]
    assert gates.refusal({"hosts": hosts, "size": size}) == (
        "a.test could not be reached (robots.txt unreadable); swap refused"
    )
    assert gates.refusal({"size": size}) == "corpus shrank to 10 parents (last build 100); swap refused"
