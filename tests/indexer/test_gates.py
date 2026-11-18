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


def test_no_stats_file_means_no_size_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "STATS_FILE", str(tmp_path / "none.json"))
    assert gates.size_gate(1, gates.previous_stats(), 0.7)["passed"]
    assert (
        gates.size_gate(70, {"parents": 100}, 0.7)["passed"]
        and not gates.size_gate(69, {"parents": 100}, 0.7)["passed"]
    )


def test_the_stats_file_is_the_one_an_older_image_reads():
    assert gates.STATS_FILE == "/app/data/vector_database_contents/corpus_build_stats.json"


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
    assert gates.refusal({"size": size}) == "corpus shrank to 10 parents (last build 100); swap refused"
