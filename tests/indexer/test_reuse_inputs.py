"""The frozen-input mode reuse_inputs: a build from what the last build fetched,
fetching nothing, writing no registry or safeguard state. The connector side (the crawl and the PDF sync) is tested with
the connectors; here the build passes reuse to every source's context."""

from indexer import build


def test_frozen_inputs_pass_reuse_to_every_context(settings, fakes):
    sources = {name: fakes.Source() for name in ("web", "pdfs", "maps")}
    result = build.build_corpus(settings, sources, reuse_inputs=True, embedding=fakes.Embedding(None))
    assert [s.asked[0].reuse for s in sources.values()] == [True, True, True] and result["reused"] is True
