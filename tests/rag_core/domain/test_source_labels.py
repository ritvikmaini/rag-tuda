"""Origin labels come from the profile's messages: with the example labels every case gives the labels the code
produced before they moved into the profile."""

import pytest
from rag_core.domain.source_meta import SourceLabels, describe_source

EXAMPLE = SourceLabels(
    source_types={"map": "Karte"},
    origins={"archive": "Archiv", "website": "example.org"},
    hosts={"docs.example.org": "Archiv", "example.org": "example.org", "www.example.org": "example.org"},
    sources_lead="Es liegen {n} Quellen vor:",
    nothing_found="Es wurden keine relevanten Informationen gefunden.",
    date="Stand",
    page="Seite",
    link="Link",
)
CASES = [  # (metadata, the expected origin label)
    ({"source_type": "map", "origin": "website", "url": "https://map.example.org/#uid=1"}, "Karte"),
    ({"origin": "archive", "url": "https://elsewhere.example/x.pdf"}, "Archiv"),
    ({"origin": "website", "url": "https://elsewhere.example/"}, "example.org"),
    ({"url": "http://docs.example.org/a/b.pdf"}, "Archiv"),
    ({"url": "https://example.org/a"}, "example.org"),
    ({"url": "https://www.example.org/a"}, "example.org"),
    ({"download_url": "https://www.example.org/d.pdf", "url": "https://elsewhere.example/"}, "example.org"),
    ({"origin": "other", "url": "https://elsewhere.example/"}, ""),
    ({"url": "https://map.example.org/"}, ""),
    ({}, ""),
]


@pytest.mark.parametrize("meta, origin", CASES)
def test_the_example_labels_give_what_the_code_gave(meta, origin):
    assert describe_source(meta, EXAMPLE)["origin"] == origin


def test_another_profile_names_its_own_origin():
    labels = SourceLabels(
        **{**EXAMPLE.__dict__, "source_types": {}, "origins": {"website": "Gesetze im Internet"}, "hosts": {}}
    )
    assert describe_source({"origin": "website", "url": "https://www.gesetze-im-internet.de/burlg/"}, labels)[
        "origin"
    ] == ("Gesetze im Internet")
    assert describe_source({"source_type": "map", "url": "https://example.org/"}, labels)["origin"] == ""
