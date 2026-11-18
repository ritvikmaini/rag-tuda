"""The schemas of sources.yaml, roles.yaml and acronyms.yaml: strict like the profile, and each rule refuses what it names."""

import copy

import pytest
from pydantic import ValidationError
from rag_core.settings.profile import Acronyms, Roles, Sources

SELECTORS = {"main": ".main-column", "section": "section.content-section", "drop": None}
HOST = {
    "set": "core",
    "seeds": ["https://a.example/start.de.jsp"],
    "sitemap": ["https://a.example/sitemap.de.xml"],
    "prefixes": ["/study/"],
    "exclude": ["/study/old/"],
    "depth": 1,
    "twins": True,
    "selectors": SELECTORS,
}
TERMS = {"start_from": "WiSe 2023/24", "start_to": None, "official_url": None}
PROGRAMME = {
    "label_de": "B.Sc. X",
    "label_en": "B.Sc. X",
    "degree": "bachelor",
    "url_prefix": "https://a.example/x",
    "pdf_prefix": "https://a.example/media/x/",
    "po_versions": [{"year": 2023, **TERMS}],
    "po_change": None,
}
WORDING = {"text_de": "In der Regel gilt die aktuelle.", "text_en": "As a rule the current one applies."}
ROLES = {
    "po_rule": WORDING,
    "programmes": {"bsc-x": PROGRAMME},
    "groups": {
        "students": {"label": "Studierende", "side": "study"},
        "bachelor": {"label": "Bachelor", "side": "study", "part_of": "students"},
    },
    "offices": {"office": {"label_de": "Büro", "label_en": "Office", "url": None}},
}


def changed(base, **edits):
    """A deep copy of base with edits keyed by a path joined with "|" ("hosts|a.example|depth": 2)."""
    copy_ = copy.deepcopy(base)
    for path, value in edits.items():
        *parents, last = path.split("|")
        node = copy_
        for key in parents:
            node = node[key]
        node[last] = value
    return copy_


def sources(**edits):
    return changed({"hosts": {"a.example": HOST}}, **{f"hosts|a.example|{k}": v for k, v in edits.items()})


def test_sources_accept_a_host_and_a_full_block_on_a_core_host():
    parsed = Sources.model_validate(sources())
    assert parsed.hosts["a.example"].full is None
    full = {"sitemap": [], "prefixes": ["/more/"], "exclude": [], "depth": 0}
    extended = Sources.model_validate(sources(full=full)).hosts["a.example"].full
    assert extended is not None and extended.prefixes == ["/more/"]


@pytest.mark.parametrize(
    ("edits", "needle"),
    [
        ({"seeds": ["https://b.example/x"]}, "https URLs on the host"),
        ({"seeds": ["http://a.example/x"]}, "https URLs on the host"),
        ({"sitemap": ["https://b.example/sitemap.xml"]}, "https URLs on the host"),
        ({"prefixes": ["study/"]}, "starting with /"),
        ({"exclude": ["old/"]}, "starting with /"),
        ({"prefixes": [], "depth": 1}, "needs prefixes"),
        ({"depth": 4}, "depth"),
        ({"set": "all"}, "set"),
        ({"set": "full", "full": {"sitemap": [], "prefixes": [], "exclude": [], "depth": 0}}, "only a core host"),
        (
            {"full": {"sitemap": ["https://b.example/s.xml"], "prefixes": [], "exclude": [], "depth": 0}},
            "https URLs on the host",
        ),
        ({"full": {"sitemap": [], "prefixes": ["x/"], "exclude": [], "depth": 0}}, "starting with /"),
        ({"full": {"sitemap": [], "prefixes": [], "exclude": [], "depth": 1}}, "needs prefixes"),
        ({"twin": True}, "twin"),
        ({"selectors": {"main": "main"}}, "selectors"),
    ],
)
def test_a_crawl_host_that_breaks_a_rule_is_refused(edits, needle):
    with pytest.raises(ValidationError, match=needle):
        Sources.model_validate(sources(**edits))


@pytest.mark.parametrize(
    "host", ["www.darmstadt.de", "darmstadt.de", "h-da.de", "www.h-da.de", "A.example", "a.example/x"]
)
def test_the_never_crawled_hosts_and_malformed_hosts_are_refused(host):
    with pytest.raises(ValidationError, match="never crawled|lowercase"):
        Sources.model_validate({"hosts": {host: HOST}})


def test_a_subdomain_of_a_city_host_is_not_the_main_site():
    host = changed(HOST, seeds=["https://digitales-rathaus.darmstadt.de/x"], sitemap=[])
    assert Sources.model_validate({"hosts": {"digitales-rathaus.darmstadt.de": host}})


def test_an_empty_host_list_is_refused():
    with pytest.raises(ValidationError):
        Sources.model_validate({"hosts": {}})


def test_roles_accept_programmes_groups_and_offices():
    roles = Roles.model_validate(ROLES)
    assert roles.programmes["bsc-x"].po_versions[0].start_from == "WiSe 2023/24"
    assert roles.groups["bachelor"].part_of == "students"


@pytest.mark.parametrize(
    ("edits", "needle"),
    [
        ({"programmes|bsc-x|po_versions": [{"year": 2023, **TERMS}, {"year": 2023, **TERMS}]}, "once per programme"),
        ({"programmes|bsc-x|po_versions": []}, "po_versions"),
        ({"programmes|bsc-x|po_versions": [{"year": 2023, **{**TERMS, "start_from": "WS 2023"}}]}, "WiSe 2023/24"),
        ({"programmes|bsc-x|po_versions": [{"year": 2023, **{**TERMS, "start_from": "WiSe 2023/25"}}]}, "consecutive"),
        (
            {
                "programmes|bsc-x|po_versions": [
                    {"year": 2023, **{**TERMS, "start_from": "SoSe 2024", "start_to": "WiSe 2023/24"}}
                ]
            },
            "after start_to",
        ),
        ({"programmes|bsc-x|degree": "promotion"}, "degree"),
        ({"programmes|bsc-x|url_prefix": "http://a.example/x"}, "https URL"),
        ({"programmes|bsc-x|pdf_prefix": "/media/x/"}, "https URL"),
        ({"programmes|bsc-x|po_change": {"text_de": "", "text_en": "x"}}, "text_de"),
        ({"programmes|BSc X": PROGRAMME}, "lowercase words"),
        ({"groups|bachelor|part_of": "nobody"}, "part_of"),
        ({"groups|students|label": 1}, "label"),
        ({"offices|office|url_": None}, "url_"),
        ({"poRule": WORDING}, "poRule"),
    ],
)
def test_roles_that_break_a_rule_are_refused(edits, needle):
    with pytest.raises(ValidationError, match=needle):
        Roles.model_validate(changed(ROLES, **edits))


def test_a_pair_of_terms_in_order_or_with_one_end_open_is_accepted():
    both = {"year": 2015, "start_from": "WiSe 2015/16", "start_to": "SoSe 2023", "official_url": None}
    roles = changed(ROLES, **{"programmes|bsc-x|po_versions": [both, {"year": 2023, **TERMS}]})
    assert [v.year for v in Roles.model_validate(roles).programmes["bsc-x"].po_versions] == [2015, 2023]


def test_acronym_groups_are_strict_and_a_term_belongs_to_one_group():
    good = {"groups": [{"terms": ["PO", "Prüfungsordnung"]}, {"terms": ["MHB", "Modulhandbuch"]}]}
    assert Acronyms.model_validate(good).groups[1].terms == ("MHB", "Modulhandbuch")
    for bad, needle in (
        ({"groups": [{"terms": ["PO"]}]}, "terms"),
        ({"groups": [{"terms": ["PO", " "]}]}, "trimmed"),
        ({"groups": [{"terms": ["PO", "po"]}]}, "two groups"),
        ({"groups": [{"terms": ["PO", "Ordnung"]}, {"terms": ["ordnung", "Regel"]}]}, "two groups"),
        ({"groups": []}, "groups"),
        ({"groups": [{"terms": ["PO", "Ordnung"], "also": 1}]}, "also"),
    ):
        with pytest.raises(ValidationError, match=needle):
            Acronyms.model_validate(bad)
