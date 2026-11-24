"""The TU Informatik profile: its files load with the schemas, the sources and roles keep to what the pages state, and the
prompts keep their variables and the fake LLM's markers."""

from pathlib import Path
from urllib.parse import urlsplit

import yaml
from assistant_api.settings import Messages
from indexer.crawl.discover import Source
from indexer.crawl.extract import Selectors
from rag_core.settings import load_profile
from rag_core.settings.profile import Acronyms, Profile, Roles, Sources

ROOT = Path(__file__).resolve().parents[2]
TU = ROOT / "profiles" / "tu-informatik"
HUBS = "https://www.informatik.tu-darmstadt.de/studium_fb20/im_studium/formulare_und_dokumente/ordnungen/"


def _shape(tree):
    """The keys of a loaded YAML tree down to its texts."""
    return {k: _shape(v) if isinstance(v, dict) else None for k, v in tree.items()}


def test_profile_and_roles_load():
    profile = load_profile(Profile, TU / "profile.yaml")
    assert [(c.name, c.audiences) for c in profile.collections] == [("rag", ["public"])]
    assert list(profile.ingest.sources) == ["tu_web", "laws"]
    assert profile.ingest.sources["tu_web"].connector == "tuda_web"
    assert profile.ingest.sources["laws"].connector == "law_xml"
    assert profile.generation.analysis_max_tokens == 800
    assert profile.identity.language == "de" and profile.ranking.recency_types == []
    assert profile.ranking.relevance_floor == 0.1

    roles = load_profile(Roles, TU / "roles.yaml")
    assert {p: [v.year for v in r.po_versions] for p, r in roles.programmes.items()} == {
        "bsc-informatik": [2023, 2015],
        "msc-informatik": [2023, 2015],
        "msc-computer-science": [2023],
    }
    assert {p: r.degree for p, r in roles.programmes.items()} == {
        "bsc-informatik": "bachelor",
        "msc-informatik": "master",
        "msc-computer-science": "master",
    }
    # a start term is written only where a hub page states it
    terms = {
        (p, v.year): (v.start_from, v.start_to)
        for p, r in roles.programmes.items()
        for v in r.po_versions
        if v.start_from or v.start_to
    }
    assert terms == {
        ("msc-informatik", 2023): ("WiSe 2023/24", None),
        ("msc-computer-science", 2023): ("WiSe 2023/24", None),
    }
    assert roles.po_rule.text_de.startswith("In der Regel gelten für Sie")
    change = roles.programmes["bsc-informatik"].po_change
    assert (
        change is not None
        and "Erklärung zum Wechsel" in change.text_de
        and change == roles.programmes["msc-informatik"].po_change
    )
    assert roles.programmes["msc-computer-science"].po_change is None
    for programme in roles.programmes.values():
        assert programme.url_prefix.startswith(HUBS)
    assert set(roles.offices) == {
        "studienbuero",
        "pruefungsausschuss",
        "studienberatung",
        "promotionsbuero",
        "international-office",
        "auslaenderbehoerde-stadt",
        "auslaenderbehoerde-landkreis",
    }
    assert all(o.url is None or o.url.startswith("https://") for o in roles.offices.values())
    assert roles.groups["bachelor"].part_of == "studierende" and roles.groups["promotion"].side == "promotion"
    assert set(roles.programmes) <= set(roles.groups)  # a programme group per programme id

    acronyms = load_profile(Acronyms, TU / "acronyms.yaml")
    terms = {t for g in acronyms.groups for t in g.terms}
    assert {
        "PO",
        "APB",
        "MHB",
        "SPP",
        "CP",
        "LP",
        "ESO",
        "Ersti",
        "Prüfungsrücktritt",
        "Anerkennung",
        "Härtefall",
    } <= terms
    assert {"PO-Wechsel", "Studiengang", "Studienprogramm"} <= terms


def test_sources_are_the_planned_hosts_and_build_the_crawlers_sources():
    sources = load_profile(Sources, TU / "sources.yaml")
    assert {h: e.set for h, e in sources.hosts.items()} == {
        "www.informatik.tu-darmstadt.de": "core",
        "www.intern.tu-darmstadt.de": "core",
        "www.tu-darmstadt.de": "core",
        "digitales-rathaus.darmstadt.de": "core",
        "www.ladadi.de": "full",
        "www.gesetze-im-internet.de": "core",
    }
    assert [(law.start, law.end) for law in sources.laws] == [("§ 16b", "§ 16b"), ("§ 45", "§ 45")]
    assert sources.hosts["www.tu-darmstadt.de"].full is not None
    for host, entry in sources.hosts.items():
        prefixes = tuple(entry.prefixes)
        # both the core and the full block are accepted by the crawler's own types (selectors included)
        Source(
            host=host,
            prefixes=prefixes,
            exclude=tuple(entry.exclude),
            sitemaps=tuple(entry.sitemap),
            seeds=tuple(entry.seeds),
            depth=entry.depth,
            twins=entry.twins,
        )
        if entry.full:
            Source(
                host=host,
                prefixes=tuple(entry.full.prefixes),
                exclude=tuple(entry.full.exclude),
                sitemaps=tuple(entry.full.sitemap),
                seeds=tuple(entry.seeds),
                depth=entry.full.depth,
            )
        Selectors(
            main=entry.selectors.main,
            section=entry.selectors.section,
            **({"drop": entry.selectors.drop} if entry.selectors.drop else {}),
        )
        if prefixes:  # a seed of a host with prefixes is under one of them
            assert all(any(urlsplit(s).path.startswith(p) for p in prefixes) for s in entry.seeds), host
    # the three programmes' hubs are seeds of the FB20 host and sit under their programme's url_prefix
    roles = load_profile(Roles, TU / "roles.yaml")
    fb20 = sources.hosts["www.informatik.tu-darmstadt.de"]
    for programme in roles.programmes.values():
        assert any(s.startswith(programme.url_prefix + ".") for s in fb20.seeds), programme.label_de


FB20 = "https://www.informatik.tu-darmstadt.de/media/informatik/"
PO_FOLDER = FB20 + "fb20_studium/formulare_und_dokumente/ordnungen/"
PROGRAMME_PDFS = {  # the PDFs the programme hubs link, by programme
    "bsc-informatik": [
        PO_FOLDER + "ordnungen_bsc/pruefungsordnung_2023/2022_09_30_MHB_BSC_INF.pdf",
        PO_FOLDER + "ordnungen_bsc/pruefungsordnung_2023/Studiengangsordnung_Bachelor_Informatik_2023.pdf",
        PO_FOLDER + "ordnungen_bsc/pruefungsordnung_2023/SPP_B.Sc._Informatik.pdf",
    ],
    "msc-informatik": [  # the folder is named pruefungsordnung_9, not by the year
        PO_FOLDER + "ordnungen_msc/pruefungsordnung_9/2023_05_11_MHB_MSC_INF.pdf",
        PO_FOLDER + "ordnungen_msc/pruefungsordnung_9/Ordnung_des_Studiengangs_M_Sc_Informatik_PO_2023.pdf",
        PO_FOLDER + "ordnungen_msc/pruefungsordnung_9/Studien-_und_Pruefungsplan_M_Sc_Informatik_PO_2023.pdf",
    ],
    "msc-computer-science": [
        PO_FOLDER + "ordnungen_cs/pruefungsordnungen_cs_2023/2023_05_11_MHB_MSC_CS.de.pdf",
        PO_FOLDER
        + "ordnungen_cs/pruefungsordnungen_cs_2023/Studien-_und_Pruefungsplan_M_Sc_Computer_Science_PO_2023.de.pdf",
    ],
}
OTHER_PDFS = [  # promotion documents and the university's APB
    FB20 + "fb20_dekanat/dokumente_promotion/Allgemeine_Bestimmungen_der_Promotionsordnung_der_TU_Darmstadt.pdf",
    FB20 + "fb20_dekanat/dokumente_promotion/Besondere_Bestimmungen_FB20_30_08_SB_2024_VI.pdf",
    "https://www.intern.tu-darmstadt.de/media/dezernat_ii/ordnungen/apb.pdf",
]


def _listed(entry, url):
    """Whether the crawl lists a URL of the host: under a prefix and under no exclusion."""
    path = urlsplit(url).path
    return any(path.startswith(p) for p in entry.prefixes) and not any(path.startswith(p) for p in entry.exclude)


def test_the_pdfs_the_pages_link_are_listed_and_matched_to_their_programme():
    sources = load_profile(Sources, TU / "sources.yaml")
    roles = load_profile(Roles, TU / "roles.yaml")
    for programme, urls in PROGRAMME_PDFS.items():
        for url in urls:
            assert _listed(sources.hosts[urlsplit(url).netloc], url), url
            assert url.startswith(roles.programmes[programme].pdf_prefix), url
            assert not any(url.startswith(r.pdf_prefix) for p, r in roles.programmes.items() if p != programme), url
    for url in OTHER_PDFS:
        assert _listed(sources.hosts[urlsplit(url).netloc], url), url
    # other programmes' folders and unrelated media stay out
    fb20 = sources.hosts["www.informatik.tu-darmstadt.de"]
    for path in ("fb20_studium/formulare_und_dokumente/ordnungen/ordnungen_other/x.pdf", "fb20_news/x.pdf"):
        assert not _listed(fb20, FB20 + path), path


def test_every_crawled_host_has_a_label_in_both_languages_and_the_languages_match():
    sources = load_profile(Sources, TU / "sources.yaml")
    de = yaml.safe_load((TU / "messages.de.yaml").read_text(encoding="utf-8"))
    en = yaml.safe_load((TU / "messages.en.yaml").read_text(encoding="utf-8"))
    for tree in (de, en):
        messages = Messages.model_validate(tree)
        assert set(sources.hosts) <= set(messages.labels.hosts)
    assert _shape(de) == _shape(en)
    assert de["docs"]["none"].startswith("Hinweis") and en["docs"]["none"].startswith("Note")
    for text in (de["question"]["too_long"], en["question"]["too_long"]):
        assert "{limit}" in text
    for key in ("past_event", "past_events"):
        assert "{dates}" in de["answer"][key] and "{dates}" in en["answer"][key]
