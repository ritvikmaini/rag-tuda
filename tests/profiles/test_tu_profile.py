"""The TU Informatik profile: its files load with the schemas, the sources and roles keep to what the pages state, and the
prompts keep their variables and the fake LLM's markers."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from assistant_api.settings import Messages
from indexer.crawl.discover import Source
from indexer.crawl.extract import Selectors
from rag_core.services.prompts import PROMPT_NAMES, PromptService, load_profile_prompts
from rag_core.settings import load_profile
from rag_core.settings.profile import Acronyms, Profile, Roles, Sources
from rag_testing import llm

ROOT = Path(__file__).resolve().parents[2]
TU = ROOT / "profiles" / "tu-informatik"
PROMPT_OF = {
    "analysis": "question-rewriting",
    "direct": "direct-reply",
    "no_docs": "no-docs-found",
    "answer": "rag-generation",
}
VARIABLES = {  # the variables the assistant fills in for each prompt
    "question-rewriting": {"bot_name", "current_date", "history", "question", "today_iso"},
    "direct-reply": {"bot_name", "conversation_state", "current_date", "intent", "language", "question"},
    "no-docs-found": {"bot_name", "current_date", "language", "question"},
    "rag-generation": {"bot_name", "context", "current_date", "language", "question"},
}
HUBS = "https://www.informatik.tu-darmstadt.de/studium_fb20/im_studium/formulare_und_dokumente/ordnungen/"


def _shape(tree):
    """The keys of a loaded YAML tree down to its texts."""
    return {k: _shape(v) if isinstance(v, dict) else None for k, v in tree.items()}


def test_profile_and_roles_load():
    profile = load_profile(Profile, TU / "profile.yaml")
    assert [(c.name, c.audiences) for c in profile.collections] == [("rag", ["public"])]
    assert list(profile.ingest.sources) == ["tu_web"]
    assert profile.ingest.sources["tu_web"].connector == "tuda_web"
    assert profile.generation.analysis_max_tokens == 800
    assert profile.identity.language == "de" and profile.ranking.recency_types == []

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
    }
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


def _rendered(name):
    """The prompt filled in with a value for each of its variables: (role, text) messages."""
    prompts = load_profile_prompts(TU / "prompts")
    values = {v: f"<{v}>" for v in prompts[name].variables} | {"question": "Frage?", "language": "Deutsch"}
    return PromptService(prompts).render(name, values)


def test_prompts_render_de():
    prompts = load_profile_prompts(TU / "prompts")
    assert tuple(prompts) == PROMPT_NAMES
    for name, prompt in prompts.items():
        assert prompt.version == 1 and set(prompt.variables) == VARIABLES[name], name
        system, user = _rendered(name)
        assert system[0] == "system" and user[0] == "user"
        assert "{{" not in system[1] + user[1] and "<bot_name>" in system[1]
        if name != "question-rewriting":  # the analysis answers in JSON, not to the person
            assert 'mit "Sie" an' in system[1], name  # German answers address the person with Sie
            assert "Antworte in: Deutsch" in system[1], name
        assert "Studierende und Promovierende des Fachbereichs Informatik der TU Darmstadt" in system[1], name
    rules = _rendered("rag-generation")[0][1]
    assert [int(n) for n in re.findall(r"^(\d+)\. ", rules, re.M)] == list(range(1, 12))
    assert "Maßgeblich ist der deutsche Wortlaut" in rules and "Rechtsberatung" in rules
    assert "PO 2015" in rules and "nebeneinander" in rules  # the versions of an Ordnung stand side by side
    assert "kein offizieller Dienst der TU Darmstadt" in rules
    direct = _rendered("direct-reply")[0][1]
    assert "kein offizieller Dienst der TU Darmstadt" in direct and "112" in direct and "116 117" in direct
    for name in ("direct-reply", "no-docs-found", "rag-generation"):
        text = _rendered(name)[0][1]
        assert not re.search(r"@|06151", text), name  # no contact of the university is named
        assert set(re.findall(r"\b\d{3,}\b", text)) <= {
            "110",
            "111",
            "112",
            "116",
            "117",
            "222",
            "0800",
            "250",
            "2015",
            "2023",
        }, name
    rewriting = _rendered("question-rewriting")[0][1]
    assert "B.Sc. Informatik, M.Sc. Informatik und M.Sc. Computer Science" in rewriting
    assert "auch bei einer englischen Nachricht" in rewriting  # German queries for every question


def test_markers_classify_v0_prompts():
    texts = {n: (TU / "prompts" / f"{n}.yaml").read_text(encoding="utf-8") for n in PROMPT_OF.values()}
    for purpose, marker in llm.MARKERS:
        assert [n for n, text in texts.items() if marker in text] == [PROMPT_OF[purpose]], purpose
    for purpose, name in PROMPT_OF.items():
        messages = [{"role": role, "content": text} for role, text in _rendered(name)]
        assert llm.purpose(messages) == purpose, name
    analysis = [{"role": r, "content": t} for r, t in _rendered("question-rewriting")]
    assert llm.question(analysis, "analysis") == "Frage?"
    answer = [{"role": r, "content": t} for r, t in _rendered("rag-generation")]
    assert llm.question(answer, "answer") == "Frage?"
