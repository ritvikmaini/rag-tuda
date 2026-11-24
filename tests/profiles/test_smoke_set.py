"""The TU Informatik smoke set: its shape, and that its gold quotes are still in the latest local crawl.

The schema test runs everywhere. The crawl test reads `.rag/snapshots` and `.rag/crawl` of a local build and is skipped
without them; a gold quote that is in no text of its url is stale (the page changed): the test names the stale questions, and
a stale volatile one (a fee or a date) is only a warning."""

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from evals.gate import conversations, load_config
from indexer.crawl.cache import Cache
from indexer.crawl.extract import Selectors, extract
from indexer.connectors.law_xml import law_markdown, xml_of_zip
from indexer.crawl.manifest import manifest_urls, snapshots_dir
from rag_core.settings import load_profile
from rag_core.settings.profile import Roles, Sources

ROOT = Path(__file__).resolve().parents[2]
TU = ROOT / "profiles" / "tu-informatik"
RAG = ROOT / ".rag"
KINDS = {
    "bsc", "msc", "cs", "po_compare", "apb", "deadline", "visa", "phd", "english", "trap", "decline", "multi", "table",
    "widening", "eso",
}  # fmt: skip
CHECKS = {
    "contains_any", "contains_none", "digits_any", "source_url_any", "polarity", "group_values", "po_line_present",
    "visa_disclaimer_present", "sie_form", "answer_language_matches", "citations_map_to_sources",
}  # fmt: skip
CONTENT = ("contains_any", "digits_any", "source_url_any")  # the checks on what the answer says or cites
# pages that answer the question without being a gold: a `source_url_any` pattern of them matches no gold url
EXTRA_URL_PATTERNS = {
    "S01": {"abschlussarbeiten_fb20"},  # the thesis periods of the Studienbüro page
    "S03": {"bewerbung_2.en.jsp"},  # the application page of the English-taught masters
    "S04": {"themen_9088"},  # the topic page of the Wissenssammlung entry
    "S07": {"artikel_details_de_en_286144"},  # the semester fee page names the re-registration period
    "S08": {"bewerbung_2.en.jsp"},
    "S12": {"bewerbung_2.en.jsp"},
    "S29": {"bewerbung_2.en.jsp"},  # the application page's part-time work rule (140 full days)
    "S13": {"abschlussarbeiten_fb20"},
}
MINIMUM = {"english": 5, "trap": 3, "decline": 3, "multi": 2, "table": 2}  # the mix of the spec; the set may hold more
WIDENING = [
    ("Kann ich schon Mastermodule belegen?", {"degree": "bachelor"}),
    ("Was hat sich zwischen der PO 2015 und der PO 2023 bei der Masterarbeit geändert?", {"degree": "bachelor"}),
]


ABBREVIATION = re.compile(r"\b(?:[BM]\.Sc|z\.\s?B|Nr|Abs|Art)\.")


def squeeze(text: str) -> str:
    return " ".join(text.split())


def load() -> list[dict[str, Any]]:
    """The conversations as `evals` reads them through the profile's gate.yaml."""
    return [conv for _, conv in conversations(str(TU), load_config(str(TU))).values()]


def sources_of(gold: dict[str, Any]) -> list[dict[str, Any]]:
    return [gold, *gold.get("also", [])]


def test_smoke_set_schema() -> None:
    convs = load()
    gate = load_config(str(TU))
    assert json.loads((TU / "smoke" / "smoke_v1.json").read_text(encoding="utf-8"))["version"] == "smoke-v1"
    roles = load_profile(Roles, TU / "roles.yaml")
    hosts = set(load_profile(Sources, TU / "sources.yaml").hosts)
    ids = [c["id"] for c in convs]
    assert len(set(ids)) == len(ids) >= 35  # S01 to S30 on the core set, S31 to S35 on the full set's sources
    assert gate.sets["smoke"].file == "smoke/smoke_v1.json" and gate.core == ["smoke"]
    assert set(gate.retrieval.volatile) <= set(ids)
    assert all(p.id in ids and p.set == "smoke" and p.kind == "lookup" for p in gate.smoke.probes)
    kinds = [c["kind"] for c in convs]
    assert set(kinds) <= KINDS and all(kinds.count(kind) >= n for kind, n in MINIMUM.items())
    for conv in convs:
        assert conv["set"] == "smoke" and len(conv["turns"]) == 1, conv["id"]
        turn = conv["turns"][0]
        expect = turn["expect"]
        assert turn["q"].strip() and set(expect) <= CHECKS, conv["id"]
        assert any(expect.get(check) for check in CONTENT), conv[
            "id"
        ]  # a fact, a source or a decline phrase is expected
        english = expect.get("answer_language_matches") == "en"
        assert english == (conv["kind"] == "english") and ("sie_form" in expect) == (not english), conv["id"]
        assert ("gold" in turn) != (conv["kind"] == "decline"), conv["id"]  # a decline has no gold source
        for key, value in (turn.get("chips") or {}).items():
            assert key in ("degree", "programme", "po"), conv["id"]
            if key == "degree":
                assert value in ("bachelor", "master", "promotion")
            if key == "programme":
                assert value in roles.programmes
        if programme := (turn.get("chips") or {}).get("programme"):
            assert not (po := turn["chips"].get("po")) or po in [
                v.year for v in roles.programmes[programme].po_versions
            ]
        if "gold" not in turn:
            assert expect.get("contains_any"), conv["id"]  # a decline names the phrases that mark it
            continue
        gold = turn["gold"]
        assert gold.get("rule", "any") in ("any", "all") and bool(gold.get("also")) == ("rule" in gold), conv["id"]
        assert gold.get("also", [None]), conv["id"]  # no empty `also`
        urls = " ".join(s["url"] for s in sources_of(gold))
        for pattern in expect.get("source_url_any", ()):
            assert pattern in urls or pattern in EXTRA_URL_PATTERNS.get(conv["id"], ()), (conv["id"], pattern)
        for source in sources_of(gold):
            assert urlsplit(source["url"]).hostname in hosts, (conv["id"], source["url"])
            heading, page = source.get("heading"), source.get("page")
            assert (isinstance(heading, str) and heading.strip()) or (isinstance(page, int) and page > 0), conv["id"]
            quote = source["quote"]
            assert 20 <= len(quote) <= 400 and "\n" not in quote, conv["id"]  # one sentence, whitespace-normalised
            assert quote == squeeze(quote), conv["id"]
            assert not re.search(r"(?<!\d)[.!?]\s+[A-ZÄÖÜ]", ABBREVIATION.sub("", quote)), conv[
                "id"
            ]  # a second sentence
            if "po_version" in source:
                assert source["po_version"] in (2015, 2023), conv["id"]
                assert expect.get("po_line_present") is True, conv[
                    "id"
                ]  # a version-bound source: the PO line is checked
    assert [(c["turns"][0]["q"], c["turns"][0].get("chips")) for c in convs if c["kind"] == "widening"] == WIDENING


def latest_texts(urls: set[str]) -> dict[str, list[str]]:
    """url -> the whitespace-squeezed texts of the latest crawl: a PDF's extracted text, a page's documents."""
    snapshots = snapshots_dir(RAG / "crawl")
    if not (generations := [int(p.stem) for p in snapshots.glob("*.json")]):
        pytest.skip("no snapshot manifest in .rag/snapshots")
    generation = max(generations)
    listed = {
        u.partition("#")[0] for u in manifest_urls(str(generation), directory=snapshots)
    }  # a split page lists its parts
    sources = load_profile(Sources, TU / "sources.yaml")
    cache = Cache(RAG / "crawl")
    laws = {law.page_url: law for law in sources.laws}
    pdfs: dict[str, list[str]] = {}
    for path in (RAG / "crawl" / "extracted").glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        pdfs.setdefault(entry["download_url"], []).append(squeeze(entry["text"]))
    texts: dict[str, list[str]] = {}
    for url in urls & listed:
        if url in pdfs:
            texts[url] = pdfs[url]
        elif url in laws and (entry := cache.entry(laws[url].xml_url)):  # a statute is read from its XML, not crawled
            law = laws[url]
            texts[url] = [squeeze(law_markdown(xml_of_zip(cache.body(entry)), law.start, law.end)[1])]
        elif entry := cache.entry(url):
            selectors = {
                k: v for k, v in sources.hosts[urlsplit(url).hostname or ""].selectors.model_dump().items() if v
            }
            page = extract(cache.body(entry), url, Selectors(**selectors), entry.content_type)
            texts[url] = [squeeze(d.markdown) for d in page.documents]
    return texts


def test_an_empty_snapshots_folder_skips_the_crawl_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "snapshots").mkdir()
    monkeypatch.setattr(sys.modules[__name__], "RAG", tmp_path)
    with pytest.raises(pytest.skip.Exception):
        latest_texts({"https://example.org/"})


@pytest.mark.local_crawl
def test_gold_present_in_latest_snapshot() -> None:
    if not (RAG / "snapshots").is_dir() or not (RAG / "crawl").is_dir():
        pytest.skip("no local build: .rag/snapshots and .rag/crawl are missing")
    volatile = set(load_config(str(TU)).retrieval.volatile)
    convs = [c for c in load() if "gold" in c["turns"][0]]
    texts = latest_texts({s["url"].partition("#")[0] for c in convs for s in sources_of(c["turns"][0]["gold"])})
    stale = {
        c["id"]: s["url"]
        for c in convs
        for s in sources_of(c["turns"][0]["gold"])
        if not any(squeeze(s["quote"]) in t for t in texts.get(s["url"].partition("#")[0], ()))
    }
    if drifted := sorted(set(stale) & volatile):
        warnings.warn(f"stale volatile golds (a fee or a date changed): {', '.join(drifted)}", stacklevel=1)
    assert not {k: v for k, v in stale.items() if k not in volatile}, "stale golds, the quote is in no text of its url"
