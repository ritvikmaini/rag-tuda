"""The TU profile's leg: the real indexer and assistant images, built with PROFILE=tu-informatik, crawl the fixture mini-site
(two services, reached under the profile's own host names through the overlay's routes) into the collection `rag` and answer
from it. The crawl follows the profile's sources.yaml, robots.txt and the page rules, so what is in the index and what the
build says it left out are the mini-site's cases: the original of a regulation instead of its copy, the English twin, the
page that bars AI, the print view and login the robots.txt disallows, the programme hub outside the prefixes."""

import re
from typing import Any

import psycopg
import pytest
from rag_testing import harness
from rag_testing.minisite import pages

FACULTY, UNIVERSITY = "https://www.informatik.tu-darmstadt.de", "https://www.intern.tu-darmstadt.de"
HUB = f"{FACULTY}{pages.BSC_HUB}.de.jsp"
PO_PDF = FACULTY + pages.PO_2023_PDF
APB_ORIGINAL, APB_COPY = UNIVERSITY + pages.APB_ORIGINAL_PDF, FACULTY + pages.APB_COPY_PDF
OFFICE = f"{FACULTY}{pages.STUDIENBUERO}/index.de.jsp#kontakt"
REQUEST = re.compile(r"HTTP Request: GET http://([^/\s]+)/")
CITE_ALL = {
    "purpose": "answer",
    "reply": "Antwort " + " ".join(f"[{n}]" for n in range(1, 15)) + ".",
}


@pytest.fixture(scope="module")
def built(first_build: dict[str, Any]) -> Any:
    """The session's first build on the empty database, shared with the rest of the suite."""
    build = first_build["build"]
    assert build.returncode == 0, build.stdout + build.stderr
    return build


def urls(db: psycopg.Connection[Any]) -> set[str]:
    return {r[0] for r in db.execute("SELECT DISTINCT url FROM rag_sections").fetchall()}


def metadata(db: psycopg.Connection[Any], url: str) -> dict[str, Any]:
    row = db.execute("SELECT metadata FROM rag_sections WHERE url = %s", (url,)).fetchone()
    assert row is not None, url
    return dict(row[0])


def test_the_first_build_publishes_generation_1_and_names_what_it_left_out(
    built: Any, db: psycopg.Connection[Any]
) -> None:
    assert harness.live_generation(db) == 1
    lines = built.stdout.splitlines()
    assert lines[0] == "source tu_web: 10 documents"
    assert lines[-1].startswith("generation 1 published (previous none): 9 sections, ")
    assert not [line for line in lines if line.startswith("warning")]  # no empty page, nothing fail-closed
    notes = set(lines[1:-1])
    assert f"note: tu_web: copy skipped, its original is indexed: {APB_COPY}" in notes
    assert f"note: tu_web: not fetched (meta robots): {FACULTY}{pages.NOAI_PAGE}" in notes
    assert f"note: tu_web: not fetched (robots): {FACULTY}{pages.LOGIN_PAGE}" in notes
    assert (
        f"note: tu_web: not fetched (robots): {FACULTY}{pages.STUDIENBUERO}/pruefungen/index.de.jsp?view=print" in notes
    )


def test_the_crawl_asked_the_two_services_and_nothing_else(built: Any) -> None:
    hosts = [h for h in REQUEST.findall(built.stderr) if h != "embedding_service:8001"]  # the build's own model server
    assert len(hosts) > 10  # positive control: the log holds the crawl's requests
    assert set(hosts) == {"minisite:8091", "minisite-intern:8092"}


def test_the_index_holds_the_in_scope_pages_and_pdfs_of_the_two_hosts_only(
    built: Any, db: psycopg.Connection[Any]
) -> None:
    got = urls(db)
    assert got == {
        f"{HUB}#pruefungsordnung_2023",
        f"{HUB}#pruefungsordnung_2015",
        f"{HUB}#allg_pruefungsbest___dateiliste",
        PO_PDF,
        APB_ORIGINAL,
        OFFICE,
        f"{FACULTY}{pages.STUDIENBUERO}/index.en.jsp#kontakt",
        f"{FACULTY}{pages.STUDIENBUERO}/pruefungen/index.de.jsp#pruefungsanmeldung",
        f"{UNIVERSITY}{pages.INTERN_INDEX}#apb",
    }
    # the copy yields to the original, the English hub is a fallback twin of the German text, the other programme's hub and
    # the English twin of the German-body page are out; so is everything the crawl did not reach
    assert not {APB_COPY, f"{FACULTY}{pages.BSC_HUB}.en.jsp", FACULTY + pages.EXCLUDED_HUB + ".de.jsp"} & got


def test_the_documents_carry_their_typed_metadata_and_the_university_wide_regulation_no_scope(
    built: Any, db: psycopg.Connection[Any]
) -> None:
    for url, version in (
        (f"{HUB}#pruefungsordnung_2023", [2023]),
        (f"{HUB}#pruefungsordnung_2015", [2015]),
        (PO_PDF, [2023]),
    ):
        meta = metadata(db, url)
        assert (meta["regulation"], meta["degree"], meta["programme"], meta["po_version"], meta["lang"]) == (
            "po",
            ["bachelor"],
            ["bsc-informatik"],
            version,
            "de",
        ), url
    apb = metadata(db, APB_ORIGINAL)  # the university-wide regulation applies to every programme: no scope keys, not []
    assert apb["regulation"] == "apb" and not {"degree", "programme", "po_version"} & apb.keys()
    german, english = metadata(db, OFFICE), metadata(db, f"{FACULTY}{pages.STUDIENBUERO}/index.en.jsp#kontakt")
    assert (german["lang"], english["lang"]) == ("de", "en") and german["twin_key"] == english["twin_key"]


@pytest.mark.parametrize(
    ("question", "first", "also"),
    [
        (harness.THESIS, None, {PO_PDF}),
        (harness.RETAKE, APB_ORIGINAL, set()),
        (harness.OFFICE_HOURS, OFFICE, set()),
    ],
)
def test_a_lookup_cites_the_sections_that_state_the_answer(
    built: Any,
    assistant: harness.Assistant,
    llm: harness.FakeLLM,
    user: str,
    question: str,
    first: str | None,
    also: set[str],
) -> None:
    llm.script(CITE_ALL)
    reply = assistant.ask(question, user)
    assert reply.path == harness.LOOKUP
    cited = [e["url"] for e in reply.first("citations")]
    assert cited and all(u.startswith((FACULTY, UNIVERSITY)) for u in cited)
    assert (first is None or cited[0] == first) and also <= set(cited)
    assert APB_COPY not in cited
