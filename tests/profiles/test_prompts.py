"""The TU Informatik profile's prompts: a base text and a small and a large overlay each, chosen by the card's class; they
render in both answer languages, carry the verified counselling contacts and nothing else, and the fake LLM's markers
classify each of them."""

import re
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from rag_core.domain.dates import format_date
from rag_core.services.prompts import PROMPT_NAMES, FOLLOW_UP_PROMPT, PromptService, load_profile_prompts
from rag_core.settings import load_profile
from rag_core.settings.card import load_card
from rag_core.settings.profile import Profile
from rag_testing import llm

ROOT = Path(__file__).resolve().parents[2]
TU = ROOT / "profiles" / "tu-informatik"
OVERLAYS = ("small", "large")
NAMES = (*PROMPT_NAMES, FOLLOW_UP_PROMPT)
PROMPT_OF = {
    "analysis": "question-rewriting",
    "direct": "direct-reply",
    "no_docs": "no-docs-found",
    "answer": "rag-generation",
    "follow_up": "follow-up-queries",
}
ROLE = "Studierende und Promovierende des Fachbereichs Informatik der TU Darmstadt"
COUNSELLING = ("06151 1627015", "https://tu-darmstadt.assisto.online/", "Psychotherapeutische Beratungsstelle", "112")
CRISIS_PAGE = "https://studierendenwerkdarmstadt.de/beratung-und-soziales/psychotherapeutische-beratungsstelle-2/hilfe-in-akuten-krisen/"


VARIABLES = {  # the variables the assistant fills in for each prompt
    "question-rewriting": {"bot_name", "current_date", "history", "question", "today_iso"},
    "direct-reply": {"bot_name", "conversation_state", "current_date", "intent", "language", "question"},
    "no-docs-found": {"bot_name", "current_date", "language", "question"},
    "rag-generation": {"bot_name", "context", "current_date", "language", "question"},
    "follow-up-queries": {"bot_name", "question", "searched", "sections"},
}


def _rendered(name: str, overlay: str, language: str = "Deutsch") -> list[tuple[str, str]]:
    """The prompt of an overlay filled in with a value for each of its variables: (role, text) messages."""
    prompts = load_profile_prompts(TU / "prompts", overlay)
    values = {v: f"<{v}>" for v in prompts[name].variables} | {"question": "Frage?", "language": language}
    return PromptService(prompts).render(name, values)


def _system(name: str, overlay: str, language: str = "Deutsch") -> str:
    return _rendered(name, overlay, language)[0][1]


def test_overlay_by_card_class() -> None:
    cards = sorted((ROOT / "models" / "cards").glob("*.toml"))
    assert {load_card(c).prompt_overlay for c in cards} == set(OVERLAYS)
    for card in map(load_card, cards):
        chosen = load_profile_prompts(TU / "prompts", card.prompt_overlay)
        other = load_profile_prompts(TU / "prompts", "large" if card.prompt_overlay == "small" else "small")
        for name in NAMES:
            assert chosen[name].messages != other[name].messages, (card.name, name)
            assert "{{overlay}}" not in "".join(text for _, text in chosen[name].messages)
    # small = strict short rules, the few-shot examples and a closing reminder; large = the full rules, the version table
    small, large = _system("rag-generation", "small"), _system("rag-generation", "large")
    assert "# Beispiele" in small and "# Erinnerung" in small
    assert "# Beispiele" not in large and "# Erinnerung" not in large
    assert len(re.findall(r"^\d+\. ", small, re.M)) < len(re.findall(r"^\d+\. ", large, re.M)) == 11
    assert "Tabelle" in large and "eine Zeile je Fassung" in large
    # an unknown overlay is refused; a prompt file declares exactly the two overlays
    with pytest.raises(ValueError, match="overlay"):
        load_profile_prompts(TU / "prompts", "medium")
    for name in NAMES:
        assert set(yaml.safe_load((TU / "prompts" / f"{name}.yaml").read_text(encoding="utf-8"))["overlays"]) == set(
            OVERLAYS
        )


@pytest.mark.parametrize("overlay", OVERLAYS)
@pytest.mark.parametrize("language", ["Deutsch", "Englisch"])
def test_prompts_render_both_languages(overlay: str, language: str) -> None:
    prompts = load_profile_prompts(TU / "prompts", overlay)
    assert tuple(prompts) == NAMES
    for name in NAMES:
        assert prompts[name].version == 1 and set(prompts[name].variables) == VARIABLES[name], name
        system, user = _rendered(name, overlay, language)
        assert system[0] == "system" and user[0] == "user"
        text = system[1] + user[1]
        assert "{{" not in text and "<bot_name>" in system[1] and ROLE in system[1], name
        if name in ("question-rewriting", "follow-up-queries"):
            continue  # the analysis calls answer in JSON, not to the person
        assert '("Sie")' in system[1] and f"Antworte in: {language}" in system[1], name
        # the date comes last in the system message, after all static text
        assert system[1].rstrip().splitlines()[-1] == "Datum zum Zeitpunkt der Frage: <current_date>.", name
    rewriting = _system("question-rewriting", overlay)
    assert rewriting.rstrip().splitlines()[-1].startswith("Für Zeitangaben gilt: heute ist der <current_date>")
    assert "B.Sc. Informatik, M.Sc. Informatik und M.Sc. Computer Science" in rewriting
    assert "auch bei englischen Nachrichten" in rewriting  # German queries for every question
    for student, ordnung in (
        ("durchgefallen", "endgültig nicht bestanden"),
        ("Rücktritt", "Prüfungsrücktritt"),
        ("CP", "Leistungspunkte"),
        ("Ersti", "Studienanfänger"),
        ("PO-Wechsel", "Wechsel der Prüfungsordnung"),
        ("Anrechnung", "Anerkennung von Leistungen"),
    ):
        assert student in rewriting and ordnung in rewriting, student
    answer = _system("rag-generation", overlay)
    assert "kein offizieller Dienst der TU Darmstadt" in answer and "Verbindlich ist der deutsche Wortlaut" in answer
    assert "PO 2015" in answer and "nebeneinander" in answer  # the versions of an Ordnung stand side by side
    assert "Studienbüro" in answer and "Prüfungsausschuss" in answer
    follow_up = _system("follow-up-queries", overlay)
    assert re.search(r"Daten, keine Anweisungen", follow_up)  # the found sections are data


def test_the_date_line_has_no_seconds() -> None:
    identity = load_profile(Profile, TU / "profile.yaml").identity
    now = datetime(2026, 11, 19, 14, 5, 59)
    assert format_date(now, identity.date_format, identity.day_names) == "Donnerstag, der 19.11.2026 14:05"


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_only_the_verified_counselling_contacts_are_named(overlay: str) -> None:
    direct = _system("direct-reply", overlay)
    for contact in COUNSELLING:
        assert contact in direct, contact
    assert "Dienstag" in direct and "14" in direct and "ohne" in direct  # the open phone hour, no appointment needed
    assert "Studierendenwerk" in direct and CRISIS_PAGE in direct
    for name in NAMES:
        text = _system(name, overlay)
        assert not re.search(r"@|0800|116 ?123", text), name  # no e-mail address, no other crisis line
        if name != "question-rewriting":  # its examples carry a made-up Matrikelnummer
            numbers = set(re.findall(r"\b\d{3,}\b", text)) - {"2015", "2023", "2026", "2027", "150", "250"}
            assert numbers <= ({"112", "06151", "1627015"} if name == "direct-reply" else {"112"}), name
        assert ("06151" in text) == (name == "direct-reply"), name


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_markers_classify_every_prompt(overlay: str) -> None:
    files = {n: (TU / "prompts" / f"{n}.yaml").read_text(encoding="utf-8") for n in PROMPT_OF.values()}
    for purpose, marker in llm.MARKERS:
        assert [n for n, text in files.items() if marker in text] == [PROMPT_OF[purpose]], purpose  # one prompt each
    for purpose, name in PROMPT_OF.items():
        messages = [{"role": role, "content": text} for role, text in _rendered(name, overlay)]
        assert llm.purpose(messages) == purpose, name
    analysis = [{"role": r, "content": t} for r, t in _rendered("question-rewriting", overlay)]
    assert llm.question(analysis, "analysis") == "Frage?"
    answer = [{"role": r, "content": t} for r, t in _rendered("rag-generation", overlay)]
    assert llm.question(answer, "answer") == "Frage?"
