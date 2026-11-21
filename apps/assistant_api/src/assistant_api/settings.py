"""The assistant's settings: the baked profile (RAG_PROFILE, set in the
image), the model card (RAG__CARD or RAG__CARD_FILE), its messages, and the technical sections from the environment and the secret files. Read once, by the lifespan."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from rag_core.domain.source_meta import SourceLabels
from rag_core.services.prompts import PromptText, load_profile_prompts
from rag_core.settings import OverlaySection, SettingsError, load, load_profile
from rag_core.settings.card import (
    DEFAULT_CARDS_DIR,
    Card,
    load_card,
    with_local_analysis_model,
    with_local_model,
    with_local_url,
)
from rag_core.settings.profile import Acronyms, Guard, Profile, Roles
from rag_core.settings.sections import Compute, DatabaseSection, ModelServersSection, Server, Timeouts, Tracing


class AssistantOverlay(OverlaySection):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())  # model_servers
    database: DatabaseSection = DatabaseSection()
    model_servers: ModelServersSection = ModelServersSection()
    compute: Compute
    timeouts: Timeouts
    tracing: Tracing
    server: Server = Server()


class _Texts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GuardMessages(_Texts):
    not_ready: str
    busy: str
    error: str
    timeout: str


class QuestionMessages(_Texts):
    too_long: str


class VariantMessages(_Texts):
    refused: str


class AnswerMessages(_Texts):
    unavailable: str
    context_too_long: str
    # the sentence that replaces an answer showing a past date as coming ({dates}: the answer's wording)
    past_event: str
    past_events: str
    no_newer_event: str
    # status groups: the prompt lines (scope_compare {groups}: the labels of the groups compared; scope_versions {versions}:
    # the PO years; version_table: the side-by-side format for several versions) and the closing notes ({group}, {other},
    # {asked}: a group's label)
    scope_compare: str = ""
    scope_differ: str = ""
    scope_versions: str = ""
    version_table: str = ""
    scope_note_one: str = ""
    scope_note_other: str = ""
    # the lines appended to an answer from the sources it cites: the PO versions (po_line {entries}: one entry per programme;
    # po_line_term for a PO that follows from a start term; po_one {version}; po_programme {entry}, {programme}), the note on an
    # English information translation of a binding text and the disclaimer on a residence source
    po_line: str = ""
    po_line_term: str = ""
    po_one: str = ""
    po_programme: str = ""
    non_binding: str = ""
    visa_disclaimer: str = ""
    # the line that closes an answer to a search that found nothing under a hard filter: no_match names the filter's one part
    # ({filter}), no_match_more its first part and the others ({first}, {rest}); the parts are the programmes ({labels}) or the
    # degree, and the PO years ({years}). Empty: the answer is left as the model wrote it.
    no_match: str = ""
    no_match_more: str = ""
    filter_programmes: str = ""
    filter_bachelor: str = ""
    filter_master: str = ""
    filter_promotion: str = ""
    filter_po: str = ""


class DocsMessages(_Texts):
    none: str


class Labels(BaseModel):
    """messages.<language>.yaml labels: where a source comes from, and the words that frame the sources in the prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_types: dict[str, str]
    origins: dict[str, str]
    hosts: dict[str, str]
    scope: str  # the source header's Geltungsbereich line
    sources_lead: str  # {n}: the number of sources
    nothing_found: str
    date: str
    page: str
    link: str

    def source_labels(self) -> SourceLabels:
        return SourceLabels(
            self.source_types,
            self.origins,
            self.hosts,
            self.scope,
            sources_lead=self.sources_lead,
            nothing_found=self.nothing_found,
            date=self.date,
            page=self.page,
            link=self.link,
        )


class ShapeMessages(_Texts):
    """messages.<language>.yaml shapes: the instruction line of each answer form and of the length and language variants
    (rag_core.domain.answer_shapes names which one a generation needs), and the reminder that ends an English answer's question."""

    table: str
    checklist: str
    steps: str
    mindmap: str
    shorter: str
    longer: str
    simple: str
    english: str
    english_reminder: str


class Messages(_Texts):
    guard: GuardMessages
    question: QuestionMessages
    variant: VariantMessages
    answer: AnswerMessages
    docs: DocsMessages
    labels: Labels
    shapes: ShapeMessages


@dataclass(frozen=True)
class AssistantSettings:
    profile: Profile
    card: Card
    overlay: AssistantOverlay
    messages: Messages
    prompts: dict[str, PromptText]
    profile_dir: Path
    roles: Roles | None = None  # roles.yaml beside the profile, when it has one
    messages_en: Messages | None = None  # messages.en.yaml beside the profile, when it has one
    acronyms: Acronyms | None = None  # acronyms.yaml beside the profile, when it has one

    @property
    def guard(self) -> Guard:
        return self.profile.guard

    def messages_for(self, language: str | None) -> Messages:
        """The texts for a language, "de" or "en": the profile's messages.<language>.yaml. A language the profile has no file
        for (English on a profile with German texts only), and none, read the German texts."""
        return self.messages_en if language == "en" and self.messages_en else self.messages


def _messages(directory: Path, language: str) -> Messages | None:
    """messages.<language>.yaml beside the profile; None when the profile has no such file."""
    file = directory / f"messages.{language}.yaml"
    return Messages.model_validate(yaml.safe_load(file.read_text(encoding="utf-8"))) if file.exists() else None


def _card(environ: Mapping[str, str]) -> Card:
    """The model card: RAG__CARD names a shipped card (in RAG_CARDS_DIR, the image's /app/cards), RAG__CARD_FILE a card file.
    RAG__LLM_URL, when set, replaces the url of the card's local endpoints and RAG__LLM_MODEL the model of those that serve its
    answer model and RAG__LLM_ANALYSIS_MODEL the model of its own local analysis endpoint."""
    name, file = environ.get("RAG__CARD"), environ.get("RAG__CARD_FILE")
    if bool(name) == bool(file):
        raise SettingsError("RAG__CARD or RAG__CARD_FILE: exactly one names the model card (environment)")
    card = load_card(Path(file) if file else name or "", environ.get("RAG_CARDS_DIR") or DEFAULT_CARDS_DIR)
    if url := environ.get("RAG__LLM_URL"):
        card = with_local_url(card, url)
    if model := environ.get("RAG__LLM_MODEL"):
        card = with_local_model(card, model)
    if model := environ.get("RAG__LLM_ANALYSIS_MODEL"):
        card = with_local_analysis_model(card, model)
    return card


def load_settings(environ: Mapping[str, str] = os.environ, secrets_dir: str | Path | None = None) -> AssistantSettings:
    """The profile named by RAG_PROFILE, the overlay from the environment (and RAG_OVERLAY's file when one is set) and the secret
    files of secrets_dir, else RAG_SECRETS_DIR, else /run/secrets; SettingsError names a key and its source, never a value."""
    path = environ.get("RAG_PROFILE")
    if not path:
        raise SettingsError("RAG_PROFILE: not set (the image sets it; a test passes the profile's path)")
    profile_file = Path(path)
    profile = load_profile(Profile, profile_file)
    card = _card(environ)
    overlay = load(
        AssistantOverlay,
        overlay=environ.get("RAG_OVERLAY") or None,
        environ=environ,
        secrets_dir=secrets_dir,
        profile_sections=frozenset(Profile.model_fields),
    )
    if not overlay.compute.slot_wait_s < card.timeouts.idle_s <= card.timeouts.deadline_s:
        raise SettingsError(
            f"compute.slot_wait_s < timeouts.idle_s <= timeouts.deadline_s (the card {card.name}'s): the guard's "
            "silence limit must not end a request that waits for the slot"
        )
    messages = _messages(profile_file.parent, "de")
    if messages is None:
        raise SettingsError(f"messages.de.yaml: missing beside {profile_file} (the German texts every profile has)")
    prompts = load_profile_prompts(profile_file.parent / "prompts", card.prompt_overlay)
    roles_file = profile_file.parent / "roles.yaml"
    roles = load_profile(Roles, roles_file) if roles_file.exists() else None
    acronyms_file = profile_file.parent / "acronyms.yaml"
    acronyms = load_profile(Acronyms, acronyms_file) if acronyms_file.exists() else None
    return AssistantSettings(
        profile,
        card,
        overlay,
        messages,
        prompts,
        profile_file.parent,
        roles,
        _messages(profile_file.parent, "en"),
        acronyms,
    )
