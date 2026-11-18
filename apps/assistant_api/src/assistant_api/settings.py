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
from rag_core.settings.card import DEFAULT_CARDS_DIR, Card, load_card, with_local_url
from rag_core.settings.profile import Guard, Profile
from rag_core.settings.sections import Compute, DatabaseSection, Http, ModelServersSection, Server, Timeouts, Tracing


class AssistantOverlay(OverlaySection):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())  # model_servers
    database: DatabaseSection = DatabaseSection()
    model_servers: ModelServersSection = ModelServersSection()
    compute: Compute
    timeouts: Timeouts
    http: Http
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
    # status groups: the prompt lines ({groups}: the labels of the groups compared) and the closing notes ({group},
    # {other}, {asked}: a group's label)
    scope_compare: str = ""
    scope_differ: str = ""
    scope_note_one: str = ""
    scope_note_other: str = ""


class DocsMessages(_Texts):
    none: str


class Labels(BaseModel):
    """messages.<language>.yaml labels: where a source comes from."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_types: dict[str, str]
    origins: dict[str, str]
    hosts: dict[str, str]
    scope: str  # the source header's Geltungsbereich line

    def source_labels(self) -> SourceLabels:
        return SourceLabels(self.source_types, self.origins, self.hosts, self.scope)


class Messages(_Texts):
    guard: GuardMessages
    question: QuestionMessages
    variant: VariantMessages
    answer: AnswerMessages
    docs: DocsMessages
    labels: Labels


@dataclass(frozen=True)
class AssistantSettings:
    profile: Profile
    card: Card
    overlay: AssistantOverlay
    messages: Messages
    prompts: dict[str, PromptText]
    profile_dir: Path

    @property
    def guard(self) -> Guard:
        return self.profile.guard


def _card(environ: Mapping[str, str]) -> Card:
    """The model card: RAG__CARD names a shipped card (in RAG_CARDS_DIR, the image's /app/cards), RAG__CARD_FILE a card file.
    RAG__LLM_URL, when set, replaces the url of the card's local endpoints."""
    name, file = environ.get("RAG__CARD"), environ.get("RAG__CARD_FILE")
    if bool(name) == bool(file):
        raise SettingsError("RAG__CARD or RAG__CARD_FILE: exactly one names the model card (environment)")
    card = load_card(Path(file) if file else name or "", environ.get("RAG_CARDS_DIR") or DEFAULT_CARDS_DIR)
    return with_local_url(card, url) if (url := environ.get("RAG__LLM_URL")) else card


def load_settings(
    environ: Mapping[str, str] = os.environ, secrets_dir: str | Path = "/run/secrets"
) -> AssistantSettings:
    """The profile named by RAG_PROFILE, the overlay from the environment (and RAG_OVERLAY's file when one is set) and the secret
    files; SettingsError names a key and its source, never a value."""
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
    texts = yaml.safe_load((profile_file.parent / "messages.de.yaml").read_text(encoding="utf-8"))
    prompts = load_profile_prompts(profile_file.parent / "prompts")
    return AssistantSettings(profile, card, overlay, Messages.model_validate(texts), prompts, profile_file.parent)
