"""The indexer's settings: the profile baked into the image (RAG_PROFILE), the overlay mounted read-only (RAG_OVERLAY),
RAG__<SECTION>__<KEY> variables and secret files (in RAG_SECRETS_DIR, default /run/secrets)."""

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError
from rag_core.adapters.db import connect_direct
from rag_core.settings import OverlaySection, SettingsError, load, load_profile
from rag_core.settings.profile import Profile
from rag_core.settings.sections import DatabaseSection, ModelServersSection

from indexer.index.generations import Target

SECRET_KEYS = {"database": {"name", "user", "password"}}


class Endpoint(BaseModel):
    """connectors.<source>: where the connector reads and which hosts its data-derived downloads may use."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    url: str | None = None
    manifest_url: str | None = None
    dir: str | None = None
    allowed_hosts: list[str] | None = None


class ConnectorsSection(OverlaySection):
    """One Endpoint per source of the profile; the names are the profile's, so the section is open and checked at load."""

    model_config = ConfigDict(extra="allow", frozen=True)


class CrawlSection(OverlaySection):
    """How the crawler introduces itself. The contact is the user's own (RAG__CRAWL__CONTACT, never a file of the repo);
    without one the user agent says "student research prototype"."""

    contact: str | None = None


class PathsSection(OverlaySection):
    """The folders the indexer reads and writes; the defaults are the compose container's."""

    crawl_dir: str = "/app/data/crawl"
    manifest_dir: str = "/app/data/vector_database_contents/documents"
    state_dir: str = "/app/data/vector_database_contents"  # the registry, the list state and the build stats
    extracted_dir: str = "/app/data/extracted"  # the PDF extraction cache
    pdf_dir: str = "/app/data/pdfs"
    skipped_pdfs_log: str = "/app/data/skipped_pdfs_page_limit.json"


class Overlay(OverlaySection):
    model_config = ConfigDict(
        extra="forbid", frozen=True, protected_namespaces=()
    )  # the section is called model_servers

    database: DatabaseSection = DatabaseSection()
    model_servers: ModelServersSection = ModelServersSection()
    connectors: ConnectorsSection = ConnectorsSection()
    paths: PathsSection = PathsSection()
    crawl: CrawlSection = CrawlSection()


@dataclass(frozen=True)
class Settings:
    profile: Profile
    overlay: Overlay
    endpoints: Mapping[str, Endpoint]  # every source of the profile, in its order
    profile_dir: Path  # the folder of the profile file, where sources.yaml and roles.yaml are

    @property
    def collection(self) -> str:
        return self.profile.collections[0].name

    def config_hash(self) -> str:
        """sha-256 of the effective settings as canonical JSON, secrets left out (index_generations.config_hash)."""
        overlay = self.overlay.model_dump(
            mode="json", exclude={**SECRET_KEYS, "connectors": True, "paths": True, "crawl": True}
        )  # type: ignore[arg-type]
        data = {
            "profile": self.profile.model_dump(mode="json"),
            "overlay": overlay,
            "endpoints": {name: e.model_dump(mode="json") for name, e in self.endpoints.items()},
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_settings(environ: Mapping[str, str] = os.environ, secrets_dir: str | None = None) -> Settings:
    profile_path = environ.get("RAG_PROFILE")
    if not profile_path:
        raise SettingsError("RAG_PROFILE: not set (the image sets it to its baked profile)")
    profile = load_profile(Profile, profile_path)
    overlay = load(
        Overlay,
        overlay=environ.get("RAG_OVERLAY") or None,
        environ=environ,
        secrets_dir=secrets_dir,
        profile_sections=frozenset(Profile.model_fields),
    )
    entries = overlay.connectors.model_extra or {}
    unknown = sorted(set(entries) - set(profile.ingest.sources))
    if unknown:
        raise SettingsError(f"connectors.{unknown[0]}: no such source in the profile (overlay)")
    endpoints: dict[str, Endpoint] = {}
    for name in profile.ingest.sources:
        try:
            endpoints[name] = Endpoint.model_validate(entries.get(name) or {})
        except ValidationError as exc:
            raise SettingsError(f"connectors.{name}: {exc.errors()[0]['msg']} (overlay)") from None
    return Settings(profile, overlay, endpoints, Path(profile_path).parent)


@dataclass(frozen=True)
class Wiring:
    """What the app and the CLI both derive from the settings."""

    settings: Settings
    target: Target
    connect: Callable[..., Any]  # connect(application_name=...): a direct connection as the container's role


def wire(settings: Settings | None = None, connect: Callable[..., Any] | None = None) -> Wiring:
    settings = settings or load_settings()
    database = settings.overlay.database
    return Wiring(
        settings,
        Target.for_collection(settings.collection),
        connect or (lambda application_name: connect_direct(database, application_name)),
    )
