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
from rag_core.settings.profile import Profile, Source
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
    """How the crawler introduces itself and, for the harness, where it is sent. The contact is the user's own
    (RAG__CRAWL__CONTACT, never a file of the repo); without one the user agent says "student research prototype"."""

    contact: str | None = None
    routes: dict[str, str] = {}  # host -> origin that answers for it (indexer.crawl.routes); none: the real network


class PathsSection(OverlaySection):
    """The folders the indexer reads and writes; the defaults are the compose container's."""

    crawl_dir: str = "/app/data/crawl"
    manifest_dir: str = "/app/data/vector_database_contents/documents"
    state_dir: str = "/app/data/vector_database_contents"  # the registry, the list state and the build stats
    extracted_dir: str = "/app/data/extracted"  # the PDF extraction cache
    pdf_dir: str = "/app/data/pdfs"
    skipped_pdfs_log: str = "/app/data/skipped_pdfs_page_limit.json"
    user_sources: str = "/app/data/sources.json"  # the folders `rag add` registered; a missing file lists none


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
    profile_name: str = ""  # the profile's folder name (`example`, `tu-informatik`); RAG_PROFILE_NAME in the image

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


def user_folders(path: str) -> dict[str, str]:
    """name -> folder of the sources file `rag add` writes ({"sources": [{"name", "target"}]}); none when the file is missing or empty."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise SettingsError(f"{path}: cannot be read ({exc.strerror})") from None
    if not text.strip():
        return {}
    try:
        entries = json.loads(text)["sources"]
        if not isinstance(entries, list):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        raise SettingsError(f'{path}: not a sources file ({{"sources": [{{"name": ..., "target": ...}}]}})') from None
    folders: dict[str, str] = {}
    for i, entry in enumerate(entries):
        for key in ("name", "target"):
            if not isinstance(entry, dict) or not isinstance(entry.get(key), str) or not entry[key]:
                raise SettingsError(f'{path}: sources[{i}] needs a "{key}" (a string), got {entry!r}')
        folders[entry["name"]] = entry["target"]
    return folders


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
    folders = user_folders(overlay.paths.user_sources)
    if folders:  # beside the profile's sources: each a folder source without a base_url, so its urls read folder:<path>
        if clash := sorted(set(folders) & set(profile.ingest.sources)):
            raise SettingsError(f"{overlay.paths.user_sources}: {clash[0]!r} is a source of the profile already")
        added = {
            name: Source.model_validate({"connector": "folder", "audience": "public", "label": name})
            for name in folders
        }
        ingest = profile.ingest.model_copy(update={"sources": {**profile.ingest.sources, **added}})
        profile = profile.model_copy(update={"ingest": ingest})
        endpoints |= {name: Endpoint(dir=target) for name, target in folders.items()}
    profile_dir = Path(profile_path).parent
    return Settings(profile, overlay, endpoints, profile_dir, environ.get("RAG_PROFILE_NAME") or profile_dir.name)


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
