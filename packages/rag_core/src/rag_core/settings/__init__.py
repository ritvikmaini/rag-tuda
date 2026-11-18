"""Layered settings for the workspace packages.

Technical sections are OverlaySection models (extra keys forbidden, frozen) and a settings model holds sections. load()
reads, each layer overriding the one before:
  1. the overlay file, when one is named (YAML: {section: {key: value}});
  2. environment variables RAG__<SECTION>__<KEY> (RAG__CARD and RAG__CARD_FILE name the model card, see card.py);
  3. secret files: a field declared with secret("<name>") reads <secrets_dir>/<name> (trailing line breaks stripped);
     secrets_dir is the argument, else RAG_SECRETS_DIR, else /run/secrets.
An unknown section or key, in the overlay or the environment, raises SettingsError naming the key and its source; so does a
value that does not validate, and a key the environment sets twice (names that differ only in case). Secret fields are
SecretStr: their values never appear in a message or a repr; a blank secret (empty or whitespace only) counts as unset. The
profile (P) layer is separate: load_profile() validates the profile file baked into an image, and load(..., profile_sections=...)
refuses any overlay, environment or secret entry for a profile section."""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.fields import FieldInfo

PREFIX = "RAG__"
CARD_VARIABLES = frozenset({"RAG__CARD", "RAG__CARD_FILE"})  # select the model card; they set no section key
_SECRET_FILE = "secret_file"

Layer = dict[tuple[str, str], tuple[object, str]]  # (section, key) -> (value, the source's name)


class SettingsError(ValueError):
    """A source sets a key no section declares, or a value that does not validate. The message names the key and its
    source, never a value."""


class OverlaySection(BaseModel):
    """A technical (O) section: the overlay file, the environment and secret files may set its keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


SettingsModel = TypeVar("SettingsModel", bound=OverlaySection)


class ProfileModel(BaseModel):
    """A profile (P) model: read only from the profile file baked into the image; no overlay,
    environment variable or secret file sets its keys, so an image tag fixes every value that decides the answers and the
    index."""

    model_config = ConfigDict(extra="forbid", frozen=True)


ProfileType = TypeVar("ProfileType", bound=ProfileModel)


def load_profile(model: type[ProfileType], path: str | Path) -> ProfileType:
    """The profile file validated as model; an unknown, missing or invalid key raises SettingsError naming it and the
    file, never a value."""
    source = f"profile {path}"
    try:
        data: object = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SettingsError(f"{source}: cannot be read ({exc.strerror})") from None
    try:
        return model.model_validate(data if data is not None else {})
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']} ({source})"
            for error in exc.errors(include_url=False, include_input=False)
        ]
        raise SettingsError("; ".join(problems)) from None


def secret(file: str) -> Any:
    """Declare a SecretStr field read from <secrets_dir>/<file> (or RAG__<SECTION>__<KEY>); None when neither is set."""
    return Field(default=None, json_schema_extra={_SECRET_FILE: file})


def env_key(name: str) -> tuple[str, str] | None:
    """RAG__<SECTION>__<KEY> -> (section, key) in lower case; None for a variable without the prefix. A variable with the
    prefix but without a section or a key raises SettingsError, except the card variables (CARD_VARIABLES): None."""
    if not name.startswith(PREFIX) or name in CARD_VARIABLES:
        return None
    section, separator, key = name[len(PREFIX) :].partition("__")
    if not section or not separator or not key:
        raise SettingsError(f"{name}: expected RAG__<SECTION>__<KEY> (environment)")
    return section.lower(), key.lower()


def _sections(model: type[OverlaySection]) -> dict[str, type[OverlaySection]]:
    sections: dict[str, type[OverlaySection]] = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if not (isinstance(annotation, type) and issubclass(annotation, OverlaySection)):
            raise SettingsError(f"{model.__name__}.{name}: a settings model holds sections only")
        sections[name] = annotation
    return sections


def _overlay_layer(path: str | Path) -> Layer:
    source = f"overlay {path}"
    try:
        data: object = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SettingsError(f"{source}: cannot be read ({exc.strerror})") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SettingsError(f"{source}: expected a mapping of sections")
    layer: Layer = {}
    for section, keys in cast("dict[object, object]", data).items():
        if not isinstance(keys, dict):
            raise SettingsError(f"{section}: expected a mapping of keys ({source})")
        for key, value in cast("dict[object, object]", keys).items():
            layer[(str(section), str(key))] = (value, source)
    return layer


def _environment_layer(environ: Mapping[str, str]) -> Layer:
    layer: Layer = {}
    for name, value in environ.items():
        parsed = env_key(name)
        if parsed is not None:
            if parsed in layer:  # env_key folds case: RAG__EVALS__X and RAG__evals__x are one key
                raise SettingsError(f"{parsed[0]}.{parsed[1]}: set twice ({layer[parsed][1]} and {name} (environment))")
            layer[parsed] = (value, f"{name} (environment)")
    return layer


def _secret_file(field: FieldInfo) -> str | None:
    extra = field.json_schema_extra
    file = cast("dict[str, object]", extra).get(_SECRET_FILE) if isinstance(extra, dict) else None
    return file if isinstance(file, str) else None


def _secret_layer(sections: dict[str, type[OverlaySection]], secrets_dir: str | Path) -> Layer:
    layer: Layer = {}
    for section, model in sections.items():
        for key, field in model.model_fields.items():
            file = _secret_file(field)
            if file is not None:
                path = Path(secrets_dir) / file
                if path.is_file():
                    layer[(section, key)] = (path.read_text(encoding="utf-8").rstrip("\r\n"), f"secret file {path}")
    return layer


def load(
    model: type[SettingsModel],
    *,
    overlay: str | Path | None = None,
    environ: Mapping[str, str] = os.environ,
    secrets_dir: str | Path | None = None,
    profile_sections: Collection[str] = (),
) -> SettingsModel:
    """The settings model filled from its layers (module docstring)."""
    sections = _sections(model)
    secrets_dir = secrets_dir or environ.get("RAG_SECRETS_DIR") or "/run/secrets"
    merged: Layer = {}
    layers = (
        _overlay_layer(overlay) if overlay else {},
        _environment_layer(environ),
        _secret_layer(sections, secrets_dir),
    )
    for layer in layers:
        for (section, key), entry in layer.items():
            if section in profile_sections:
                raise SettingsError(f"{section}.{key}: a profile (P) key, set only by the image's profile ({entry[1]})")
            if section not in sections:
                raise SettingsError(f"{section}: unknown section ({entry[1]})")
            model_of = sections[section]
            field = model_of.model_fields.get(key)
            if field is None and model_of.model_config.get("extra") != "allow":
                raise SettingsError(f"{section}.{key}: unknown key ({entry[1]})")
            value = entry[0]
            if isinstance(value, str) and not value.strip() and field is not None and _secret_file(field):
                continue  # a blank secret is no secret, and it hides no earlier layer
            merged[(section, key)] = entry
    data: dict[str, dict[str, object]] = {}
    for (section, key), (value, _source) in merged.items():
        data.setdefault(section, {})[key] = value
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems: list[str] = []
        for error in exc.errors(include_url=False, include_input=False):
            loc = [str(part) for part in error["loc"]]
            entry = merged.get((loc[0], loc[1])) if len(loc) >= 2 else None
            problems.append(f"{'.'.join(loc)}: {error['msg']} ({entry[1] if entry else 'default'})")
        raise SettingsError("; ".join(problems)) from None
