"""Bundled bucket-set presets and loader utilities.

Built-in TOML presets live as data files alongside this module. The
:func:`load_preset` function reads them via :mod:`importlib.resources` so they
work both from a source checkout and from an installed wheel. External presets
can be loaded from arbitrary paths with :func:`load_from_toml` or
:func:`load_from_json`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from bucketsampler.core.bucket import BucketSet
from bucketsampler.exceptions import InvalidPresetError, PresetNotFoundError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]


class _BucketDims(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: PositiveInt
    height: PositiveInt


class _PresetSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    description: str = ""
    vae_factor: PositiveInt = 1
    buckets: list[_BucketDims] = Field(min_length=1)


def list_presets() -> list[str]:
    """List the names of presets bundled with the package."""
    pkg = resources.files(__name__)
    names: list[str] = []
    for entry in pkg.iterdir():
        name = entry.name
        if name.endswith(".toml") and not name.startswith("_"):
            names.append(name[: -len(".toml")])
    return sorted(names)


def load_preset(name: str) -> BucketSet:
    """Load a bundled preset by name.

    Args:
        name: Preset stem (e.g. ``"sdxl"``, ``"sd15"``, ``"novelai"``).

    Returns:
        A frozen :class:`BucketSet`.

    Raises:
        PresetNotFoundError: If no preset with that name is bundled.
        InvalidPresetError: If the bundled file fails schema validation
            (should not happen in a healthy install, but guards corruption).
    """
    pkg = resources.files(__name__)
    target = pkg / f"{name}.toml"
    if not target.is_file():
        raise PresetNotFoundError(name=name, available=list_presets())
    with target.open("rb") as fh:
        data = tomllib.load(fh)
    return _bucket_set_from_dict(data, source=f"preset:{name}")


def load_from_toml(path: str | Path) -> BucketSet:
    """Load a user-supplied bucket set from a TOML file.

    Args:
        path: Filesystem path to a TOML file matching the preset schema.

    Raises:
        FileNotFoundError: If the path does not exist.
        InvalidPresetError: If the file fails schema validation.
    """
    p = Path(path)
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    return _bucket_set_from_dict(data, source=p)


def load_from_json(path: str | Path) -> BucketSet:
    """Load a user-supplied bucket set from a JSON file.

    Args:
        path: Filesystem path to a JSON file matching the preset schema.

    Raises:
        FileNotFoundError: If the path does not exist.
        InvalidPresetError: If the file fails schema validation.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return _bucket_set_from_dict(data, source=p)


def _bucket_set_from_dict(data: dict[str, Any], *, source: Path | str) -> BucketSet:
    try:
        validated = _PresetSchema.model_validate(data)
    except ValidationError as exc:
        raise InvalidPresetError(source=source, reason=str(exc)) from exc
    dims: Iterable[tuple[int, int]] = ((b.width, b.height) for b in validated.buckets)
    try:
        return BucketSet.from_dims(
            dims,
            name=validated.name,
            description=validated.description,
            vae_factor=validated.vae_factor,
        )
    except Exception as exc:
        raise InvalidPresetError(source=source, reason=str(exc)) from exc


__all__ = [
    "list_presets",
    "load_from_json",
    "load_from_toml",
    "load_preset",
]
