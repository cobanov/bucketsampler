"""Custom exceptions for bucketsampler.

Every exception that escapes a public API should inherit from
:class:`BucketSamplerError`, so callers can catch the whole family with one
handler. Each exception carries structured fields (paths, dims, suggestions)
to make errors actionable rather than mysterious.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class BucketSamplerError(Exception):
    """Base class for all bucketsampler errors."""


class InvalidBucketError(BucketSamplerError, ValueError):
    """A :class:`Bucket` was constructed with invalid dimensions.

    Args:
        width: Offending width value.
        height: Offending height value.
        reason: Short description of what is wrong.
    """

    def __init__(self, *, width: int, height: int, reason: str) -> None:
        self.width = width
        self.height = height
        self.reason = reason
        super().__init__(f"invalid bucket ({width}x{height}): {reason}")


class EmptyBucketSetError(BucketSamplerError, ValueError):
    """A :class:`BucketSet` was constructed without any buckets."""

    def __init__(self) -> None:
        super().__init__("BucketSet must contain at least one bucket")


class DuplicateBucketError(BucketSamplerError, ValueError):
    """A :class:`BucketSet` was constructed with duplicate (width, height) entries.

    Args:
        duplicates: The duplicated dimension tuples.
    """

    def __init__(self, *, duplicates: Iterable[tuple[int, int]]) -> None:
        dups = sorted(set(duplicates))
        self.duplicates = dups
        joined = ", ".join(f"{w}x{h}" for w, h in dups)
        super().__init__(f"BucketSet contains duplicate buckets: {joined}")


class ImageTooSmallError(BucketSamplerError, ValueError):
    """An image is smaller than its assigned bucket on at least one axis.

    Args:
        path: Image path, if known.
        actual: Native (width, height) of the image.
        required: Target (width, height) of the assigned bucket.
        suggestion: Human-facing remediation hint.
    """

    def __init__(
        self,
        *,
        path: Path | str | None,
        actual: tuple[int, int],
        required: tuple[int, int],
        suggestion: str = "",
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.actual = actual
        self.required = required
        self.suggestion = suggestion
        where = f" at {self.path}" if self.path is not None else ""
        tip = f" {suggestion}" if suggestion else ""
        super().__init__(
            f"image{where} is {actual[0]}x{actual[1]} but bucket requires "
            f"{required[0]}x{required[1]}.{tip}"
        )


class PresetNotFoundError(BucketSamplerError, KeyError):
    """Requested preset name was not bundled with the package.

    Args:
        name: The preset name that was asked for.
        available: Names that ARE bundled, for the error message.
    """

    def __init__(self, *, name: str, available: Iterable[str]) -> None:
        self.name = name
        self.available = sorted(available)
        opts = ", ".join(self.available) if self.available else "(none)"
        super().__init__(f"preset {name!r} not found. available: {opts}")

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


class InvalidPresetError(BucketSamplerError, ValueError):
    """A preset file (TOML or JSON) failed schema validation.

    Args:
        source: File path or URL of the bad preset.
        reason: Validation failure description.
    """

    def __init__(self, *, source: Path | str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"invalid preset {source!s}: {reason}")
