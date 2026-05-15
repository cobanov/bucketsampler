"""Value types for buckets and bucket sets.

A :class:`Bucket` is a single (width, height) target. A :class:`BucketSet` is
an immutable, deduplicated collection of buckets. Both are frozen dataclasses
so they can be safely hashed, cached, and shared across processes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from bucketsampler.exceptions import (
    DuplicateBucketError,
    EmptyBucketSetError,
    InvalidBucketError,
)


@dataclass(frozen=True, slots=True)
class Bucket:
    """A single (width, height) training target.

    Buckets are immutable value types. Two buckets with the same width and
    height compare equal and hash to the same value, regardless of order in
    the parent :class:`BucketSet`.

    Args:
        width: Pixel width. Must be a positive int.
        height: Pixel height. Must be a positive int.

    Example:
        >>> b = Bucket(1024, 768)
        >>> b.aspect_ratio
        1.3333333333333333
        >>> b.pixel_count
        786432
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or isinstance(self.width, bool):
            raise InvalidBucketError(
                width=self.width, height=self.height, reason="width must be an int"
            )
        if not isinstance(self.height, int) or isinstance(self.height, bool):
            raise InvalidBucketError(
                width=self.width, height=self.height, reason="height must be an int"
            )
        if self.width <= 0 or self.height <= 0:
            raise InvalidBucketError(
                width=self.width,
                height=self.height,
                reason="width and height must be positive",
            )

    @property
    def aspect_ratio(self) -> float:
        """Width-over-height ratio. ``1.0`` is square, ``>1.0`` is landscape."""
        return self.width / self.height

    @property
    def log_aspect_ratio(self) -> float:
        """Natural log of :attr:`aspect_ratio`. Symmetric around ``0.0``."""
        return math.log(self.width / self.height)

    @property
    def pixel_count(self) -> int:
        """Total pixels (``width * height``)."""
        return self.width * self.height

    def is_multiple_of(self, factor: int) -> bool:
        """Whether both dims are exact multiples of ``factor``.

        Args:
            factor: VAE downsample factor (e.g. 8 for SD VAE, 16 for some new VAEs).
        """
        return self.width % factor == 0 and self.height % factor == 0

    def as_tuple(self) -> tuple[int, int]:
        """Return ``(width, height)`` as a plain tuple."""
        return (self.width, self.height)

    def __repr__(self) -> str:
        return f"Bucket(width={self.width}, height={self.height})"

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True, slots=True)
class BucketSet:
    """An immutable collection of unique buckets.

    Construct directly from a tuple of :class:`Bucket` instances, or use
    :meth:`from_dims` for raw ``(w, h)`` tuples. Order is preserved (it can
    affect which bucket wins ties during assignment), but duplicates are
    rejected to keep distance lookups well-defined.

    Args:
        buckets: Tuple of :class:`Bucket` instances. At least one is required.
        name: Optional human label (used by presets).

    Example:
        >>> bs = BucketSet.from_dims([(512, 512), (768, 512), (512, 768)])
        >>> len(bs)
        3
        >>> bs[0]
        Bucket(width=512, height=512)
    """

    buckets: tuple[Bucket, ...]
    name: str = ""
    description: str = ""
    vae_factor: int = field(default=1)

    def __post_init__(self) -> None:
        if not isinstance(self.buckets, tuple):
            object.__setattr__(self, "buckets", tuple(self.buckets))
        if len(self.buckets) == 0:
            raise EmptyBucketSetError()
        if self.vae_factor <= 0:
            raise ValueError("vae_factor must be a positive int")
        seen: dict[tuple[int, int], int] = {}
        dups: list[tuple[int, int]] = []
        for b in self.buckets:
            key = b.as_tuple()
            if key in seen:
                dups.append(key)
            else:
                seen[key] = 1
        if dups:
            raise DuplicateBucketError(duplicates=dups)

    @classmethod
    def from_dims(
        cls,
        dims: Iterable[tuple[int, int]],
        *,
        name: str = "",
        description: str = "",
        vae_factor: int = 1,
    ) -> BucketSet:
        """Build a :class:`BucketSet` from raw ``(width, height)`` tuples.

        Args:
            dims: Iterable of ``(width, height)`` pairs.
            name: Optional human label.
            description: Optional longer description.
            vae_factor: Expected VAE downsample factor for the buckets.

        Returns:
            A new :class:`BucketSet`.

        Raises:
            EmptyBucketSetError: If ``dims`` is empty.
            DuplicateBucketError: If any dims are repeated.
            InvalidBucketError: If any dim is non-positive.
        """
        buckets = tuple(Bucket(w, h) for w, h in dims)
        return cls(
            buckets=buckets,
            name=name,
            description=description,
            vae_factor=vae_factor,
        )

    def __iter__(self) -> Iterator[Bucket]:
        return iter(self.buckets)

    def __len__(self) -> int:
        return len(self.buckets)

    def __getitem__(self, index: int) -> Bucket:
        return self.buckets[index]

    def __contains__(self, item: object) -> bool:
        return item in self.buckets

    def index_of(self, bucket: Bucket) -> int:
        """Return the position of ``bucket`` in the set.

        Raises:
            ValueError: If the bucket is not present.
        """
        return self.buckets.index(bucket)

    def sorted_by_aspect_ratio(self) -> tuple[Bucket, ...]:
        """Return buckets sorted by ascending aspect ratio (tall to wide)."""
        return tuple(sorted(self.buckets, key=lambda b: b.aspect_ratio))

    def pixel_budget(self) -> int:
        """Median pixel count of the contained buckets.

        Useful for picking a comparable bucket set during auto-bucket generation.
        """
        counts = sorted(b.pixel_count for b in self.buckets)
        mid = len(counts) // 2
        if len(counts) % 2 == 1:
            return counts[mid]
        return (counts[mid - 1] + counts[mid]) // 2

    def all_multiples_of(self, factor: int) -> bool:
        """Whether every bucket has both dims divisible by ``factor``."""
        return all(b.is_multiple_of(factor) for b in self.buckets)
