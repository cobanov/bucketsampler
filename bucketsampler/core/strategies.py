"""Assignment strategies.

A strategy decides which bucket an image gets, given its dimensions. The
common case is :class:`FixedBuckets`, which uses a predefined
:class:`BucketSet` and the log-AR nearest-bucket rule. Data-derived
strategies (k-means in log-AR space) live in
:mod:`bucketsampler.core.auto_bucket`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from bucketsampler.core.assignment import assign_many_indices, best_bucket
from bucketsampler.core.bucket import Bucket, BucketSet


@runtime_checkable
class Strategy(Protocol):
    """Duck-typed bucket assignment interface.

    Any object that exposes :attr:`bucket_set`, :meth:`assign`, and
    :meth:`assign_many_indices` (returning a numpy index array) is a valid
    Strategy. Used by samplers and dataset wrappers in downstream packages.
    """

    @property
    def bucket_set(self) -> BucketSet:
        """The full set of buckets the strategy can pick from."""
        ...

    def assign(self, width: int, height: int) -> Bucket:
        """Pick a bucket for a single image."""
        ...

    def assign_many_indices(self, dims: Sequence[tuple[int, int]] | np.ndarray) -> np.ndarray:
        """Vectorized bucket-index assignment for many images."""
        ...


class FixedBuckets:
    """Assign images to a fixed, predefined :class:`BucketSet`.

    Uses the minimum log-aspect-ratio rule from
    :mod:`bucketsampler.core.assignment`. The bucket set itself is frozen, so
    repeated assignments are deterministic and safe to share between threads
    or processes.

    Args:
        bucket_set: The buckets to assign into.

    Example:
        >>> from bucketsampler import BucketSet, FixedBuckets
        >>> bs = BucketSet.from_dims([(512, 512), (768, 512), (512, 768)])
        >>> strat = FixedBuckets(bs)
        >>> strat.assign(1024, 768)
        Bucket(width=768, height=512)
    """

    def __init__(self, bucket_set: BucketSet) -> None:
        self._bucket_set = bucket_set

    @property
    def bucket_set(self) -> BucketSet:
        return self._bucket_set

    def assign(self, width: int, height: int) -> Bucket:
        """Return the closest bucket for the image dims."""
        return best_bucket(width, height, self._bucket_set)

    def assign_many_indices(self, dims: Sequence[tuple[int, int]] | np.ndarray) -> np.ndarray:
        """Vectorized assignment. Returns an int64 array of bucket indices."""
        return assign_many_indices(dims, self._bucket_set)

    def __repr__(self) -> str:
        return f"FixedBuckets(n_buckets={len(self._bucket_set)})"
