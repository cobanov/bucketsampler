"""Distribution analysis utilities.

Lightweight summaries that callers (CLI analyzer, notebook users, the auto-
bucket generator in M4) can use to inspect how a dataset interacts with a
:class:`BucketSet`. Designed to operate on numpy arrays of dims for speed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bucketsampler.core.assignment import assign_many_indices
from bucketsampler.core.bucket import Bucket, BucketSet


@dataclass(frozen=True, slots=True)
class AspectRatioSummary:
    """Aggregate statistics over a set of image aspect ratios.

    Attributes:
        count: Number of images.
        mean_log_ar: Arithmetic mean of ``log(w/h)``.
        median_log_ar: Median of ``log(w/h)``.
        std_log_ar: Population standard deviation of ``log(w/h)``.
        min_ar: Smallest ``w/h``.
        max_ar: Largest ``w/h``.
    """

    count: int
    mean_log_ar: float
    median_log_ar: float
    std_log_ar: float
    min_ar: float
    max_ar: float


@dataclass(frozen=True, slots=True)
class CropLossSummary:
    """Summary of crop-loss values for assigned dims.

    Attributes:
        count: Number of images.
        mean: Average pixel fraction discarded.
        median: Median pixel fraction discarded.
        p95: 95th-percentile pixel fraction discarded.
        max: Worst case pixel fraction discarded.
    """

    count: int
    mean: float
    median: float
    p95: float
    max: float


def _to_dims_array(dims: Sequence[tuple[int, int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(dims, dtype=np.int64)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"dims must have shape (N, 2), got {arr.shape}")
    if (arr <= 0).any():
        raise ValueError("all dims must be positive")
    return arr


def aspect_ratio_summary(dims: Sequence[tuple[int, int]] | np.ndarray) -> AspectRatioSummary:
    """Compute basic AR statistics for a set of images.

    Args:
        dims: Either an iterable of ``(width, height)`` tuples or an
            ``(N, 2)`` array.

    Returns:
        An :class:`AspectRatioSummary`. If ``dims`` is empty, count is zero
        and all numeric fields are NaN.
    """
    arr = _to_dims_array(dims)
    if arr.size == 0:
        nan = float("nan")
        return AspectRatioSummary(
            count=0,
            mean_log_ar=nan,
            median_log_ar=nan,
            std_log_ar=nan,
            min_ar=nan,
            max_ar=nan,
        )
    ratios = arr[:, 0].astype(np.float64) / arr[:, 1].astype(np.float64)
    log_ratios = np.log(ratios)
    return AspectRatioSummary(
        count=int(arr.shape[0]),
        mean_log_ar=float(log_ratios.mean()),
        median_log_ar=float(np.median(log_ratios)),
        std_log_ar=float(log_ratios.std()),
        min_ar=float(ratios.min()),
        max_ar=float(ratios.max()),
    )


def bucket_distribution(
    dims: Sequence[tuple[int, int]] | np.ndarray,
    bucket_set: BucketSet,
) -> dict[Bucket, int]:
    """Count how many images are assigned to each bucket.

    Buckets that receive zero images are still present in the returned dict
    (with value ``0``), so callers can see the full landscape.
    """
    counts: dict[Bucket, int] = {b: 0 for b in bucket_set}
    arr = _to_dims_array(dims)
    if arr.size == 0:
        return counts
    indices = assign_many_indices(arr, bucket_set)
    for idx in indices:
        counts[bucket_set[int(idx)]] += 1
    return counts


def underutilized_buckets(
    dims: Sequence[tuple[int, int]] | np.ndarray,
    bucket_set: BucketSet,
    *,
    min_count: int,
) -> list[Bucket]:
    """Return buckets that received fewer than ``min_count`` assignments.

    Args:
        dims: Image dimensions.
        bucket_set: Candidate buckets.
        min_count: Threshold below which a bucket is considered underutilized.

    Returns:
        Buckets in the same order as in ``bucket_set``.

    Raises:
        ValueError: If ``min_count`` is negative.
    """
    if min_count < 0:
        raise ValueError(f"min_count must be non-negative, got {min_count}")
    counts = bucket_distribution(dims, bucket_set)
    return [b for b in bucket_set if counts[b] < min_count]


def crop_loss_summary(
    dims: Sequence[tuple[int, int]] | np.ndarray,
    bucket_set: BucketSet,
) -> CropLossSummary:
    """Summarize crop loss across a dataset under a given bucket set.

    Each image is assigned to its nearest bucket, then its crop loss is
    computed. The result aggregates those per-image losses.

    Args:
        dims: Image dimensions.
        bucket_set: Candidate buckets.

    Returns:
        A :class:`CropLossSummary`. If ``dims`` is empty, count is zero and
        all numeric fields are NaN.
    """
    arr = _to_dims_array(dims)
    if arr.size == 0:
        nan = float("nan")
        return CropLossSummary(count=0, mean=nan, median=nan, p95=nan, max=nan)
    indices = assign_many_indices(arr, bucket_set)
    bucket_ars = np.fromiter(
        (b.aspect_ratio for b in bucket_set),
        dtype=np.float64,
        count=len(bucket_set),
    )
    src_ars = arr[:, 0].astype(np.float64) / arr[:, 1].astype(np.float64)
    assigned_ars = bucket_ars[indices]
    kept = np.where(
        src_ars > assigned_ars,
        assigned_ars / src_ars,
        src_ars / assigned_ars,
    )
    losses = 1.0 - kept
    return CropLossSummary(
        count=int(arr.shape[0]),
        mean=float(losses.mean()),
        median=float(np.median(losses)),
        p95=float(np.percentile(losses, 95)),
        max=float(losses.max()),
    )
