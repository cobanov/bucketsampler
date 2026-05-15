"""Image-to-bucket assignment.

The matching policy is **minimum log-aspect-ratio distance**: pixel count is
treated as a property of the bucket set's design, not of the assignment step.
Each image picks the bucket whose ``log(w / h)`` is closest to its own.

For datasets of any meaningful size, prefer :func:`assign_many_indices` (the
vectorized form), which scales O(N x K) with cheap numpy ops, over per-image
Python loops.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from bucketsampler.core.bucket import Bucket, BucketSet


def log_ar_distance(width: int, height: int, bucket: Bucket) -> float:
    """Absolute log-aspect-ratio distance between an image and a bucket.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        bucket: Candidate bucket.

    Returns:
        ``abs(log(w/h) - log(bucket.w/bucket.h))``. Zero means identical AR.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got ({width}, {height})")
    return abs(math.log(width / height) - bucket.log_aspect_ratio)


def best_bucket(width: int, height: int, bucket_set: BucketSet) -> Bucket:
    """Pick the closest bucket for a single image.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        bucket_set: Candidate buckets.

    Returns:
        The bucket with the smallest log-aspect-ratio distance to the image.
        Ties resolve to the bucket that appears earlier in ``bucket_set``.

    Raises:
        ValueError: If ``width`` or ``height`` is non-positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got ({width}, {height})")
    img_log_ar = math.log(width / height)
    best_idx = 0
    best_distance = abs(bucket_set[0].log_aspect_ratio - img_log_ar)
    for i in range(1, len(bucket_set)):
        d = abs(bucket_set[i].log_aspect_ratio - img_log_ar)
        if d < best_distance:
            best_distance = d
            best_idx = i
    return bucket_set[best_idx]


def assign_many_indices(
    dims: Sequence[tuple[int, int]] | np.ndarray,
    bucket_set: BucketSet,
) -> np.ndarray:
    """Vectorized assignment. Returns bucket indices into ``bucket_set``.

    Args:
        dims: Either a sequence of ``(width, height)`` tuples, or an
            ``(N, 2)`` int array.
        bucket_set: Candidate buckets.

    Returns:
        ``(N,)`` int64 array. Entry ``i`` is the index in ``bucket_set`` of
        the closest bucket to ``dims[i]``.

    Raises:
        ValueError: If any dim is non-positive or ``dims`` has the wrong shape.
    """
    arr = np.asarray(dims, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"dims must have shape (N, 2), got {arr.shape}")
    if (arr <= 0).any():
        bad = np.argwhere(arr <= 0)
        first = tuple(int(x) for x in arr[bad[0, 0]])
        raise ValueError(f"all dims must be positive, found {first}")

    img_log_ars = np.log(arr[:, 0].astype(np.float64) / arr[:, 1].astype(np.float64))
    bucket_log_ars = np.fromiter(
        (b.log_aspect_ratio for b in bucket_set),
        dtype=np.float64,
        count=len(bucket_set),
    )
    diffs = np.abs(img_log_ars[:, None] - bucket_log_ars[None, :])
    out: np.ndarray = np.argmin(diffs, axis=1).astype(np.int64)
    return out


def assign_many(
    dims: Sequence[tuple[int, int]] | Iterable[tuple[int, int]],
    bucket_set: BucketSet,
) -> list[Bucket]:
    """Vectorized assignment. Returns :class:`Bucket` instances.

    Convenience wrapper over :func:`assign_many_indices` that materializes the
    bucket objects. Prefer the index form when feeding into a sampler or
    storing assignments compactly.
    """
    seq = list(dims) if not isinstance(dims, Sequence) else dims
    indices = assign_many_indices(seq, bucket_set)
    return [bucket_set[int(i)] for i in indices]


def resize_to_bucket_dims(
    width: int,
    height: int,
    bucket: Bucket,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute the resize and crop offsets to fit an image into a bucket.

    The strategy is **fit longer dim, then center-crop**: scale so the image
    fully covers the bucket on both axes (without distortion), then crop the
    overhang symmetrically.

    Args:
        width: Native image width.
        height: Native image height.
        bucket: Target bucket.

    Returns:
        A tuple ``(resized_dims, crop_offset)`` where ``resized_dims`` is the
        ``(w, h)`` after scaling and ``crop_offset`` is the top-left ``(x, y)``
        of the center crop within those resized dims. Applying
        ``resize(img, resized_dims)`` followed by
        ``crop(img, crop_offset, bucket.size)`` produces a bucket-shaped image.

    Raises:
        ValueError: If ``width`` or ``height`` is non-positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got ({width}, {height})")
    scale = max(bucket.width / width, bucket.height / height)
    new_w = max(bucket.width, round(width * scale))
    new_h = max(bucket.height, round(height * scale))
    crop_x = (new_w - bucket.width) // 2
    crop_y = (new_h - bucket.height) // 2
    return (new_w, new_h), (crop_x, crop_y)


def crop_loss(width: int, height: int, bucket: Bucket) -> float:
    """Fraction of source pixels discarded by :func:`resize_to_bucket_dims`.

    Args:
        width: Native image width.
        height: Native image height.
        bucket: Target bucket.

    Returns:
        Value in ``[0, 1)``. ``0.0`` means the source AR equals the bucket AR
        (no crop needed). Higher values mean more pixels lost.

    Raises:
        ValueError: If ``width`` or ``height`` is non-positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got ({width}, {height})")
    src_ar = width / height
    bkt_ar = bucket.aspect_ratio
    kept = bkt_ar / src_ar if src_ar > bkt_ar else src_ar / bkt_ar
    return 1.0 - kept
