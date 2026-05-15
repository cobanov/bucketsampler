"""Auto-bucket generation via 1-D k-means on log-aspect-ratio.

Given a dataset's image dims, this module picks ``k`` ``(width, height)``
targets that minimise crop loss for that specific distribution. Generic
presets (SDXL, SD1.5) are great defaults, but a dataset of, say, mostly
ultrawide screenshots will waste capacity on portrait buckets it never
fills. ``generate_buckets`` reads the actual distribution and emits a
bucket set sized to it.

Algorithm:

  1. Compute ``log(w / h)`` for every image.
  2. Cluster into ``k`` 1-D centers via Lloyd's algorithm, initialised
     with evenly spaced quantiles for deterministic, near-optimal
     starts.
  3. Convert each center ``r`` into a ``(w, h)`` target with
     ``w * h ~= pixel_budget`` and ``w / h = exp(r)``; round each
     dimension to a multiple of ``vae_factor`` so the latents stay
     VAE-friendly.
  4. Deduplicate any centers that rounded to the same bucket.

The result is fully deterministic for a given ``seed`` and input dims.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bucketsampler.core.bucket import Bucket, BucketSet
from bucketsampler.core.stats import crop_loss_summary


@dataclass(frozen=True, slots=True)
class AutoBucketResult:
    """Output of :func:`generate_buckets`.

    Attributes:
        bucket_set: The generated buckets.
        cluster_centers: Final 1-D k-means centers, in log-AR space, in
            the same order as ``bucket_set``. Useful for inspecting how
            tight each cluster is.
        cluster_sizes: Number of dataset images that landed in each
            cluster pre-rounding (so the caller can spot tiny clusters
            that maybe should have been merged).
        iterations: Number of Lloyd's-algorithm iterations until
            convergence. Capped by ``max_iter``.
        crop_loss_mean: Mean crop loss when the input dims are assigned
            to the generated bucket set. Lower is better; comparing
            against a preset gives a quick "did we improve?" signal.
        requested_k: The ``num_buckets`` originally asked for. May
            exceed ``len(bucket_set)`` if two centers rounded together.
    """

    bucket_set: BucketSet
    cluster_centers: tuple[float, ...]
    cluster_sizes: tuple[int, ...]
    iterations: int
    crop_loss_mean: float
    requested_k: int


def generate_buckets(
    dims: Sequence[tuple[int, int]] | np.ndarray,
    *,
    num_buckets: int = 8,
    target: int = 1024,
    vae_factor: int = 64,
    seed: int = 0,
    max_iter: int = 100,
    name: str = "auto",
    description: str = "",
) -> AutoBucketResult:
    """Generate a bucket set tailored to a dataset.

    Args:
        dims: ``(N, 2)`` array or iterable of ``(width, height)`` tuples,
            usually obtained via the analyzer's scan.
        num_buckets: Desired bucket count. The final set may be smaller
            if two cluster centers round to the same ``(w, h)``.
        target: Approximate target side length in pixels (so the pixel
            budget is ``target ** 2``). For example, ``1024`` produces
            roughly 1024x1024-equivalent buckets at AR 1, 1448x724 at
            AR 2, and so on, before rounding.
        vae_factor: Round each dim to a multiple of this. ``64`` matches
            SD-family VAEs (8x downsample, with model strides forcing
            64 in practice); use ``8`` or ``16`` for VAEs with finer
            grids.
        seed: RNG seed. The init is quantile-based and deterministic,
            so the seed only matters in degenerate ties.
        max_iter: Lloyd's-algorithm iteration cap.
        name: Stored on the returned :class:`BucketSet`. Useful when
            writing the result out to TOML.
        description: Stored on the returned :class:`BucketSet`.

    Returns:
        An :class:`AutoBucketResult` with the bucket set, cluster
        details, and a crop-loss summary you can compare against a
        preset.

    Raises:
        ValueError: If ``num_buckets`` is non-positive, ``target`` is
            non-positive, ``vae_factor`` is non-positive, or ``dims``
            is empty.

    Example:
        >>> import numpy as np
        >>> dims = np.array([[1024, 1024], [2048, 1024], [1024, 2048]] * 100)
        >>> result = generate_buckets(dims, num_buckets=3, target=1024, seed=0)
        >>> len(result.bucket_set)
        3
    """
    if num_buckets <= 0:
        raise ValueError(f"num_buckets must be positive, got {num_buckets}")
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")
    if vae_factor <= 0:
        raise ValueError(f"vae_factor must be positive, got {vae_factor}")

    arr = np.asarray(dims, dtype=np.int64)
    if arr.size == 0:
        raise ValueError("dims must contain at least one image")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"dims must have shape (N, 2), got {arr.shape}")
    if (arr <= 0).any():
        raise ValueError("all dims must be positive")

    log_ars = np.log(arr[:, 0].astype(np.float64) / arr[:, 1].astype(np.float64))
    effective_k = min(num_buckets, int(np.unique(log_ars).size))
    centers, labels, iterations = _kmeans_1d(log_ars, effective_k, seed=seed, max_iter=max_iter)

    pixel_budget = target * target
    raw_buckets = [_center_to_bucket(c, pixel_budget, vae_factor) for c in centers]
    seen: dict[tuple[int, int], int] = {}
    deduped: list[Bucket] = []
    surviving_centers: list[float] = []
    surviving_sizes: list[int] = []
    for i, bucket in enumerate(raw_buckets):
        key = bucket.as_tuple()
        if key in seen:
            seen[key] += int((labels == i).sum())
            continue
        seen[key] = int((labels == i).sum())
        deduped.append(bucket)
        surviving_centers.append(float(centers[i]))
        surviving_sizes.append(int((labels == i).sum()))

    bucket_set = BucketSet(
        buckets=tuple(deduped),
        name=name,
        description=description,
        vae_factor=vae_factor,
    )
    loss = crop_loss_summary(arr, bucket_set)
    return AutoBucketResult(
        bucket_set=bucket_set,
        cluster_centers=tuple(surviving_centers),
        cluster_sizes=tuple(surviving_sizes),
        iterations=iterations,
        crop_loss_mean=float(loss.mean),
        requested_k=num_buckets,
    )


def _kmeans_1d(
    values: np.ndarray,
    k: int,
    *,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """1-D k-means with quantile init. Deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    if k == 1:
        centers = np.array([float(values.mean())])
        labels = np.zeros(values.shape, dtype=np.int64)
        return centers, labels, 0

    quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]
    centers = np.quantile(values, quantiles)
    centers = np.unique(centers)
    if centers.size < k:
        extra = rng.choice(values, size=k - centers.size, replace=False)
        centers = np.unique(np.concatenate([centers, extra]))
    centers.sort()

    labels = np.zeros(values.shape, dtype=np.int64)
    iterations = 0
    for it in range(1, max_iter + 1):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new_centers = np.empty_like(centers)
        for i in range(centers.size):
            mask = labels == i
            new_centers[i] = values[mask].mean() if mask.any() else centers[i]
        if np.allclose(new_centers, centers, atol=1e-9):
            centers = new_centers
            iterations = it
            break
        centers = new_centers
        iterations = it
    return centers, labels, iterations


def _center_to_bucket(log_ar: float, pixel_budget: int, vae_factor: int) -> Bucket:
    """Snap a target log-AR + pixel budget to the nearest VAE-aligned bucket."""
    ar = math.exp(log_ar)
    h = math.sqrt(pixel_budget / ar)
    w = ar * h
    w_rounded = max(vae_factor, round(w / vae_factor) * vae_factor)
    h_rounded = max(vae_factor, round(h / vae_factor) * vae_factor)
    return Bucket(int(w_rounded), int(h_rounded))


class AutoBuckets:
    """A :class:`bucketsampler.Strategy` backed by data-derived buckets.

    The class is a thin convenience wrapper: it runs
    :func:`generate_buckets` at construction and then delegates
    ``assign`` / ``assign_many_indices`` to a :class:`FixedBuckets`
    over the generated bucket set. Use it when you want one-line
    bucket derivation inline with your training script, or use
    :func:`generate_buckets` directly when you also want to inspect
    or persist the cluster diagnostics.

    Example:
        >>> import numpy as np
        >>> dims = np.array([[1024, 1024], [2048, 1024], [1024, 2048]] * 50)
        >>> strat = AutoBuckets.from_dims(dims, num_buckets=3, target=1024)
        >>> isinstance(strat.assign(1500, 1500), Bucket)
        True
    """

    def __init__(self, result: AutoBucketResult) -> None:
        from bucketsampler.core.strategies import FixedBuckets

        self.result = result
        self._fixed = FixedBuckets(result.bucket_set)

    @classmethod
    def from_dims(
        cls,
        dims: Sequence[tuple[int, int]] | np.ndarray,
        *,
        num_buckets: int = 8,
        target: int = 1024,
        vae_factor: int = 64,
        seed: int = 0,
        max_iter: int = 100,
        name: str = "auto",
        description: str = "",
    ) -> AutoBuckets:
        """Generate buckets from dims and wrap them in a Strategy."""
        result = generate_buckets(
            dims,
            num_buckets=num_buckets,
            target=target,
            vae_factor=vae_factor,
            seed=seed,
            max_iter=max_iter,
            name=name,
            description=description,
        )
        return cls(result)

    @property
    def bucket_set(self) -> BucketSet:
        return self._fixed.bucket_set

    def assign(self, width: int, height: int) -> Bucket:
        """Pick the closest auto-generated bucket for a single image."""
        return self._fixed.assign(width, height)

    def assign_many_indices(self, dims: Sequence[tuple[int, int]] | np.ndarray) -> np.ndarray:
        """Vectorized assignment over many images."""
        return self._fixed.assign_many_indices(dims)

    def __repr__(self) -> str:
        return (
            "AutoBuckets("
            f"n_buckets={len(self._fixed.bucket_set)}, "
            f"crop_loss_mean={self.result.crop_loss_mean:.3f}"
            ")"
        )


def bucket_set_to_toml(bucket_set: BucketSet) -> str:
    """Serialize a :class:`BucketSet` to a TOML string.

    The output round-trips through :func:`bucketsampler.load_from_toml`.

    Args:
        bucket_set: Buckets to serialize.

    Returns:
        A TOML string ready to be written to disk.
    """
    parts: list[str] = []
    if bucket_set.name:
        parts.append(f'name = "{_escape_toml_string(bucket_set.name)}"')
    if bucket_set.description:
        parts.append(f'description = "{_escape_toml_string(bucket_set.description)}"')
    parts.append(f"vae_factor = {bucket_set.vae_factor}")
    parts.append("")
    for b in bucket_set:
        parts.append("[[buckets]]")
        parts.append(f"width = {b.width}")
        parts.append(f"height = {b.height}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _escape_toml_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')
