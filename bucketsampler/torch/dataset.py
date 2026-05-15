"""``BucketedDataset``: a torch.utils.data.Dataset that buckets images.

The dataset accepts a list of image paths plus a :class:`Strategy` and
pre-computes each image's bucket at construction time via a lazy PIL
header read (``Image.open(path).size`` without ``.load()``). The dataset's
:attr:`bucket_indices` array is the bridge to
:class:`bucketsampler.torch.BucketBatchSampler`, which uses it to build
per-bucket index queues.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from bucketsampler.core.bucket import Bucket
from bucketsampler.core.strategies import Strategy
from bucketsampler.torch.transforms import BucketResize


class BucketedDataset(Dataset[dict[str, Any]]):
    """Wrap a list of image paths with bucket assignments and per-bucket transforms.

    On construction, the dataset reads each image's header to get its
    ``(width, height)`` (no full decode) and assigns it to a bucket via the
    given :class:`Strategy`. The expensive image decode happens lazily inside
    :meth:`__getitem__`, where the image is resized + center-cropped to its
    bucket's exact dims so that every sample within a bucket has identical
    output shape.

    Args:
        paths: Image file paths.
        strategy: Bucket assignment strategy (typically
            :class:`bucketsampler.FixedBuckets` wrapping a preset).
        captions: Optional per-image caption strings, returned under the
            ``"caption"`` key. If provided, must be the same length as ``paths``.
        transform: Optional callable applied to the resized + cropped tensor
            (CHW float32 in ``[0, 1]``). Use this for normalization or any
            user-side augmentation. Defaults to identity.
        num_workers: Threads used during construction to read image headers
            in parallel. Header reads are I/O bound; ``8`` is usually fine
            even on slow disks. Set to ``0`` to do serial reads.

    Attributes:
        paths: The input image paths.
        bucket_indices: ``(N,)`` int64 array of bucket-set indices, in the
            same order as ``paths``. Consumed by the sampler.

    Example:
        >>> from bucketsampler import FixedBuckets, load_preset
        >>> ds = BucketedDataset(  # doctest: +SKIP
        ...     paths=image_paths,
        ...     strategy=FixedBuckets(load_preset("sdxl")),
        ... )
        >>> sample = ds[0]  # doctest: +SKIP
        >>> sample["image"].shape  # doctest: +SKIP
        torch.Size([3, 1024, 1024])
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        strategy: Strategy,
        *,
        captions: Sequence[str] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_workers: int = 8,
    ) -> None:
        if len(paths) == 0:
            raise ValueError("BucketedDataset requires at least one path")
        if captions is not None and len(captions) != len(paths):
            raise ValueError(
                f"captions length ({len(captions)}) must match paths length ({len(paths)})"
            )
        self.paths: list[Path] = [Path(p) for p in paths]
        self.captions: list[str] | None = list(captions) if captions is not None else None
        self.strategy = strategy
        self.transform = transform

        dims = _read_dims_parallel(self.paths, num_workers=num_workers)
        self.bucket_indices: np.ndarray = strategy.assign_many_indices(dims).astype(np.int64)
        self._resizers: dict[int, BucketResize] = {}

    @property
    def bucket_set(self) -> Any:
        return self.strategy.bucket_set

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        bucket_idx = int(self.bucket_indices[idx])
        bucket: Bucket = self.strategy.bucket_set[bucket_idx]
        resizer = self._resizers.get(bucket_idx)
        if resizer is None:
            resizer = BucketResize(bucket)
            self._resizers[bucket_idx] = resizer
        with Image.open(path) as raw:
            tensor = resizer(raw)
        if self.transform is not None:
            tensor = self.transform(tensor)
        sample: dict[str, Any] = {
            "image": tensor,
            "bucket": bucket,
            "bucket_idx": bucket_idx,
            "path": str(path),
        }
        if self.captions is not None:
            sample["caption"] = self.captions[idx]
        return sample

    def __repr__(self) -> str:
        return f"BucketedDataset(n={len(self.paths)}, n_buckets={len(self.strategy.bucket_set)})"


def _read_one_dim(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return (int(img.width), int(img.height))


def _read_dims_parallel(paths: Sequence[Path], *, num_workers: int) -> np.ndarray:
    if num_workers <= 1:
        return np.array([_read_one_dim(p) for p in paths], dtype=np.int64)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_read_one_dim, paths))
    return np.array(results, dtype=np.int64)
