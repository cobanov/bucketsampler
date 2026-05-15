"""``BucketedDataset``: a torch.utils.data.Dataset that buckets images.

The dataset delegates to a :class:`_DataSource` for dim reads and lazy
image opens. Filesystem paths and HuggingFace ``datasets.Dataset``
sources are both supported. The :attr:`bucket_indices` array is the
bridge to :class:`bucketsampler.torch.BucketBatchSampler`, which uses
it to build per-bucket index queues.

Construct via one of:

  - ``BucketedDataset(paths, strategy, ...)`` (back-compat, paths-based)
  - ``BucketedDataset.from_paths(paths, strategy, ...)`` (same, explicit)
  - ``BucketedDataset.from_hf(hf_dataset, strategy, ...)``
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from bucketsampler.core.bucket import Bucket, BucketSet
from bucketsampler.core.strategies import Strategy
from bucketsampler.torch._source import _DataSource, _PathSource
from bucketsampler.torch.transforms import BucketResize

if TYPE_CHECKING:
    import datasets


class BucketedDataset(Dataset[dict[str, Any]]):
    """Wrap an image source with bucket assignments and per-bucket transforms.

    On construction, the dataset reads each item's dim cheaply (PIL header
    read for filesystem images) and assigns it to a bucket via the given
    :class:`Strategy`. The expensive image decode happens lazily inside
    :meth:`__getitem__`, where the image is resized + center-cropped to its
    bucket's exact dims so every sample within a bucket has identical
    output shape.

    Args:
        paths: Image file paths.
        strategy: Bucket assignment strategy (typically
            :class:`bucketsampler.FixedBuckets` wrapping a preset).
        captions: Optional per-image caption strings, returned under the
            ``"caption"`` key. Must match ``paths`` in length.
        transform: Optional callable applied to the resized + cropped tensor
            (CHW float32 in ``[0, 1]``). Use this for normalization or
            user-side augmentation. Defaults to identity.
        num_workers: Threads used during construction to read item dims in
            parallel. Header reads are I/O bound; ``8`` is fine even on
            slow disks. Set to ``<= 1`` for serial reads.

    Attributes:
        bucket_indices: ``(N,)`` int64 array of bucket-set indices, in
            source order. Consumed by the sampler.
        strategy: The bucketing strategy passed in at construction.

    Example:
        >>> from bucketsampler import FixedBuckets, load_preset
        >>> ds = BucketedDataset(  # doctest: +SKIP
        ...     paths=image_paths,
        ...     strategy=FixedBuckets(load_preset("sdxl")),
        ... )
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
        source = _PathSource(paths, captions=captions)
        self._init_with_source(
            source=source,
            strategy=strategy,
            transform=transform,
            num_workers=num_workers,
        )

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[str | Path],
        strategy: Strategy,
        *,
        captions: Sequence[str] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_workers: int = 8,
    ) -> BucketedDataset:
        """Construct from a list of image paths. Same as the default constructor."""
        return cls(
            paths=paths,
            strategy=strategy,
            captions=captions,
            transform=transform,
            num_workers=num_workers,
        )

    @classmethod
    def from_hf(
        cls,
        hf_dataset: datasets.Dataset,
        strategy: Strategy,
        *,
        image_column: str = "image",
        caption_column: str | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_workers: int = 8,
    ) -> BucketedDataset:
        """Construct from a HuggingFace ``datasets.Dataset``.

        Args:
            hf_dataset: A map-style ``datasets.Dataset``. Streaming
                ``IterableDataset`` is not supported in this milestone.
            strategy: Bucket assignment strategy.
            image_column: Column holding the image. PIL, raw bytes
                (``datasets.Image(decode=False)``), and numpy / torch
                tensors are all accepted.
            caption_column: Optional column holding captions.
            transform: Optional tensor transform (same semantics as the
                path-based constructor).
            num_workers: Threads for the dim-read pass at construction.

        Returns:
            A :class:`BucketedDataset` that pulls images from the wrapped
            HF dataset on demand.

        Raises:
            ImportError: If the ``datasets`` package is not installed.

        Example:
            >>> from datasets import Dataset  # doctest: +SKIP
            >>> hf = Dataset.from_dict({"image": pil_list, "text": captions})  # doctest: +SKIP
            >>> ds = BucketedDataset.from_hf(  # doctest: +SKIP
            ...     hf, strategy, image_column="image", caption_column="text",
            ... )
        """
        try:
            from bucketsampler.hf.adapter import _HFSource
        except ImportError as exc:
            raise ImportError(
                "BucketedDataset.from_hf requires the datasets package. "
                "Install with: pip install bucketsampler[hf]"
            ) from exc
        source = _HFSource(
            hf_dataset,
            image_column=image_column,
            caption_column=caption_column,
        )
        obj = cls.__new__(cls)
        obj._init_with_source(
            source=source,
            strategy=strategy,
            transform=transform,
            num_workers=num_workers,
        )
        return obj

    def _init_with_source(
        self,
        *,
        source: _DataSource,
        strategy: Strategy,
        transform: Callable[[torch.Tensor], torch.Tensor] | None,
        num_workers: int,
    ) -> None:
        if len(source) == 0:
            raise ValueError("BucketedDataset requires at least one item")
        self._source = source
        self.strategy = strategy
        self.transform = transform

        dims = source.read_all_dims(num_workers=num_workers)
        self.bucket_indices: np.ndarray = strategy.assign_many_indices(dims).astype(np.int64)
        self._resizers: dict[int, BucketResize] = {}

    @property
    def bucket_set(self) -> BucketSet:
        return self.strategy.bucket_set

    @property
    def paths(self) -> list[Path]:
        """Convenience for path-backed datasets; raises for other sources."""
        if not isinstance(self._source, _PathSource):
            raise AttributeError(
                "paths is only available for path-backed datasets; "
                "use identifier()/__getitem__ for other sources"
            )
        return self._source.paths

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        bucket_idx = int(self.bucket_indices[idx])
        bucket: Bucket = self.strategy.bucket_set[bucket_idx]
        resizer = self._resizers.get(bucket_idx)
        if resizer is None:
            resizer = BucketResize(bucket)
            self._resizers[bucket_idx] = resizer
        image = self._source.open_image(idx)
        try:
            tensor = resizer(image)
        finally:
            if hasattr(image, "close"):
                with contextlib.suppress(Exception):
                    image.close()
        if self.transform is not None:
            tensor = self.transform(tensor)
        sample: dict[str, Any] = {
            "image": tensor,
            "bucket": bucket,
            "bucket_idx": bucket_idx,
            "path": self._source.identifier(idx),
        }
        caption = self._source.get_caption(idx)
        if caption is not None:
            sample["caption"] = caption
        return sample

    def __repr__(self) -> str:
        return f"BucketedDataset(n={len(self)}, n_buckets={len(self.strategy.bucket_set)})"
