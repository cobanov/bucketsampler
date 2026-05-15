"""``BucketBatchSampler``: per-bucket batches, shuffle, DDP-aware.

Each yielded batch is a list of dataset indices that all live in the same
bucket, so the corresponding samples have identical ``(H, W)`` after the
:class:`bucketsampler.torch.BucketResize` transform. The sampler is
DDP-correct: with the same ``seed`` and ``epoch``, ranks shuffle in lock
step but slice into disjoint subsets.

Defaults are tuned for diffusion training: ``shuffle=True``,
``drop_last=True``, and bucket-size-weighted ordering of batches (a bucket
with 10K images is seen 100x more often than one with 100, by virtue of
contributing 100x more batches to the shuffled batch list).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from bucketsampler.torch.dataset import BucketedDataset


class BucketBatchSampler(Sampler[list[int]]):
    """Yields per-bucket batches of dataset indices.

    Args:
        bucket_indices: Either a :class:`BucketedDataset`, an
            ``(N,)`` int array of bucket assignments, or any sequence of
            ints. Index ``i`` is the bucket assignment of dataset item ``i``.
        batch_size: Batch size. Must be positive.
        shuffle: If ``True``, shuffle within each bucket every epoch and
            shuffle the order of batches across buckets.
        drop_last: If ``True``, drop the trailing incomplete batch within
            each bucket per rank. If ``False``, the final partial batch is
            yielded. Either way, all ranks always yield the same number of
            batches (the per-bucket index array is trimmed to a multiple of
            ``num_replicas`` before slicing so DDP stays in sync).
        num_replicas: World size for DDP. ``1`` means no DDP.
        rank: This process's rank. Must be in ``[0, num_replicas)``.
        seed: Base RNG seed. Combined with ``epoch`` so every epoch shuffles
            differently while remaining reproducible.

    Note:
        For DDP, you MUST call :meth:`set_epoch` at the start of every epoch.
        Without it, all epochs see the same shuffle order. Per-rank batch
        counts are kept equal by trimming each bucket's index array to a
        multiple of ``num_replicas`` before slicing; the dropped tail
        samples are seen again on the next epoch thanks to the reshuffle.

    Example:
        >>> from bucketsampler.torch import BucketedDataset, BucketBatchSampler
        >>> from torch.utils.data import DataLoader
        >>> ds = BucketedDataset(paths, strategy=strategy)  # doctest: +SKIP
        >>> sampler = BucketBatchSampler(ds, batch_size=4)  # doctest: +SKIP
        >>> loader = DataLoader(ds, batch_sampler=sampler)  # doctest: +SKIP
    """

    def __init__(
        self,
        bucket_indices: BucketedDataset | np.ndarray | Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        raw = getattr(bucket_indices, "bucket_indices", bucket_indices)
        indices_array = np.asarray(raw, dtype=np.int64)
        if indices_array.ndim != 1:
            raise ValueError(f"bucket_indices must be 1-D, got shape {indices_array.shape}")
        if indices_array.size and indices_array.min() < 0:
            raise ValueError("bucket_indices must be non-negative")

        self.bucket_indices: np.ndarray = indices_array
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch. Re-seeds the within-epoch shuffle.

        Must be called at the start of every epoch when running under DDP.
        """
        self.epoch = int(epoch)

    def _build_batches(self) -> list[list[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        if self.bucket_indices.size == 0:
            return []
        n_buckets = int(self.bucket_indices.max()) + 1

        all_batches: list[list[int]] = []
        for b in range(n_buckets):
            idxs = np.where(self.bucket_indices == b)[0]
            if idxs.size == 0:
                continue
            if self.shuffle:
                rng.shuffle(idxs)
            if self.num_replicas > 1:
                trimmed = (idxs.size // self.num_replicas) * self.num_replicas
                idxs = idxs[:trimmed][self.rank :: self.num_replicas]
            n_full = idxs.size // self.batch_size
            for i in range(n_full):
                start = i * self.batch_size
                all_batches.append(idxs[start : start + self.batch_size].tolist())
            if not self.drop_last:
                tail = idxs[n_full * self.batch_size :]
                if tail.size > 0:
                    all_batches.append(tail.tolist())

        if self.shuffle and all_batches:
            order = rng.permutation(len(all_batches))
            all_batches = [all_batches[int(i)] for i in order]
        return all_batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._build_batches())

    def __len__(self) -> int:
        if self.bucket_indices.size == 0:
            return 0
        n_buckets = int(self.bucket_indices.max()) + 1
        total = 0
        for b in range(n_buckets):
            n = int((self.bucket_indices == b).sum())
            if n == 0:
                continue
            if self.num_replicas > 1:
                n = n // self.num_replicas
            if n == 0:
                continue
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

    def __repr__(self) -> str:
        return (
            "BucketBatchSampler("
            f"n={self.bucket_indices.size}, "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, "
            f"drop_last={self.drop_last}, "
            f"num_replicas={self.num_replicas}, "
            f"rank={self.rank}"
            ")"
        )
