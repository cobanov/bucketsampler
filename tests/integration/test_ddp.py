"""DDP correctness via ``torch.multiprocessing.spawn``.

Spins up two CPU-only worker processes that play the role of two ranks.
Each rank materializes its own ``BucketBatchSampler`` with shared seed and
epoch but its own ``rank`` index, then dumps its batches to a JSON file.
The parent test reads both files and verifies the DDP invariants:

  - ranks yield disjoint dataset indices across all buckets
  - ranks yield the same total batch count
  - within a batch, all indices live in the same bucket

CPU spawn is used so the test runs in CI without GPUs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch.multiprocessing as mp


def _ddp_worker(
    rank: int,
    num_replicas: int,
    bucket_indices: list[int],
    batch_size: int,
    seed: int,
    epoch: int,
    out_dir: str,
) -> None:
    from bucketsampler.torch import BucketBatchSampler

    arr = np.array(bucket_indices, dtype=np.int64)
    sampler = BucketBatchSampler(
        arr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_replicas=num_replicas,
        rank=rank,
        seed=seed,
    )
    sampler.set_epoch(epoch)
    batches = [list(b) for b in sampler]
    Path(out_dir, f"rank_{rank}.json").write_text(json.dumps(batches))


@pytest.mark.integration
class TestDDPSpawn:
    def test_two_ranks_disjoint_and_balanced(self, tmp_path: Path) -> None:
        bucket_indices = [0] * 14 + [1] * 10 + [2] * 8
        out_dir = tmp_path / "ddp_out"
        out_dir.mkdir()

        mp.spawn(
            _ddp_worker,
            args=(2, bucket_indices, 2, 42, 0, str(out_dir)),
            nprocs=2,
            join=True,
        )

        r0 = json.loads((out_dir / "rank_0.json").read_text())
        r1 = json.loads((out_dir / "rank_1.json").read_text())

        # Same total batch count: required for DDP gradient sync
        assert len(r0) == len(r1), f"rank 0 yielded {len(r0)} batches, rank 1 yielded {len(r1)}"

        # Each batch is single-bucket
        idx_arr = np.array(bucket_indices)
        for batch in r0 + r1:
            assigned = {int(idx_arr[i]) for i in batch}
            assert len(assigned) == 1

        # Disjoint indices across ranks
        seen_0 = {i for b in r0 for i in b}
        seen_1 = {i for b in r1 for i in b}
        assert seen_0.isdisjoint(seen_1), f"ranks overlap on indices: {seen_0 & seen_1}"

    def test_two_ranks_same_seed_different_epochs(self, tmp_path: Path) -> None:
        # Verifies set_epoch is honored across processes
        bucket_indices = [0] * 12 + [1] * 8
        out_e0 = tmp_path / "epoch0"
        out_e1 = tmp_path / "epoch1"
        out_e0.mkdir()
        out_e1.mkdir()

        mp.spawn(
            _ddp_worker,
            args=(2, bucket_indices, 2, 99, 0, str(out_e0)),
            nprocs=2,
            join=True,
        )
        mp.spawn(
            _ddp_worker,
            args=(2, bucket_indices, 2, 99, 1, str(out_e1)),
            nprocs=2,
            join=True,
        )

        e0_r0 = json.loads((out_e0 / "rank_0.json").read_text())
        e1_r0 = json.loads((out_e1 / "rank_0.json").read_text())
        # Different epochs should produce different orderings
        assert e0_r0 != e1_r0
