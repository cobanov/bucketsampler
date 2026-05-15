"""DDP training loop sketch.

Three lines differ from the single-process loop:

  - sampler is constructed with ``num_replicas=world_size`` and ``rank=rank``
  - ``sampler.set_epoch(epoch)`` is called at the top of each epoch
  - DataLoader's ``num_workers`` and ``pin_memory`` are tuned per rank
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from bucketsampler import (
    BucketBatchSampler,
    BucketedDataset,
    FixedBuckets,
    load_preset,
)


def main(data_dir: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    paths = sorted(Path(data_dir).glob("*.jpg"))
    strategy = FixedBuckets(load_preset("sdxl"))
    dataset = BucketedDataset(paths=paths, strategy=strategy)

    sampler = BucketBatchSampler(
        dataset,
        batch_size=4,
        num_replicas=world_size,
        rank=rank,
        seed=42,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)

    for epoch in range(3):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            pixels = batch["image"].to(f"cuda:{local_rank}" if world_size > 1 else "cpu")
            # ... model forward / backward here
            if rank == 0 and step % 50 == 0:
                print(f"epoch={epoch} step={step} batch={pixels.shape}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "./data")
