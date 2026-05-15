"""Derive buckets from the dataset itself, no preset, no TOML.

Useful when the dataset has an unusual AR distribution (most of SDXL's
9 buckets sit unused, for example, if you train on screenshots).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from bucketsampler import AutoBuckets, BucketBatchSampler, BucketedDataset


def read_dims(paths):
    dims = []
    for p in paths:
        with Image.open(p) as img:
            dims.append((int(img.width), int(img.height)))
    return np.asarray(dims, dtype=np.int64)


def main(data_dir: str) -> None:
    paths = sorted(Path(data_dir).glob("*.jpg"))
    dims = read_dims(paths)

    strategy = AutoBuckets.from_dims(
        dims,
        num_buckets=8,
        target=1024,
        vae_factor=64,
        seed=0,
    )
    print(f"derived {len(strategy.bucket_set)} buckets:")
    for b in strategy.bucket_set:
        print(f"  {b}  AR={b.aspect_ratio:.3f}")
    print(f"mean crop loss: {strategy.result.crop_loss_mean * 100:.2f}%")

    dataset = BucketedDataset(paths=paths, strategy=strategy)
    sampler = BucketBatchSampler(dataset, batch_size=4)
    loader = DataLoader(dataset, batch_sampler=sampler)

    for step, batch in enumerate(loader):
        print(f"step {step}: shape={batch['image'].shape}")
        if step >= 5:
            break


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "./data")
