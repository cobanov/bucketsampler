"""Minimal end-to-end training loop using bucketsampler.

Replace the FakeUNet with your real model. Everything else (dataset
wrapping, sampler, DataLoader) is what you would copy into a real
training script.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bucketsampler import (
    BucketBatchSampler,
    BucketedDataset,
    FixedBuckets,
    load_preset,
)


class FakeUNet(torch.nn.Module):
    """Placeholder. Swap with diffusers' UNet2DConditionModel or similar."""

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.mean(dim=(2, 3))


def main(data_dir: str) -> None:
    paths = sorted(Path(data_dir).glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no .jpg files under {data_dir}")

    strategy = FixedBuckets(load_preset("sdxl"))
    dataset = BucketedDataset(paths=paths, strategy=strategy)
    sampler = BucketBatchSampler(dataset, batch_size=4, seed=0)

    def collate(samples):
        return {
            "image": torch.stack([s["image"] for s in samples]),
            "bucket": [s["bucket"] for s in samples],
        }

    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=2)
    model = FakeUNet()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(2):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            pixels = batch["image"]
            optim.zero_grad()
            output = model(pixels)
            loss = output.pow(2).mean()
            loss.backward()
            optim.step()
            if step % 10 == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"bucket={batch['bucket'][0]} loss={loss.item():.4f}"
                )


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "./data")
