"""Train directly from a HuggingFace ``datasets.Dataset``.

Skips the "save to disk, then point at the folder" step. Works for any
dataset whose image column is PIL, raw bytes, or a tensor / numpy
array.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from bucketsampler import (
    BucketBatchSampler,
    BucketedDataset,
    FixedBuckets,
    load_preset,
)


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "this example needs the datasets extra: "
            "pip install 'bucketsampler[hf]'"
        ) from exc

    hf = load_dataset("lambdalabs/pokemon-blip-captions", split="train")

    dataset = BucketedDataset.from_hf(
        hf,
        FixedBuckets(load_preset("sdxl")),
        image_column="image",
        caption_column="text",
    )

    sampler = BucketBatchSampler(dataset, batch_size=4)
    loader = DataLoader(dataset, batch_sampler=sampler)

    for step, batch in enumerate(loader):
        images = batch["image"]
        captions = batch["caption"]
        print(f"step {step}: images={images.shape} captions={captions[:2]}")
        if step >= 3:
            break


if __name__ == "__main__":
    main()
