"""Precompute VAE latents, then train without the VAE forward pass.

Run this once per dataset / VAE pair (e.g. when you switch base models).
The training loop then iterates ``BucketedLatentDataset`` instead of
``BucketedDataset`` and skips the encoder entirely.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from bucketsampler import (
    BucketBatchSampler,
    BucketedDataset,
    BucketedLatentDataset,
    FixedBuckets,
    load_preset,
    precompute_latents,
)


def precompute(data_dir: str, latent_dir: str, vae_id: str) -> None:
    from bucketsampler.cache.vae_adapters.sd import SDVAEEncoder

    paths = sorted(Path(data_dir).glob("*.jpg"))
    strategy = FixedBuckets(load_preset("sdxl"))
    dataset = BucketedDataset(paths=paths, strategy=strategy)
    encoder = SDVAEEncoder(model_id=vae_id, dtype="bfloat16")  # type: ignore[arg-type]
    manifest = precompute_latents(
        dataset,
        encoder,
        output_dir=latent_dir,
        batch_size=8,
        progress_callback=lambda d, t: print(f"\rencoded {d}/{t}", end=""),
    )
    print()
    print(f"wrote {len(manifest.buckets)} bucket files to {latent_dir}")


def train(latent_dir: str) -> None:
    latent_ds = BucketedLatentDataset(latent_dir)
    sampler = BucketBatchSampler(latent_ds, batch_size=8)
    loader = DataLoader(latent_ds, batch_sampler=sampler)

    for step, batch in enumerate(loader):
        latents = batch["latents"]
        print(f"step {step}: latents.shape={latents.shape}")
        if step >= 3:
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("usage: precompute_latents.py <data_dir> <latent_dir> <vae_id>")
        sys.exit(2)
    precompute(sys.argv[1], sys.argv[2], sys.argv[3])
    train(sys.argv[2])
