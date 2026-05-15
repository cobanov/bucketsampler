"""Tests for VAE precomputation and the latent dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from bucketsampler import (
    BucketedLatentDataset,
    FixedBuckets,
    LatentManifest,
    VAEEncoder,
    load_preset,
    precompute_latents,
)
from bucketsampler.torch import BucketBatchSampler, BucketedDataset


class FakeVAE:
    """Deterministic, dependency-free VAE for tests.

    Produces ``(B, C, H/f, W/f)`` latents by averaging non-overlapping
    ``fxf`` patches and projecting to ``C`` channels with a fixed
    permutation. Conforms to the :class:`VAEEncoder` protocol.
    """

    downsample_factor = 8
    latent_channels = 4
    scale_factor = 0.18215

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        _b, _c, h, w = pixels.shape
        f = self.downsample_factor
        if h % f or w % f:
            raise ValueError(f"({h}, {w}) not divisible by {f}")
        pooled = torch.nn.functional.avg_pool2d(pixels, kernel_size=f, stride=f)
        # Project 3 input channels onto 4 output channels with a fixed mix
        rng = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.33, 0.33, 0.34],
            ]
        )
        flat = pooled.permute(0, 2, 3, 1)
        latent = flat @ rng.t()
        return latent.permute(0, 3, 1, 2) * self.scale_factor


def _write_image(p: Path, w: int, h: int, color: tuple[int, int, int] = (128, 128, 128)) -> None:
    Image.new("RGB", (w, h), color).save(p, format="JPEG", quality=40)


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    specs = [
        (1024, 1024),
        (1024, 1024),
        (2048, 1024),
        (2048, 1024),
        (1024, 2048),
    ]
    for i, (w, h) in enumerate(specs):
        _write_image(tmp_path / f"img_{i}.jpg", w, h)
    return tmp_path


@pytest.fixture
def dataset(image_dir):
    paths = sorted(image_dir.glob("*.jpg"))
    strategy = FixedBuckets(load_preset("sdxl"))
    return BucketedDataset(paths=paths, strategy=strategy, captions=[f"cap {i}" for i in range(5)])


class TestVAEEncoderProtocol:
    def test_fake_vae_conforms(self):
        assert isinstance(FakeVAE(), VAEEncoder)

    def test_fake_vae_shape(self):
        v = FakeVAE()
        out = v.encode(torch.zeros(2, 3, 1024, 1024))
        assert out.shape == (2, 4, 128, 128)


class TestPrecomputeLatents:
    def test_writes_manifest_and_buckets(self, dataset, tmp_path):
        out = tmp_path / "latents"
        manifest = precompute_latents(dataset, FakeVAE(), output_dir=out, batch_size=2)
        assert (out / "manifest.json").is_file()
        assert manifest.version == 1
        assert manifest.latent_channels == 4
        assert manifest.downsample_factor == 8
        # 3 distinct buckets in this dataset
        assert len(manifest.buckets) == len(set(dataset.bucket_indices.tolist()))
        for row in manifest.buckets:
            assert (out / row.file).is_file()

    def test_manifest_matches_disk(self, dataset, tmp_path):
        out = tmp_path / "latents"
        manifest = precompute_latents(dataset, FakeVAE(), output_dir=out)
        loaded = LatentManifest.from_dict(json.loads((out / "manifest.json").read_text()))
        assert loaded.version == manifest.version
        assert loaded.scale_factor == manifest.scale_factor
        assert len(loaded.buckets) == len(manifest.buckets)

    def test_captions_written_when_provided(self, dataset, tmp_path):
        out = tmp_path / "latents"
        manifest = precompute_latents(dataset, FakeVAE(), output_dir=out)
        assert manifest.has_captions
        assert (out / "captions.json").is_file()
        captions = json.loads((out / "captions.json").read_text())
        # Each bucket carries its captions
        total = sum(len(v) for v in captions.values())
        assert total == len(dataset)

    def test_no_captions(self, image_dir, tmp_path):
        paths = sorted(image_dir.glob("*.jpg"))
        strategy = FixedBuckets(load_preset("sdxl"))
        ds = BucketedDataset(paths=paths, strategy=strategy)  # no captions
        out = tmp_path / "latents"
        manifest = precompute_latents(ds, FakeVAE(), output_dir=out)
        assert not manifest.has_captions
        assert not (out / "captions.json").exists()

    def test_progress_callback(self, dataset, tmp_path):
        out = tmp_path / "latents"
        calls = []
        precompute_latents(
            dataset,
            FakeVAE(),
            output_dir=out,
            batch_size=2,
            progress_callback=lambda d, t: calls.append((d, t)),
        )
        assert calls
        assert calls[-1] == (len(dataset), len(dataset))

    def test_dtype_persistence(self, dataset, tmp_path):
        out = tmp_path / "latents"
        manifest = precompute_latents(dataset, FakeVAE(), output_dir=out, dtype=torch.bfloat16)
        assert "bfloat16" in manifest.dtype


class TestBucketedLatentDataset:
    def test_load_and_iterate(self, dataset, tmp_path):
        out = tmp_path / "latents"
        precompute_latents(dataset, FakeVAE(), output_dir=out, batch_size=2)
        latent_ds = BucketedLatentDataset(out)
        assert len(latent_ds) == len(dataset)

    def test_sample_shape_matches_bucket(self, dataset, tmp_path):
        out = tmp_path / "latents"
        precompute_latents(dataset, FakeVAE(), output_dir=out, batch_size=2)
        latent_ds = BucketedLatentDataset(out)
        for i in range(len(latent_ds)):
            sample = latent_ds[i]
            b = sample["bucket"]
            assert sample["latents"].shape == (4, b.height // 8, b.width // 8)

    def test_caption_propagates(self, dataset, tmp_path):
        out = tmp_path / "latents"
        precompute_latents(dataset, FakeVAE(), output_dir=out)
        latent_ds = BucketedLatentDataset(out)
        sample = latent_ds[0]
        assert "caption" in sample

    def test_bucket_indices_for_sampler(self, dataset, tmp_path):
        out = tmp_path / "latents"
        precompute_latents(dataset, FakeVAE(), output_dir=out)
        latent_ds = BucketedLatentDataset(out)
        assert latent_ds.bucket_indices.dtype == np.int64
        # Should be usable as the indices source for BucketBatchSampler
        sampler = BucketBatchSampler(latent_ds, batch_size=1, shuffle=False, drop_last=False)
        seen: list[int] = []
        for batch in sampler:
            seen.extend(batch)
        assert sorted(seen) == list(range(len(latent_ds)))

    def test_loader_round_trip(self, dataset, tmp_path):
        out = tmp_path / "latents"
        precompute_latents(dataset, FakeVAE(), output_dir=out)
        latent_ds = BucketedLatentDataset(out)
        sampler = BucketBatchSampler(latent_ds, batch_size=2, shuffle=False, drop_last=False)

        def collate(samples):
            return {
                "latents": torch.stack([s["latents"] for s in samples]),
                "bucket": [s["bucket"] for s in samples],
            }

        loader = DataLoader(latent_ds, batch_sampler=sampler, collate_fn=collate)
        for batch in loader:
            assert all(b == batch["bucket"][0] for b in batch["bucket"])

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BucketedLatentDataset(tmp_path / "no-such-dir")

    def test_bucket_set_reconstructed(self, dataset, tmp_path):
        out = tmp_path / "latents"
        precompute_latents(dataset, FakeVAE(), output_dir=out)
        latent_ds = BucketedLatentDataset(out)
        bs = latent_ds.bucket_set
        assert bs.vae_factor == 8
        assert len(bs) == len(latent_ds.manifest.buckets)
