"""Tests for bucketsampler.torch.dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from bucketsampler import FixedBuckets, load_preset
from bucketsampler.torch import BucketBatchSampler, BucketedDataset


@pytest.fixture
def strategy():
    return FixedBuckets(load_preset("sdxl"))


class TestConstruction:
    def test_basic(self, image_paths, strategy):
        ds = BucketedDataset(paths=image_paths, strategy=strategy)
        assert len(ds) == len(image_paths)
        assert ds.bucket_indices.shape == (len(image_paths),)
        assert ds.bucket_indices.dtype == np.int64

    def test_empty_paths_raises(self, strategy):
        with pytest.raises(ValueError):
            BucketedDataset(paths=[], strategy=strategy)

    def test_caption_length_mismatch_raises(self, image_paths, strategy):
        with pytest.raises(ValueError):
            BucketedDataset(
                paths=image_paths,
                strategy=strategy,
                captions=["only one"],
            )

    def test_serial_dim_read(self, image_paths, strategy):
        # num_workers=0 forces serial reads; must match parallel result
        ds_parallel = BucketedDataset(paths=image_paths, strategy=strategy, num_workers=8)
        ds_serial = BucketedDataset(paths=image_paths, strategy=strategy, num_workers=0)
        assert np.array_equal(ds_parallel.bucket_indices, ds_serial.bucket_indices)


class TestGetitem:
    def test_returns_image_and_bucket(self, image_paths, strategy):
        ds = BucketedDataset(paths=image_paths, strategy=strategy)
        sample = ds[0]
        assert "image" in sample
        assert "bucket" in sample
        assert "bucket_idx" in sample
        assert "path" in sample
        assert isinstance(sample["image"], torch.Tensor)

    def test_image_shape_matches_bucket(self, image_paths, strategy):
        ds = BucketedDataset(paths=image_paths, strategy=strategy)
        for i in range(len(ds)):
            sample = ds[i]
            bucket = sample["bucket"]
            assert sample["image"].shape == (3, bucket.height, bucket.width)

    def test_captions_propagate(self, image_paths, strategy):
        captions = [f"caption {i}" for i in range(len(image_paths))]
        ds = BucketedDataset(paths=image_paths, strategy=strategy, captions=captions)
        assert ds[3]["caption"] == "caption 3"

    def test_no_captions_omitted_from_sample(self, image_paths, strategy):
        ds = BucketedDataset(paths=image_paths, strategy=strategy)
        assert "caption" not in ds[0]

    def test_user_transform_applied(self, image_paths, strategy):
        # Identity-ish transform that doubles values; result should reflect it
        ds = BucketedDataset(
            paths=image_paths,
            strategy=strategy,
            transform=lambda t: t * 2,
        )
        raw = BucketedDataset(paths=image_paths[:1], strategy=strategy)[0]["image"]
        transformed = ds[0]["image"]
        torch.testing.assert_close(transformed, raw * 2)


class TestSamplerIntegration:
    def test_batches_have_uniform_shape(self, image_paths, strategy):
        ds = BucketedDataset(paths=image_paths, strategy=strategy)
        sampler = BucketBatchSampler(ds, batch_size=2, shuffle=True, drop_last=False, seed=42)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=_collate)
        for batch in loader:
            images = batch["image"]
            assert images.ndim == 4
            # All shapes within a batch must match (sampler guarantee)
            assert all(t.shape == images[0].shape for t in images)

    def test_each_index_seen_once_per_epoch(self, image_paths, strategy):
        ds = BucketedDataset(paths=image_paths, strategy=strategy)
        sampler = BucketBatchSampler(ds, batch_size=2, shuffle=False, drop_last=False)
        seen = []
        for batch in sampler:
            seen.extend(batch)
        # Without DDP, drop_last=False, no shuffle: every index appears exactly once
        assert sorted(seen) == list(range(len(ds)))


def _collate(samples):
    # Collate list of dicts into a dict of lists/tensors
    out = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out
