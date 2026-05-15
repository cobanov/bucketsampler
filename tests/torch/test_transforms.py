"""Tests for bucketsampler.torch.transforms."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from bucketsampler import Bucket
from bucketsampler.torch import BucketResize


class TestBucketResize:
    def test_output_shape_matches_bucket(self):
        t = BucketResize(Bucket(1024, 512))
        img = Image.new("RGB", (2048, 1024), color=(128, 128, 128))
        out = t(img)
        assert out.shape == (3, 512, 1024)

    def test_output_dtype_and_range(self):
        t = BucketResize(Bucket(64, 64))
        img = Image.new("RGB", (128, 128), color=(64, 128, 192))
        out = t(img)
        assert out.dtype == torch.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_output_chw_layout(self):
        t = BucketResize(Bucket(64, 48))
        img = Image.new("RGB", (128, 96), color=(10, 20, 30))
        out = t(img)
        # CHW with C=3
        assert out.shape[0] == 3
        assert out.shape[1] == 48
        assert out.shape[2] == 64

    def test_grayscale_converted_to_rgb(self):
        t = BucketResize(Bucket(64, 64))
        img = Image.new("L", (128, 128), color=128)
        out = t(img)
        assert out.shape[0] == 3

    def test_handles_wide_source(self):
        # 4:1 image into square bucket: crops horizontally
        t = BucketResize(Bucket(64, 64))
        img = Image.new("RGB", (256, 64), color=(255, 0, 0))
        out = t(img)
        assert out.shape == (3, 64, 64)

    def test_handles_tall_source(self):
        t = BucketResize(Bucket(64, 64))
        img = Image.new("RGB", (64, 256), color=(0, 255, 0))
        out = t(img)
        assert out.shape == (3, 64, 64)

    def test_handles_upscale(self):
        # Source smaller than bucket: upscales (with a warning in real life, but works)
        t = BucketResize(Bucket(128, 128))
        img = Image.new("RGB", (32, 32), color=(0, 0, 255))
        out = t(img)
        assert out.shape == (3, 128, 128)

    def test_rejects_non_pil(self):
        t = BucketResize(Bucket(64, 64))
        with pytest.raises(TypeError):
            t(torch.zeros(3, 64, 64))  # type: ignore[arg-type]

    def test_repr(self):
        t = BucketResize(Bucket(1024, 512))
        assert "1024x512" in repr(t)
