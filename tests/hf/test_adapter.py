"""Tests for the HuggingFace ``datasets`` adapter."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

datasets = pytest.importorskip("datasets")

from bucketsampler import FixedBuckets, load_preset
from bucketsampler.hf.adapter import _array_to_pil, _extract_dim, _HFSource, _to_pil
from bucketsampler.torch import BucketBatchSampler, BucketedDataset


def _pil(w: int, h: int, color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _jpeg_bytes(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    _pil(w, h).save(buf, format="JPEG", quality=60)
    return buf.getvalue()


def _strategy():
    return FixedBuckets(load_preset("sdxl"))


def _collate(samples):
    out = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        out[key] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


@pytest.fixture
def pil_hf_dataset() -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {
            "image": [
                _pil(1024, 1024),
                _pil(2048, 1024),
                _pil(1024, 2048),
                _pil(1024, 1024),
            ],
            "text": ["a", "b", "c", "d"],
        }
    )


class TestExtractDim:
    def test_pil(self):
        assert _extract_dim(_pil(800, 600)) == (800, 600)

    def test_bytes_dict(self):
        b = _jpeg_bytes(640, 480)
        assert _extract_dim({"bytes": b, "path": None}) == (640, 480)

    def test_path_dict(self, tmp_path):
        p = tmp_path / "img.png"
        _pil(320, 240).save(p)
        assert _extract_dim({"bytes": None, "path": str(p)}) == (320, 240)

    def test_numpy_hwc(self):
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
        assert _extract_dim(arr) == (640, 480)

    def test_numpy_chw(self):
        arr = np.zeros((3, 480, 640), dtype=np.uint8)
        assert _extract_dim(arr) == (640, 480)

    def test_numpy_grayscale(self):
        arr = np.zeros((480, 640), dtype=np.uint8)
        assert _extract_dim(arr) == (640, 480)

    def test_torch_tensor_chw(self):
        t = torch.zeros(3, 480, 640)
        assert _extract_dim(t) == (640, 480)

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError):
            _extract_dim(42)


class TestToPil:
    def test_pil_passthrough(self):
        img = _pil(100, 100, color=(10, 20, 30))
        assert _to_pil(img) is img

    def test_bytes_dict(self):
        b = _jpeg_bytes(64, 48)
        out = _to_pil({"bytes": b, "path": None})
        assert isinstance(out, Image.Image)
        assert out.size == (64, 48)

    def test_uint8_hwc(self):
        arr = np.full((48, 64, 3), 200, dtype=np.uint8)
        out = _to_pil(arr)
        assert out.size == (64, 48)
        assert out.mode == "RGB"

    def test_float_chw(self):
        arr = np.full((3, 48, 64), 0.5, dtype=np.float32)
        out = _to_pil(arr)
        assert out.size == (64, 48)

    def test_grayscale_promoted_to_rgb(self):
        arr = np.full((48, 64), 128, dtype=np.uint8)
        out = _to_pil(arr)
        assert out.mode == "RGB"

    def test_rgba(self):
        arr = np.full((48, 64, 4), 200, dtype=np.uint8)
        out = _to_pil(arr)
        assert out.mode == "RGB"


class TestArrayToPilEdgeCases:
    def test_rejects_weird_shape(self):
        arr = np.zeros((2, 2, 2, 2), dtype=np.uint8)
        with pytest.raises(ValueError):
            _array_to_pil(arr)


class TestHFSourceValidation:
    def test_iterable_dataset_rejected(self, pil_hf_dataset):
        streaming = pil_hf_dataset.to_iterable_dataset()
        with pytest.raises(ValueError, match="map-style"):
            _HFSource(streaming, image_column="image")

    def test_missing_image_column(self, pil_hf_dataset):
        with pytest.raises(ValueError, match="image_column"):
            _HFSource(pil_hf_dataset, image_column="missing")

    def test_missing_caption_column(self, pil_hf_dataset):
        with pytest.raises(ValueError, match="caption_column"):
            _HFSource(pil_hf_dataset, image_column="image", caption_column="not_here")


class TestFromHFPIL:
    def test_basic_construction(self, pil_hf_dataset):
        ds = BucketedDataset.from_hf(pil_hf_dataset, _strategy())
        assert len(ds) == len(pil_hf_dataset)
        assert ds.bucket_indices.shape == (len(pil_hf_dataset),)

    def test_sample_shape_matches_bucket(self, pil_hf_dataset):
        ds = BucketedDataset.from_hf(pil_hf_dataset, _strategy())
        for i in range(len(ds)):
            sample = ds[i]
            bucket = sample["bucket"]
            assert sample["image"].shape == (3, bucket.height, bucket.width)

    def test_caption_propagates(self, pil_hf_dataset):
        ds = BucketedDataset.from_hf(pil_hf_dataset, _strategy(), caption_column="text")
        assert ds[0]["caption"] == "a"
        assert ds[3]["caption"] == "d"

    def test_no_caption_omitted(self, pil_hf_dataset):
        ds = BucketedDataset.from_hf(pil_hf_dataset, _strategy())
        assert "caption" not in ds[0]

    def test_identifier_is_hf_style(self, pil_hf_dataset):
        ds = BucketedDataset.from_hf(pil_hf_dataset, _strategy())
        assert ds[0]["path"].startswith("hf:")

    def test_with_sampler(self, pil_hf_dataset):
        ds = BucketedDataset.from_hf(pil_hf_dataset, _strategy())
        sampler = BucketBatchSampler(ds, batch_size=2, shuffle=False, drop_last=False)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=_collate)
        for batch in loader:
            images = batch["image"]
            assert all(t.shape == images[0].shape for t in images)


class TestFromHFNumpyTensor:
    def test_numpy_column(self):
        hf = datasets.Dataset.from_dict(
            {
                "pixels": [
                    np.full((1024, 1024, 3), 128, dtype=np.uint8),
                    np.full((2048, 1024, 3), 128, dtype=np.uint8),
                ]
            }
        ).with_format("numpy")
        ds = BucketedDataset.from_hf(hf, _strategy(), image_column="pixels")
        # 1024x1024 (square) vs 2048x1024 (landscape) -> different buckets
        assert len(set(ds.bucket_indices.tolist())) == 2

    def test_torch_tensor_column(self):
        hf = datasets.Dataset.from_dict(
            {
                "pixels": [
                    np.full((3, 1024, 1024), 128, dtype=np.uint8),  # CHW
                    np.full((3, 1024, 2048), 128, dtype=np.uint8),
                ]
            }
        ).with_format("torch")
        ds = BucketedDataset.from_hf(hf, _strategy(), image_column="pixels")
        assert len(ds) == 2
        sample = ds[1]
        assert sample["image"].shape[1:] == (sample["bucket"].height, sample["bucket"].width)


class TestRoundTripWithSampler:
    def test_every_index_seen_once_when_not_dropping(self, pil_hf_dataset):
        ds = BucketedDataset.from_hf(pil_hf_dataset, _strategy())
        sampler = BucketBatchSampler(ds, batch_size=2, shuffle=False, drop_last=False)
        seen = []
        for batch in sampler:
            seen.extend(batch)
        assert sorted(seen) == list(range(len(ds)))
