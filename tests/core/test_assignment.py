"""Tests for bucketsampler.core.assignment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bucketsampler import (
    Bucket,
    BucketSet,
    assign_many,
    assign_many_indices,
    best_bucket,
    crop_loss,
    log_ar_distance,
    resize_to_bucket_dims,
)


@pytest.fixture
def sdxl_like_set() -> BucketSet:
    """A small SDXL-shaped bucket set: square, wide, ultrawide, tall, ultratall."""
    return BucketSet.from_dims(
        [
            (1024, 1024),
            (1152, 896),
            (896, 1152),
            (1344, 768),
            (768, 1344),
            (1536, 640),
            (640, 1536),
        ]
    )


class TestLogArDistance:
    def test_identical_ar_is_zero(self):
        # Native 2:1 image, 2:1 bucket -> distance 0
        b = Bucket(1024, 512)
        assert log_ar_distance(2048, 1024, b) == 0.0

    def test_symmetric(self):
        b = Bucket(1024, 512)
        d1 = log_ar_distance(1024, 1024, b)
        d2 = log_ar_distance(512, 1024, b)
        # not necessarily equal, but both positive
        assert d1 > 0
        assert d2 > 0

    def test_zero_dim_raises(self):
        b = Bucket(512, 512)
        with pytest.raises(ValueError):
            log_ar_distance(0, 512, b)
        with pytest.raises(ValueError):
            log_ar_distance(512, 0, b)


class TestBestBucket:
    def test_square_image_picks_square_bucket(self, sdxl_like_set):
        b = best_bucket(800, 800, sdxl_like_set)
        assert b == Bucket(1024, 1024)

    def test_wide_image_picks_wide_bucket(self, sdxl_like_set):
        b = best_bucket(2000, 1000, sdxl_like_set)
        # 2:1 image. Bucket 1536x640 = 2.4, 1344x768 = 1.75. log distances:
        # |log(2) - log(2.4)| = 0.182, |log(2) - log(1.75)| = 0.134
        # so 1344x768 wins
        assert b == Bucket(1344, 768)

    def test_tall_image_picks_tall_bucket(self, sdxl_like_set):
        b = best_bucket(500, 1000, sdxl_like_set)
        assert b.height > b.width

    def test_ultrawide_picks_ultrawide(self, sdxl_like_set):
        b = best_bucket(2400, 1000, sdxl_like_set)
        assert b == Bucket(1536, 640)

    def test_ultratall_picks_ultratall(self, sdxl_like_set):
        b = best_bucket(1000, 2400, sdxl_like_set)
        assert b == Bucket(640, 1536)

    def test_exact_ar_match(self, sdxl_like_set):
        # Image with exact 1152/896 AR
        b = best_bucket(1152, 896, sdxl_like_set)
        assert b == Bucket(1152, 896)

    def test_scale_invariant(self, sdxl_like_set):
        # Same AR at different scales gives same bucket
        b1 = best_bucket(2048, 1024, sdxl_like_set)
        b2 = best_bucket(500, 250, sdxl_like_set)
        assert b1 == b2

    def test_tie_resolves_to_earlier_bucket(self):
        # Two buckets equidistant in log space; first one wins
        bs = BucketSet.from_dims([(1024, 512), (512, 1024)])
        # Square image is equidistant (log(1) = 0, both at log(2)) abs equal
        b = best_bucket(500, 500, bs)
        assert b == Bucket(1024, 512)

    def test_single_bucket_set(self):
        bs = BucketSet.from_dims([(512, 512)])
        assert best_bucket(1000, 200, bs) == Bucket(512, 512)

    def test_zero_dim_raises(self, sdxl_like_set):
        with pytest.raises(ValueError):
            best_bucket(0, 100, sdxl_like_set)
        with pytest.raises(ValueError):
            best_bucket(100, 0, sdxl_like_set)


class TestAssignManyIndices:
    def test_basic_shapes(self, sdxl_like_set):
        dims = [(800, 800), (2000, 1000), (500, 1000)]
        idx = assign_many_indices(dims, sdxl_like_set)
        assert idx.shape == (3,)
        assert idx.dtype == np.int64

    def test_matches_best_bucket(self, sdxl_like_set):
        dims = [(800, 800), (2000, 1000), (500, 1000), (2400, 1000), (1000, 2400)]
        idx = assign_many_indices(dims, sdxl_like_set)
        for i, (w, h) in enumerate(dims):
            assert sdxl_like_set[int(idx[i])] == best_bucket(w, h, sdxl_like_set)

    def test_empty_input(self, sdxl_like_set):
        idx = assign_many_indices([], sdxl_like_set)
        assert idx.shape == (0,)
        assert idx.dtype == np.int64

    def test_numpy_input(self, sdxl_like_set):
        arr = np.array([[800, 800], [2000, 1000]], dtype=np.int64)
        idx = assign_many_indices(arr, sdxl_like_set)
        assert idx.shape == (2,)

    def test_wrong_shape_raises(self, sdxl_like_set):
        with pytest.raises(ValueError):
            assign_many_indices(np.array([1, 2, 3]), sdxl_like_set)
        with pytest.raises(ValueError):
            assign_many_indices(np.array([[1, 2, 3]]), sdxl_like_set)

    def test_zero_dim_raises(self, sdxl_like_set):
        with pytest.raises(ValueError):
            assign_many_indices([(0, 100), (100, 100)], sdxl_like_set)

    def test_large_dataset(self, sdxl_like_set):
        rng = np.random.default_rng(seed=42)
        dims = rng.integers(low=100, high=4000, size=(10_000, 2)).tolist()
        idx = assign_many_indices(dims, sdxl_like_set)
        assert idx.shape == (10_000,)
        assert idx.min() >= 0
        assert idx.max() < len(sdxl_like_set)


class TestAssignMany:
    def test_returns_bucket_instances(self, sdxl_like_set):
        out = assign_many([(800, 800), (2000, 1000)], sdxl_like_set)
        assert all(isinstance(b, Bucket) for b in out)
        assert len(out) == 2

    def test_matches_best_bucket(self, sdxl_like_set):
        dims = [(800, 800), (2000, 1000), (1000, 2400)]
        out = assign_many(dims, sdxl_like_set)
        for (w, h), bucket in zip(dims, out, strict=False):
            assert bucket == best_bucket(w, h, sdxl_like_set)

    def test_generator_input(self, sdxl_like_set):
        out = assign_many(((800, 800), (2000, 1000)), sdxl_like_set)
        assert len(out) == 2


class TestResizeToBucketDims:
    def test_no_crop_when_ar_matches(self):
        # Source 2048x1024 (AR 2:1), bucket 1024x512 (AR 2:1)
        b = Bucket(1024, 512)
        (rw, rh), (cx, cy) = resize_to_bucket_dims(2048, 1024, b)
        assert (rw, rh) == (1024, 512)
        assert (cx, cy) == (0, 0)

    def test_wide_image_into_square_bucket(self):
        # Source wider than bucket: scale by height, then crop width
        b = Bucket(512, 512)
        (rw, rh), (cx, cy) = resize_to_bucket_dims(2048, 1024, b)
        assert rh == 512
        assert rw > 512
        assert cy == 0
        assert cx > 0

    def test_tall_image_into_square_bucket(self):
        b = Bucket(512, 512)
        (rw, rh), (cx, cy) = resize_to_bucket_dims(1024, 2048, b)
        assert rw == 512
        assert rh > 512
        assert cx == 0
        assert cy > 0

    def test_covers_bucket_on_both_axes(self):
        # Even with rounding, the resized image must cover the bucket
        b = Bucket(1024, 768)
        for w, h in [(1000, 1000), (1234, 567), (3001, 1001), (501, 2001)]:
            (rw, rh), _ = resize_to_bucket_dims(w, h, b)
            assert rw >= 1024
            assert rh >= 768

    def test_small_image_still_works(self):
        # Source smaller than bucket: upscale
        b = Bucket(1024, 1024)
        (rw, rh), _ = resize_to_bucket_dims(100, 100, b)
        assert rw >= 1024
        assert rh >= 1024

    def test_zero_dim_raises(self):
        b = Bucket(512, 512)
        with pytest.raises(ValueError):
            resize_to_bucket_dims(0, 100, b)
        with pytest.raises(ValueError):
            resize_to_bucket_dims(100, 0, b)


class TestCropLoss:
    def test_zero_when_ar_matches(self):
        b = Bucket(1024, 512)
        assert crop_loss(2048, 1024, b) == pytest.approx(0.0)

    def test_positive_when_ar_differs(self):
        b = Bucket(512, 512)
        loss = crop_loss(2048, 1024, b)
        assert 0.0 < loss < 1.0

    def test_symmetric_in_ar_inversion(self):
        b = Bucket(512, 512)
        wide = crop_loss(2000, 1000, b)
        tall = crop_loss(1000, 2000, b)
        assert wide == pytest.approx(tall)

    def test_worst_case_bounded(self):
        b = Bucket(512, 512)
        # very wide -> high loss but still < 1
        loss = crop_loss(10000, 100, b)
        assert 0.9 < loss < 1.0

    def test_zero_dim_raises(self):
        b = Bucket(512, 512)
        with pytest.raises(ValueError):
            crop_loss(0, 100, b)


class TestDeterminism:
    """Same input -> same output, every time. Critical for reproducibility."""

    def test_best_bucket_deterministic(self, sdxl_like_set):
        runs = [best_bucket(1234, 567, sdxl_like_set) for _ in range(10)]
        assert len(set(runs)) == 1

    def test_assign_many_deterministic(self, sdxl_like_set):
        dims = [(1234, 567), (800, 1200), (1000, 1000)]
        runs = [tuple(assign_many_indices(dims, sdxl_like_set).tolist()) for _ in range(10)]
        assert len(set(runs)) == 1


class TestMathSanity:
    """Spot-check the algorithm against hand-computed values."""

    def test_known_assignment(self):
        bs = BucketSet.from_dims([(2, 1), (1, 1), (1, 2)])
        # AR = 1.5, log(1.5) = 0.405. Distances:
        # log(2) = 0.693 -> 0.288
        # log(1) = 0.0   -> 0.405
        # log(0.5) = -0.693 -> 1.098
        # So 2:1 wins.
        assert best_bucket(3, 2, bs) == Bucket(2, 1)

    def test_log_distance_matches_formula(self):
        b = Bucket(1024, 512)
        d = log_ar_distance(1000, 1000, b)
        expected = abs(math.log(1.0) - math.log(2.0))
        assert d == pytest.approx(expected)
