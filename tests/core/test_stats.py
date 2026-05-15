"""Tests for bucketsampler.core.stats."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bucketsampler import (
    Bucket,
    BucketSet,
    aspect_ratio_summary,
    bucket_distribution,
    crop_loss_summary,
    underutilized_buckets,
)


@pytest.fixture
def bucket_set() -> BucketSet:
    return BucketSet.from_dims([(512, 512), (768, 512), (512, 768)])


class TestAspectRatioSummary:
    def test_empty(self):
        s = aspect_ratio_summary([])
        assert s.count == 0
        assert math.isnan(s.mean_log_ar)
        assert math.isnan(s.median_log_ar)
        assert math.isnan(s.std_log_ar)
        assert math.isnan(s.min_ar)
        assert math.isnan(s.max_ar)

    def test_single_image(self):
        s = aspect_ratio_summary([(1024, 512)])
        assert s.count == 1
        assert s.mean_log_ar == pytest.approx(math.log(2.0))
        assert s.min_ar == pytest.approx(2.0)
        assert s.max_ar == pytest.approx(2.0)
        assert s.std_log_ar == pytest.approx(0.0)

    def test_basic_stats(self):
        dims = [(512, 512), (1024, 512), (512, 1024)]
        s = aspect_ratio_summary(dims)
        assert s.count == 3
        assert s.min_ar == pytest.approx(0.5)
        assert s.max_ar == pytest.approx(2.0)
        assert s.mean_log_ar == pytest.approx(0.0)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            aspect_ratio_summary(np.array([1, 2, 3]))

    def test_negative_dim_raises(self):
        with pytest.raises(ValueError):
            aspect_ratio_summary([(0, 100)])


class TestBucketDistribution:
    def test_basic_counting(self, bucket_set):
        dims = [(500, 500), (500, 500), (800, 500)]
        counts = bucket_distribution(dims, bucket_set)
        assert counts[Bucket(512, 512)] == 2
        assert counts[Bucket(768, 512)] == 1
        assert counts[Bucket(512, 768)] == 0

    def test_empty(self, bucket_set):
        counts = bucket_distribution([], bucket_set)
        assert all(c == 0 for c in counts.values())
        assert set(counts.keys()) == set(bucket_set)

    def test_zero_buckets_kept_in_output(self, bucket_set):
        # Only square images: tall/wide buckets stay at zero
        counts = bucket_distribution([(500, 500)] * 100, bucket_set)
        assert counts[Bucket(512, 512)] == 100
        assert counts[Bucket(768, 512)] == 0
        assert counts[Bucket(512, 768)] == 0


class TestUnderutilizedBuckets:
    def test_finds_empty(self, bucket_set):
        under = underutilized_buckets([(500, 500)] * 5, bucket_set, min_count=1)
        assert Bucket(768, 512) in under
        assert Bucket(512, 768) in under
        assert Bucket(512, 512) not in under

    def test_threshold_inclusive(self, bucket_set):
        # min_count=5 means buckets with <5 are flagged
        under = underutilized_buckets([(500, 500)] * 5, bucket_set, min_count=5)
        assert Bucket(512, 512) not in under

    def test_preserves_order(self, bucket_set):
        # Empty dataset -> all underutilized, in bucket_set order
        under = underutilized_buckets([], bucket_set, min_count=1)
        assert under == list(bucket_set)

    def test_negative_min_count_raises(self, bucket_set):
        with pytest.raises(ValueError):
            underutilized_buckets([], bucket_set, min_count=-1)


class TestCropLossSummary:
    def test_empty(self, bucket_set):
        s = crop_loss_summary([], bucket_set)
        assert s.count == 0
        assert math.isnan(s.mean)
        assert math.isnan(s.median)
        assert math.isnan(s.p95)
        assert math.isnan(s.max)

    def test_no_loss_when_ar_matches(self, bucket_set):
        # 512x512 image -> 512x512 bucket, no crop
        s = crop_loss_summary([(512, 512)] * 10, bucket_set)
        assert s.mean == pytest.approx(0.0)
        assert s.max == pytest.approx(0.0)

    def test_positive_loss_for_mismatched(self, bucket_set):
        # 1000x1000 image -> 512x512 bucket: AR matches, no loss
        # 2000x500 (4:1) image -> closest bucket 768x512 (1.5) is far
        s = crop_loss_summary([(2000, 500)], bucket_set)
        assert s.mean > 0.5

    def test_bounded(self, bucket_set):
        # All losses must be in [0, 1)
        dims = [(2000, 500), (500, 2000), (1000, 1000), (100, 10000)]
        s = crop_loss_summary(dims, bucket_set)
        assert 0.0 <= s.mean < 1.0
        assert 0.0 <= s.median < 1.0
        assert 0.0 <= s.p95 < 1.0
        assert 0.0 <= s.max < 1.0
