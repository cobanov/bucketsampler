"""Tests for bucketsampler.core.strategies."""

from __future__ import annotations

import numpy as np
import pytest

from bucketsampler import Bucket, BucketSet, FixedBuckets, Strategy


@pytest.fixture
def bucket_set() -> BucketSet:
    return BucketSet.from_dims([(512, 512), (768, 512), (512, 768)])


class TestFixedBuckets:
    def test_assign_single(self, bucket_set):
        s = FixedBuckets(bucket_set)
        assert s.assign(500, 500) == Bucket(512, 512)
        assert s.assign(800, 500) == Bucket(768, 512)
        assert s.assign(500, 800) == Bucket(512, 768)

    def test_assign_many_indices(self, bucket_set):
        s = FixedBuckets(bucket_set)
        idx = s.assign_many_indices([(500, 500), (800, 500), (500, 800)])
        assert idx.tolist() == [0, 1, 2]
        assert idx.dtype == np.int64

    def test_bucket_set_property(self, bucket_set):
        s = FixedBuckets(bucket_set)
        assert s.bucket_set is bucket_set

    def test_repr(self, bucket_set):
        s = FixedBuckets(bucket_set)
        assert "n_buckets=3" in repr(s)

    def test_protocol_runtime_check(self, bucket_set):
        s = FixedBuckets(bucket_set)
        assert isinstance(s, Strategy)

    def test_invalid_dim_propagates(self, bucket_set):
        s = FixedBuckets(bucket_set)
        with pytest.raises(ValueError):
            s.assign(0, 100)
