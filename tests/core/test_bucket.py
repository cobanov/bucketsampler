"""Tests for bucketsampler.core.bucket."""

from __future__ import annotations

import dataclasses
import math

import pytest

from bucketsampler import Bucket, BucketSet
from bucketsampler.exceptions import (
    DuplicateBucketError,
    EmptyBucketSetError,
    InvalidBucketError,
)


class TestBucket:
    def test_basic_construction(self):
        b = Bucket(1024, 768)
        assert b.width == 1024
        assert b.height == 768

    def test_aspect_ratio_landscape(self):
        b = Bucket(1024, 512)
        assert b.aspect_ratio == 2.0

    def test_aspect_ratio_portrait(self):
        b = Bucket(512, 1024)
        assert b.aspect_ratio == 0.5

    def test_aspect_ratio_square(self):
        b = Bucket(512, 512)
        assert b.aspect_ratio == 1.0

    def test_log_aspect_ratio(self):
        b = Bucket(1024, 512)
        assert b.log_aspect_ratio == pytest.approx(math.log(2.0))

    def test_log_aspect_ratio_square_is_zero(self):
        b = Bucket(512, 512)
        assert b.log_aspect_ratio == 0.0

    def test_pixel_count(self):
        b = Bucket(1024, 768)
        assert b.pixel_count == 1024 * 768

    def test_is_multiple_of(self):
        b = Bucket(1024, 768)
        assert b.is_multiple_of(64) is True
        assert b.is_multiple_of(8) is True
        assert b.is_multiple_of(100) is False

    def test_is_multiple_of_only_one_axis_returns_false(self):
        # 1024 % 1024 == 0 but 768 % 1024 != 0
        b = Bucket(1024, 768)
        assert b.is_multiple_of(1024) is False

    def test_as_tuple(self):
        b = Bucket(1024, 768)
        assert b.as_tuple() == (1024, 768)

    def test_repr(self):
        b = Bucket(1024, 768)
        assert repr(b) == "Bucket(width=1024, height=768)"

    def test_str(self):
        b = Bucket(1024, 768)
        assert str(b) == "1024x768"

    def test_equality(self):
        assert Bucket(512, 512) == Bucket(512, 512)
        assert Bucket(512, 512) != Bucket(512, 256)

    def test_hashable(self):
        s = {Bucket(512, 512), Bucket(512, 512), Bucket(256, 256)}
        assert len(s) == 2

    def test_frozen(self):
        b = Bucket(512, 512)
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.width = 256  # type: ignore[misc]

    def test_zero_width_raises(self):
        with pytest.raises(InvalidBucketError):
            Bucket(0, 512)

    def test_zero_height_raises(self):
        with pytest.raises(InvalidBucketError):
            Bucket(512, 0)

    def test_negative_width_raises(self):
        with pytest.raises(InvalidBucketError):
            Bucket(-1, 512)

    def test_negative_height_raises(self):
        with pytest.raises(InvalidBucketError):
            Bucket(512, -1)

    def test_non_int_width_raises(self):
        with pytest.raises(InvalidBucketError):
            Bucket(512.0, 512)  # type: ignore[arg-type]

    def test_non_int_height_raises(self):
        with pytest.raises(InvalidBucketError):
            Bucket(512, "x")  # type: ignore[arg-type]

    def test_bool_width_raises(self):
        with pytest.raises(InvalidBucketError):
            Bucket(True, 512)  # type: ignore[arg-type]

    def test_bool_height_raises(self):
        with pytest.raises(InvalidBucketError):
            Bucket(512, False)  # type: ignore[arg-type]

    def test_error_message_includes_dims(self):
        with pytest.raises(InvalidBucketError) as excinfo:
            Bucket(0, 512)
        assert "0" in str(excinfo.value)
        assert "512" in str(excinfo.value)


class TestBucketSet:
    def test_from_dims_basic(self):
        bs = BucketSet.from_dims([(512, 512), (768, 512), (512, 768)])
        assert len(bs) == 3
        assert bs[0] == Bucket(512, 512)
        assert bs[1] == Bucket(768, 512)
        assert bs[2] == Bucket(512, 768)

    def test_iteration(self):
        bs = BucketSet.from_dims([(512, 512), (768, 512)])
        out = list(bs)
        assert out == [Bucket(512, 512), Bucket(768, 512)]

    def test_contains(self):
        bs = BucketSet.from_dims([(512, 512)])
        assert Bucket(512, 512) in bs
        assert Bucket(768, 512) not in bs

    def test_index_of(self):
        bs = BucketSet.from_dims([(512, 512), (768, 512)])
        assert bs.index_of(Bucket(768, 512)) == 1

    def test_index_of_missing_raises(self):
        bs = BucketSet.from_dims([(512, 512)])
        with pytest.raises(ValueError):
            bs.index_of(Bucket(1024, 1024))

    def test_metadata_fields(self):
        bs = BucketSet.from_dims(
            [(512, 512)],
            name="test",
            description="a test set",
            vae_factor=8,
        )
        assert bs.name == "test"
        assert bs.description == "a test set"
        assert bs.vae_factor == 8

    def test_empty_raises(self):
        with pytest.raises(EmptyBucketSetError):
            BucketSet.from_dims([])

    def test_duplicates_raise(self):
        with pytest.raises(DuplicateBucketError) as excinfo:
            BucketSet.from_dims([(512, 512), (768, 512), (512, 512)])
        assert (512, 512) in excinfo.value.duplicates

    def test_invalid_bucket_propagates(self):
        with pytest.raises(InvalidBucketError):
            BucketSet.from_dims([(0, 512)])

    def test_sorted_by_aspect_ratio(self):
        bs = BucketSet.from_dims([(1024, 512), (512, 512), (512, 1024)])
        sorted_buckets = bs.sorted_by_aspect_ratio()
        ars = [b.aspect_ratio for b in sorted_buckets]
        assert ars == sorted(ars)

    def test_pixel_budget_odd(self):
        bs = BucketSet.from_dims([(512, 512), (1024, 1024), (256, 256)])
        # sorted pixel counts: 65536, 262144, 1048576 -> median 262144 == 512*512
        assert bs.pixel_budget() == 512 * 512

    def test_pixel_budget_even(self):
        bs = BucketSet.from_dims([(512, 512), (1024, 1024)])
        # avg of two
        assert bs.pixel_budget() == (512 * 512 + 1024 * 1024) // 2

    def test_all_multiples_of_true(self):
        bs = BucketSet.from_dims([(512, 512), (768, 1024)])
        assert bs.all_multiples_of(64) is True

    def test_all_multiples_of_false(self):
        bs = BucketSet.from_dims([(512, 512), (513, 513)])
        assert bs.all_multiples_of(64) is False

    def test_vae_factor_positive(self):
        with pytest.raises(ValueError):
            BucketSet(buckets=(Bucket(512, 512),), vae_factor=0)

    def test_direct_construction_with_list_coerces_to_tuple(self):
        bs = BucketSet(buckets=[Bucket(512, 512)])  # type: ignore[arg-type]
        assert isinstance(bs.buckets, tuple)

    def test_frozen(self):
        bs = BucketSet.from_dims([(512, 512)])
        with pytest.raises(dataclasses.FrozenInstanceError):
            bs.buckets = ()  # type: ignore[misc]

    def test_equality(self):
        a = BucketSet.from_dims([(512, 512), (768, 512)])
        b = BucketSet.from_dims([(512, 512), (768, 512)])
        assert a == b

    def test_hashable(self):
        a = BucketSet.from_dims([(512, 512)])
        b = BucketSet.from_dims([(512, 512)])
        assert hash(a) == hash(b)
