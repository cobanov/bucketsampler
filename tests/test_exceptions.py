"""Tests for bucketsampler.exceptions."""

from __future__ import annotations

from pathlib import Path

import pytest

from bucketsampler.exceptions import (
    BucketSamplerError,
    DuplicateBucketError,
    EmptyBucketSetError,
    ImageTooSmallError,
    InvalidBucketError,
    InvalidPresetError,
    PresetNotFoundError,
)


class TestHierarchy:
    @pytest.mark.parametrize(
        "exc",
        [
            InvalidBucketError(width=0, height=10, reason="bad"),
            EmptyBucketSetError(),
            DuplicateBucketError(duplicates=[(1, 1)]),
            ImageTooSmallError(path=None, actual=(1, 1), required=(2, 2)),
            PresetNotFoundError(name="x", available=[]),
            InvalidPresetError(source="x", reason="bad"),
        ],
    )
    def test_inherits_from_base(self, exc):
        assert isinstance(exc, BucketSamplerError)


class TestInvalidBucketError:
    def test_fields(self):
        e = InvalidBucketError(width=0, height=512, reason="non-positive")
        assert e.width == 0
        assert e.height == 512
        assert e.reason == "non-positive"
        assert "0" in str(e)
        assert "512" in str(e)
        assert "non-positive" in str(e)

    def test_is_value_error(self):
        e = InvalidBucketError(width=0, height=0, reason="x")
        assert isinstance(e, ValueError)


class TestDuplicateBucketError:
    def test_sorts_and_dedups(self):
        e = DuplicateBucketError(duplicates=[(1, 2), (1, 2), (3, 4)])
        assert e.duplicates == [(1, 2), (3, 4)]

    def test_message_has_pairs(self):
        e = DuplicateBucketError(duplicates=[(1, 2)])
        assert "1x2" in str(e)


class TestImageTooSmallError:
    def test_path_str(self):
        e = ImageTooSmallError(
            path="img.png",
            actual=(100, 100),
            required=(512, 512),
            suggestion="Filter dataset",
        )
        assert e.path == Path("img.png")
        assert e.actual == (100, 100)
        assert e.required == (512, 512)
        assert "img.png" in str(e)
        assert "Filter dataset" in str(e)

    def test_no_path(self):
        e = ImageTooSmallError(path=None, actual=(1, 1), required=(2, 2))
        assert e.path is None
        s = str(e)
        assert "at" not in s.split("image")[1].split("is")[0]


class TestPresetNotFoundError:
    def test_basic(self):
        e = PresetNotFoundError(name="missing", available=["sdxl", "sd15"])
        assert e.name == "missing"
        assert e.available == ["sd15", "sdxl"]
        s = str(e)
        assert "missing" in s
        assert "sdxl" in s

    def test_is_key_error(self):
        e = PresetNotFoundError(name="x", available=[])
        assert isinstance(e, KeyError)

    def test_no_available_message(self):
        e = PresetNotFoundError(name="x", available=[])
        assert "(none)" in str(e)


class TestEmptyBucketSetError:
    def test_message(self):
        e = EmptyBucketSetError()
        assert "at least one" in str(e)


class TestInvalidPresetError:
    def test_fields(self):
        e = InvalidPresetError(source="bad.toml", reason="missing field")
        assert e.source == "bad.toml"
        assert e.reason == "missing field"
        assert "bad.toml" in str(e)
        assert "missing field" in str(e)
