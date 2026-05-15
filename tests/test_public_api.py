"""Smoke tests for the top-level public API surface."""

from __future__ import annotations

import bucketsampler


def test_version_string():
    assert isinstance(bucketsampler.__version__, str)
    assert bucketsampler.__version__.count(".") >= 1


def test_top_level_exports():
    # Public surface declared in CLAUDE.md
    expected = {
        "Bucket",
        "BucketSet",
        "FixedBuckets",
        "Strategy",
        "load_preset",
        "list_presets",
        "best_bucket",
        "assign_many",
    }
    missing = expected - set(bucketsampler.__all__)
    assert not missing, f"missing from public API: {missing}"


def test_quickstart_example():
    strategy = bucketsampler.FixedBuckets(bucketsampler.load_preset("sdxl"))
    bucket = strategy.assign(width=1280, height=720)
    assert isinstance(bucket, bucketsampler.Bucket)
    # 1280x720 is ~16:9, closest SDXL bucket should be wide
    assert bucket.aspect_ratio > 1.0


def test_unknown_attribute_raises():
    import pytest

    with pytest.raises(AttributeError, match="no attribute 'NonExistent'"):
        bucketsampler.NonExistent  # noqa: B018
