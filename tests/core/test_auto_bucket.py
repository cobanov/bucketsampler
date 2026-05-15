"""Tests for bucketsampler.core.auto_bucket."""

from __future__ import annotations

import numpy as np
import pytest

from bucketsampler import (
    AutoBuckets,
    Bucket,
    bucket_set_to_toml,
    generate_buckets,
    load_from_toml,
    load_preset,
)
from bucketsampler.core.stats import crop_loss_summary


def _three_cluster_dims(seed: int = 0, n_each: int = 200) -> np.ndarray:
    """Synthetic dataset with three well-separated AR clusters."""
    rng = np.random.default_rng(seed)
    log_ars = np.concatenate(
        [
            rng.normal(0.0, 0.05, n_each),
            rng.normal(0.7, 0.05, n_each),
            rng.normal(-0.7, 0.05, n_each),
        ]
    )
    h = rng.integers(800, 1200, size=log_ars.size)
    w = np.clip((h * np.exp(log_ars)).astype(int), 100, None)
    return np.stack([w, h], axis=1)


class TestGenerateBuckets:
    def test_returns_requested_count_when_separable(self):
        dims = _three_cluster_dims()
        result = generate_buckets(dims, num_buckets=3, target=1024, seed=0)
        assert len(result.bucket_set) == 3
        assert result.requested_k == 3

    def test_cluster_sizes_sum_to_dataset_size(self):
        dims = _three_cluster_dims()
        result = generate_buckets(dims, num_buckets=3, target=1024, seed=0)
        assert sum(result.cluster_sizes) == dims.shape[0]

    def test_bucket_set_dims_are_multiples_of_vae_factor(self):
        dims = _three_cluster_dims()
        result = generate_buckets(dims, num_buckets=4, target=1024, vae_factor=64, seed=0)
        for b in result.bucket_set:
            assert b.width % 64 == 0
            assert b.height % 64 == 0

    def test_smaller_vae_factor(self):
        dims = _three_cluster_dims()
        result = generate_buckets(dims, num_buckets=3, target=512, vae_factor=8, seed=0)
        for b in result.bucket_set:
            assert b.width % 8 == 0
            assert b.height % 8 == 0

    def test_target_controls_pixel_budget(self):
        dims = _three_cluster_dims()
        small = generate_buckets(dims, num_buckets=3, target=512, seed=0)
        large = generate_buckets(dims, num_buckets=3, target=1024, seed=0)
        # Mean pixel count should scale with target^2
        small_px = np.mean([b.pixel_count for b in small.bucket_set])
        large_px = np.mean([b.pixel_count for b in large.bucket_set])
        assert large_px > small_px * 3  # roughly 4x, but allow rounding slack

    def test_deterministic_with_seed(self):
        dims = _three_cluster_dims()
        a = generate_buckets(dims, num_buckets=4, target=1024, seed=42)
        b = generate_buckets(dims, num_buckets=4, target=1024, seed=42)
        assert a.bucket_set == b.bucket_set
        assert a.cluster_centers == b.cluster_centers
        assert a.cluster_sizes == b.cluster_sizes

    def test_kmeans_converges(self):
        dims = _three_cluster_dims()
        result = generate_buckets(dims, num_buckets=3, target=1024, seed=0, max_iter=100)
        assert result.iterations < 100

    def test_single_bucket(self):
        dims = _three_cluster_dims()
        result = generate_buckets(dims, num_buckets=1, target=1024, seed=0)
        assert len(result.bucket_set) == 1
        assert result.iterations == 0  # k=1 short-circuit

    def test_more_buckets_than_unique_ars_collapses(self):
        # Two unique ARs, but k=5 requested
        dims = np.array([[1024, 1024]] * 50 + [[2048, 1024]] * 50)
        result = generate_buckets(dims, num_buckets=5, target=1024, seed=0)
        assert len(result.bucket_set) <= 2
        assert result.requested_k == 5

    def test_improves_on_preset_for_clustered_data(self):
        """The headline M4 promise: auto > preset when data is clustered."""
        dims = _three_cluster_dims(n_each=500)
        auto = generate_buckets(dims, num_buckets=6, target=1024, seed=0)
        preset_loss = crop_loss_summary(dims, load_preset("sdxl")).mean
        assert auto.crop_loss_mean < preset_loss

    def test_metadata_propagates(self):
        dims = _three_cluster_dims()
        result = generate_buckets(
            dims,
            num_buckets=3,
            target=1024,
            name="mytest",
            description="hello",
        )
        assert result.bucket_set.name == "mytest"
        assert result.bucket_set.description == "hello"

    def test_rejects_empty_dims(self):
        with pytest.raises(ValueError):
            generate_buckets(np.zeros((0, 2), np.int64), num_buckets=4, target=1024)

    def test_rejects_invalid_shape(self):
        with pytest.raises(ValueError):
            generate_buckets(np.array([1, 2, 3]), num_buckets=4, target=1024)

    def test_rejects_non_positive_dims(self):
        with pytest.raises(ValueError):
            generate_buckets(np.array([[0, 100]]), num_buckets=4, target=1024)

    def test_rejects_zero_num_buckets(self):
        dims = _three_cluster_dims()
        with pytest.raises(ValueError):
            generate_buckets(dims, num_buckets=0, target=1024)

    def test_rejects_zero_target(self):
        dims = _three_cluster_dims()
        with pytest.raises(ValueError):
            generate_buckets(dims, num_buckets=3, target=0)

    def test_rejects_zero_vae_factor(self):
        dims = _three_cluster_dims()
        with pytest.raises(ValueError):
            generate_buckets(dims, num_buckets=3, target=1024, vae_factor=0)


class TestAutoBuckets:
    def test_from_dims_assigns(self):
        dims = _three_cluster_dims()
        strat = AutoBuckets.from_dims(dims, num_buckets=3, target=1024, seed=0)
        # Pick something near the middle cluster (AR=1)
        assigned = strat.assign(1000, 1000)
        assert isinstance(assigned, Bucket)

    def test_implements_strategy_protocol(self):
        from bucketsampler import Strategy

        dims = _three_cluster_dims()
        strat = AutoBuckets.from_dims(dims, num_buckets=3, target=1024, seed=0)
        assert isinstance(strat, Strategy)

    def test_bucket_set_exposed(self):
        dims = _three_cluster_dims()
        strat = AutoBuckets.from_dims(dims, num_buckets=3, target=1024, seed=0)
        assert len(strat.bucket_set) == 3

    def test_assign_many_indices(self):
        dims = _three_cluster_dims(n_each=10)
        strat = AutoBuckets.from_dims(dims, num_buckets=3, target=1024, seed=0)
        idx = strat.assign_many_indices(dims)
        assert idx.shape == (dims.shape[0],)
        assert int(idx.min()) >= 0
        assert int(idx.max()) < len(strat.bucket_set)

    def test_result_attribute_carries_diagnostics(self):
        dims = _three_cluster_dims()
        strat = AutoBuckets.from_dims(dims, num_buckets=3, target=1024, seed=0)
        assert strat.result.crop_loss_mean >= 0
        assert len(strat.result.cluster_sizes) == len(strat.bucket_set)

    def test_repr(self):
        dims = _three_cluster_dims()
        strat = AutoBuckets.from_dims(dims, num_buckets=3, target=1024, seed=0)
        r = repr(strat)
        assert "AutoBuckets" in r
        assert "n_buckets=3" in r


class TestBucketSetToToml:
    def test_round_trip_via_load_from_toml(self, tmp_path):
        dims = _three_cluster_dims()
        result = generate_buckets(
            dims,
            num_buckets=4,
            target=1024,
            seed=0,
            name="rt",
            description="round-trip",
        )
        toml_str = bucket_set_to_toml(result.bucket_set)
        p = tmp_path / "rt.toml"
        p.write_text(toml_str)
        loaded = load_from_toml(p)
        assert loaded == result.bucket_set
        assert loaded.name == "rt"
        assert loaded.description == "round-trip"

    def test_no_name_or_description_still_round_trips(self, tmp_path):
        dims = _three_cluster_dims()
        result = generate_buckets(dims, num_buckets=3, target=1024, name="", description="")
        toml_str = bucket_set_to_toml(result.bucket_set)
        p = tmp_path / "anon.toml"
        p.write_text(toml_str)
        loaded = load_from_toml(p)
        assert loaded == result.bucket_set

    def test_escapes_quotes_in_description(self, tmp_path):
        dims = _three_cluster_dims()
        result = generate_buckets(
            dims,
            num_buckets=2,
            target=512,
            description='has "quotes" and \\ backslash',
        )
        toml_str = bucket_set_to_toml(result.bucket_set)
        p = tmp_path / "quoted.toml"
        p.write_text(toml_str)
        loaded = load_from_toml(p)
        assert loaded.description == 'has "quotes" and \\ backslash'
