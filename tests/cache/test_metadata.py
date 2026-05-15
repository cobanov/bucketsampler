"""Tests for the metadata cache."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from bucketsampler import FixedBuckets, MetadataCache, build_metadata_cache, load_preset
from bucketsampler.cache.metadata import MetadataRow, dims_from_cache
from bucketsampler.cli import app
from bucketsampler.torch import BucketedDataset

runner = CliRunner()


def _write_image(p: Path, w: int, h: int) -> None:
    Image.new("RGB", (w, h)).save(p, format="JPEG", quality=40)


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    for i, (w, h) in enumerate([(1024, 1024), (1024, 2048), (2048, 1024), (1024, 1024)]):
        _write_image(tmp_path / f"img_{i}.jpg", w, h)
    return tmp_path


class TestBuildMetadataCache:
    def test_basic(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        cache = build_metadata_cache(paths)
        assert len(cache) == 4
        for p in paths:
            row = cache.get(p)
            assert row is not None
            assert row.width > 0 and row.height > 0
            assert row.mtime_ns > 0
            assert row.size_bytes > 0

    def test_serial(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        c_serial = build_metadata_cache(paths, num_workers=1)
        c_parallel = build_metadata_cache(paths, num_workers=8)
        assert {r.path for r in c_serial} == {r.path for r in c_parallel}

    def test_skips_broken_files(self, tmp_path: Path):
        _write_image(tmp_path / "good.jpg", 100, 100)
        (tmp_path / "bad.jpg").write_bytes(b"\xff\xd8nope")
        paths = sorted(tmp_path.glob("*.jpg"))
        cache = build_metadata_cache(paths)
        assert len(cache) == 1
        names = {Path(r.path).name for r in cache}
        assert names == {"good.jpg"}

    def test_refresh_reuses_unchanged(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        first = build_metadata_cache(paths)
        # Re-build with existing cache; mtimes match -> all reused
        second = build_metadata_cache(paths, existing=first)
        assert {r.path for r in first} == {r.path for r in second}
        # Rows should be identical objects in content
        for p in paths:
            assert first.get(p) == second.get(p)

    def test_refresh_detects_stale(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        first = build_metadata_cache(paths)
        # Mutate one file's dims and bump its mtime
        target = paths[0]
        _write_image(target, 800, 600)
        new_mtime = first.get(target).mtime_ns + 1_000_000_000
        os.utime(target, ns=(new_mtime, new_mtime))
        second = build_metadata_cache(paths, existing=first)
        row = second.get(target)
        assert row is not None
        assert (row.width, row.height) == (800, 600)

    def test_drops_removed_paths(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        first = build_metadata_cache(paths)
        # Caller scans only a subset next time
        kept = paths[:2]
        second = build_metadata_cache(kept, existing=first)
        assert len(second) == 2
        assert {r.path for r in second} == {str(p) for p in kept}


class TestMetadataCachePersistence:
    def test_round_trip(self, image_dir: Path, tmp_path: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        cache = build_metadata_cache(paths)
        out = tmp_path / "cache.parquet"
        cache.save(out)
        loaded = MetadataCache.load(out)
        assert len(loaded) == len(cache)
        for r in cache:
            assert loaded.get(r.path) == r

    def test_save_creates_parent_dirs(self, image_dir: Path, tmp_path: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        cache = build_metadata_cache(paths)
        out = tmp_path / "nested" / "sub" / "cache.parquet"
        cache.save(out)
        assert out.exists()

    def test_load_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            MetadataCache.load(tmp_path / "absent.parquet")

    def test_load_bad_schema(self, tmp_path: Path):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        bad = tmp_path / "bad.parquet"
        pq.write_table(pa.table({"foo": [1, 2, 3]}), bad)
        with pytest.raises(ValueError, match="missing column"):
            MetadataCache.load(bad)


class TestMetadataCacheBehavior:
    def test_is_stale_missing(self, tmp_path: Path):
        c = MetadataCache()
        assert c.is_stale(tmp_path / "nope.jpg") is True

    def test_is_stale_after_touch(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        cache = build_metadata_cache(paths)
        target = paths[0]
        assert cache.is_stale(target) is False
        new_mtime = cache.get(target).mtime_ns + 1_000_000_000
        os.utime(target, ns=(new_mtime, new_mtime))
        assert cache.is_stale(target) is True

    def test_upsert_overwrites(self):
        c = MetadataCache()
        c.upsert(MetadataRow("a.jpg", 100, 100, 1, 1))
        c.upsert(MetadataRow("a.jpg", 200, 200, 2, 2))
        assert c.get("a.jpg").width == 200

    def test_contains(self):
        c = MetadataCache([MetadataRow("a.jpg", 1, 1, 1, 1)])
        assert "a.jpg" in c
        assert "b.jpg" not in c

    def test_iter(self):
        c = MetadataCache([MetadataRow("a.jpg", 1, 1, 1, 1)])
        rows = list(c)
        assert rows[0].path == "a.jpg"


class TestDimsFromCache:
    def test_returns_dims(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        cache = build_metadata_cache(paths)
        dims = dims_from_cache(paths, cache)
        assert dims.shape == (4, 2)
        assert dims.dtype == np.int64

    def test_raises_on_miss(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        cache = build_metadata_cache(paths[:2])
        with pytest.raises(KeyError):
            dims_from_cache(paths, cache)


class TestBucketedDatasetCacheIntegration:
    def test_dataset_uses_cache_dims(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        cache = build_metadata_cache(paths)
        ds_cached = BucketedDataset(
            paths=paths,
            strategy=FixedBuckets(load_preset("sdxl")),
            metadata_cache=cache,
        )
        ds_fresh = BucketedDataset(
            paths=paths,
            strategy=FixedBuckets(load_preset("sdxl")),
        )
        assert ds_cached.bucket_indices.tolist() == ds_fresh.bucket_indices.tolist()

    def test_cache_with_partial_misses_falls_back(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))
        partial = build_metadata_cache(paths[:2])
        ds = BucketedDataset(
            paths=paths,
            strategy=FixedBuckets(load_preset("sdxl")),
            metadata_cache=partial,
        )
        assert ds.bucket_indices.shape == (4,)


class TestBuildCacheCLI:
    def test_basic(self, image_dir: Path, tmp_path: Path):
        out = tmp_path / "cache.parquet"
        result = runner.invoke(app, ["build-cache", str(image_dir), "--output", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        cache = MetadataCache.load(out)
        assert len(cache) == 4

    def test_refresh(self, image_dir: Path, tmp_path: Path):
        out = tmp_path / "cache.parquet"
        runner.invoke(app, ["build-cache", str(image_dir), "--output", str(out)])
        # Touch one file to invalidate it
        target = next(image_dir.glob("*.jpg"))
        os.utime(target, ns=(10**18, 10**18))
        result = runner.invoke(
            app,
            ["build-cache", str(image_dir), "--output", str(out), "--refresh"],
        )
        assert result.exit_code == 0
        assert "reused" in result.output

    def test_empty_dir(self, tmp_path: Path):
        out = tmp_path / "cache.parquet"
        result = runner.invoke(app, ["build-cache", str(tmp_path), "--output", str(out)])
        assert result.exit_code != 0
