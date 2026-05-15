"""Tests for the analyze CLI command and its building blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from bucketsampler.cli import app
from bucketsampler.cli.analyze import (
    build_report,
    format_html,
    format_json,
    format_text,
    read_dims_safe,
    scan_images,
)
from bucketsampler.presets import load_preset

runner = CliRunner()


def _write_image(path: Path, w: int, h: int, color: tuple[int, int, int] = (128, 128, 128)) -> None:
    Image.new("RGB", (w, h), color=color).save(path, format="JPEG", quality=60)


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    specs = [
        (1024, 1024),
        (1024, 1024),
        (1024, 1024),
        (2048, 1024),
        (2048, 1024),
        (1024, 2048),
        (4000, 1000),
    ]
    for i, (w, h) in enumerate(specs):
        _write_image(tmp_path / f"img_{i:03d}.jpg", w, h)
    # Non-image file to confirm filtering
    (tmp_path / "README.txt").write_text("ignore me")
    # Broken image (truncated JPEG header)
    (tmp_path / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    return tmp_path


@pytest.fixture
def nested_image_dir(tmp_path: Path) -> Path:
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_image(tmp_path / "top.jpg", 1024, 1024)
    _write_image(sub / "nested.jpg", 1024, 1024)
    return tmp_path


class TestScanImages:
    def test_finds_images_only(self, image_dir: Path):
        paths = scan_images(image_dir)
        # 7 valid + 1 broken jpeg; readme.txt excluded
        suffixes = {p.suffix.lower() for p in paths}
        assert suffixes == {".jpg"}
        assert len(paths) == 8

    def test_recursive_default(self, nested_image_dir: Path):
        paths = scan_images(nested_image_dir)
        assert len(paths) == 2

    def test_recursive_off(self, nested_image_dir: Path):
        paths = scan_images(nested_image_dir, recursive=False)
        assert len(paths) == 1

    def test_missing_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            scan_images(tmp_path / "does-not-exist")

    def test_not_a_dir_raises(self, tmp_path: Path):
        f = tmp_path / "f"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            scan_images(f)

    def test_custom_suffixes(self, tmp_path: Path):
        _write_image(tmp_path / "img.jpg", 64, 64)
        Image.new("RGB", (64, 64)).save(tmp_path / "img.png")
        png_only = scan_images(tmp_path, suffixes=(".png",))
        assert [p.suffix for p in png_only] == [".png"]


class TestReadDimsSafe:
    def test_reads_dims(self, image_dir: Path):
        paths = sorted(image_dir.glob("img_*.jpg"))
        dims, broken = read_dims_safe(paths, workers=4)
        assert dims.shape == (len(paths), 2)
        assert broken == []

    def test_separates_broken(self, image_dir: Path):
        paths = sorted(image_dir.glob("*.jpg"))  # includes broken.jpg
        dims, broken = read_dims_safe(paths, workers=4)
        assert len(broken) == 1
        assert broken[0].name == "broken.jpg"
        assert dims.shape[0] == len(paths) - 1

    def test_serial_path(self, image_dir: Path):
        paths = sorted(image_dir.glob("img_*.jpg"))
        dims_p, _ = read_dims_safe(paths, workers=8)
        dims_s, _ = read_dims_safe(paths, workers=1)
        assert np.array_equal(dims_p, dims_s)

    def test_empty_paths(self):
        dims, broken = read_dims_safe([], workers=4)
        assert dims.shape == (0, 2)
        assert broken == []


class TestBuildReport:
    def test_basic_counts(self, image_dir: Path):
        paths = sorted(image_dir.glob("img_*.jpg"))
        dims, broken = read_dims_safe(paths, workers=4)
        bs = load_preset("sdxl")
        report = build_report(
            root=image_dir,
            paths=paths,
            dims=dims,
            broken=broken,
            bucket_set=bs,
            min_count=10,
            outlier_threshold=0.5,
        )
        assert report.readable == len(paths)
        assert report.broken == 0
        assert report.bucket_set_name == "sdxl"
        assert sum(bc["count"] for bc in report.bucket_counts) == len(paths)

    def test_broken_propagates(self, image_dir: Path):
        paths_all = sorted(image_dir.glob("*.jpg"))
        dims, broken = read_dims_safe(paths_all, workers=4)
        readable = [p for p in paths_all if p not in set(broken)]
        bs = load_preset("sdxl")
        report = build_report(
            root=image_dir,
            paths=readable,
            dims=dims,
            broken=broken,
            bucket_set=bs,
        )
        assert report.broken == 1
        assert report.total_scanned == len(paths_all)
        assert any("broken.jpg" in p for p in report.broken_paths)

    def test_outliers_flagged(self, image_dir: Path):
        paths = sorted(image_dir.glob("img_*.jpg"))
        dims, _ = read_dims_safe(paths, workers=4)
        bs = load_preset("sdxl")
        report = build_report(
            root=image_dir,
            paths=paths,
            dims=dims,
            broken=[],
            bucket_set=bs,
            outlier_threshold=0.3,
        )
        # 4000x1000 (AR=4) is far from any sdxl bucket
        assert any(o.aspect_ratio == 4.0 for o in report.outliers)

    def test_outliers_capped(self, tmp_path: Path):
        # Many outliers, but max_outliers_listed should cap
        for i in range(30):
            _write_image(tmp_path / f"wide_{i}.jpg", 5000, 500)
        paths = sorted(tmp_path.glob("*.jpg"))
        dims, _ = read_dims_safe(paths, workers=4)
        bs = load_preset("sdxl")
        report = build_report(
            root=tmp_path,
            paths=paths,
            dims=dims,
            broken=[],
            bucket_set=bs,
            outlier_threshold=0.5,
            max_outliers_listed=5,
        )
        assert len(report.outliers) == 5

    def test_underutilized(self, image_dir: Path):
        paths = sorted(image_dir.glob("img_*.jpg"))
        dims, _ = read_dims_safe(paths, workers=4)
        bs = load_preset("sdxl")
        report = build_report(
            root=image_dir,
            paths=paths,
            dims=dims,
            broken=[],
            bucket_set=bs,
            min_count=10,
        )
        assert len(report.underutilized) > 0


class TestFormatters:
    @pytest.fixture
    def report(self, image_dir: Path):
        paths_all = sorted(image_dir.glob("*.jpg"))
        dims, broken = read_dims_safe(paths_all, workers=4)
        readable = [p for p in paths_all if p not in set(broken)]
        return build_report(
            root=image_dir,
            paths=readable,
            dims=dims,
            broken=broken,
            bucket_set=load_preset("sdxl"),
        )

    def test_text_contains_key_sections(self, report):
        text = format_text(report)
        assert "Scanned" in text
        assert "Aspect ratio" in text
        assert "Buckets" in text
        assert "Crop loss" in text

    def test_json_is_valid(self, report):
        payload = json.loads(format_json(report))
        for key in ("root", "readable", "broken", "bucket_counts", "ar_summary"):
            assert key in payload

    def test_json_handles_nan(self, tmp_path: Path):
        bs = load_preset("sdxl")
        report = build_report(
            root=tmp_path,
            paths=[],
            dims=np.zeros((0, 2), np.int64),
            broken=[],
            bucket_set=bs,
        )
        # No images = NaN summary; JSON must still parse
        payload = json.loads(format_json(report))
        assert payload["ar_summary"]["mean_log_ar"] is None

    def test_html_is_self_contained(self, report, image_dir, tmp_path):
        paths_all = sorted(image_dir.glob("*.jpg"))
        dims, _ = read_dims_safe(paths_all, workers=4)
        html = format_html(report, dims=dims)
        assert "<!DOCTYPE html>" in html
        assert "data:image/png;base64," in html

    def test_html_with_empty_dims(self, report):
        html = format_html(report, dims=np.zeros((0, 2), np.int64))
        assert "<!DOCTYPE html>" in html


class TestCLI:
    def test_text_output(self, image_dir: Path):
        result = runner.invoke(app, ["analyze", str(image_dir), "--preset", "sdxl"])
        assert result.exit_code == 0
        assert "readable:" in result.output
        assert "Buckets" in result.output

    def test_json_output(self, image_dir: Path):
        result = runner.invoke(app, ["analyze", str(image_dir), "--preset", "sdxl", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["bucket_set_name"] == "sdxl"

    def test_html_writes_file(self, image_dir: Path, tmp_path: Path):
        out = tmp_path / "report.html"
        result = runner.invoke(
            app, ["analyze", str(image_dir), "--preset", "sdxl", "--html", str(out)]
        )
        assert result.exit_code == 0
        assert out.exists()
        assert "<!DOCTYPE html>" in out.read_text()

    def test_custom_bucket_config(self, image_dir: Path, tmp_path: Path):
        cfg = tmp_path / "buckets.toml"
        cfg.write_text(
            """
name = "tiny"
vae_factor = 8
[[buckets]]
width = 1024
height = 1024
[[buckets]]
width = 2048
height = 1024
""",
            encoding="utf-8",
        )
        result = runner.invoke(app, ["analyze", str(image_dir), "--bucket-config", str(cfg)])
        assert result.exit_code == 0
        assert "tiny" in result.output

    def test_requires_preset_or_config(self, image_dir: Path):
        result = runner.invoke(app, ["analyze", str(image_dir)])
        assert result.exit_code != 0

    def test_rejects_both_preset_and_config(self, image_dir: Path, tmp_path: Path):
        cfg = tmp_path / "buckets.toml"
        cfg.write_text("[[buckets]]\nwidth = 512\nheight = 512\n", encoding="utf-8")
        result = runner.invoke(
            app,
            ["analyze", str(image_dir), "--preset", "sdxl", "--bucket-config", str(cfg)],
        )
        assert result.exit_code != 0

    def test_empty_directory(self, tmp_path: Path):
        result = runner.invoke(app, ["analyze", str(tmp_path), "--preset", "sdxl"])
        assert result.exit_code != 0
        assert "No images" in result.output or "No images" in (result.stderr or "")
