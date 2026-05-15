"""Tests for ``bucketsampler buckets-from-dataset``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from bucketsampler import load_from_toml
from bucketsampler.cli import app

runner = CliRunner()


def _write_image(path: Path, w: int, h: int) -> None:
    Image.new("RGB", (w, h), color=(128, 128, 128)).save(path, format="JPEG", quality=50)


@pytest.fixture
def varied_image_dir(tmp_path: Path) -> Path:
    # Three clusters: square, landscape, portrait
    specs = [(1024, 1024)] * 10 + [(2000, 1000)] * 8 + [(1000, 2000)] * 6
    for i, (w, h) in enumerate(specs):
        _write_image(tmp_path / f"img_{i:03d}.jpg", w, h)
    return tmp_path


class TestCLI:
    def test_text_output(self, varied_image_dir: Path):
        result = runner.invoke(
            app,
            [
                "buckets-from-dataset",
                str(varied_image_dir),
                "--num",
                "3",
                "--target",
                "1024",
                "--seed",
                "0",
            ],
        )
        assert result.exit_code == 0
        assert "Generated 3 buckets" in result.output
        assert "Mean crop loss" in result.output

    def test_writes_toml_round_trip(self, varied_image_dir: Path, tmp_path: Path):
        out = tmp_path / "buckets.toml"
        result = runner.invoke(
            app,
            [
                "buckets-from-dataset",
                str(varied_image_dir),
                "--num",
                "3",
                "--target",
                "1024",
                "--seed",
                "0",
                "--output",
                str(out),
                "--name",
                "rt",
            ],
        )
        assert result.exit_code == 0
        assert out.exists()
        loaded = load_from_toml(out)
        assert loaded.name == "rt"
        assert len(loaded) == 3

    def test_compare_to_preset(self, varied_image_dir: Path):
        result = runner.invoke(
            app,
            [
                "buckets-from-dataset",
                str(varied_image_dir),
                "--num",
                "3",
                "--target",
                "1024",
                "--seed",
                "0",
                "--compare-to",
                "sdxl",
            ],
        )
        assert result.exit_code == 0
        assert "sdxl" in result.output

    def test_json_output(self, varied_image_dir: Path):
        result = runner.invoke(
            app,
            [
                "buckets-from-dataset",
                str(varied_image_dir),
                "--num",
                "3",
                "--target",
                "1024",
                "--seed",
                "0",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["requested_k"] == 3
        assert len(data["buckets"]) == 3
        for bucket in data["buckets"]:
            assert "width" in bucket
            assert "height" in bucket
            assert "aspect_ratio" in bucket

    def test_empty_directory(self, tmp_path: Path):
        result = runner.invoke(app, ["buckets-from-dataset", str(tmp_path)])
        assert result.exit_code != 0

    def test_creates_output_parent_dir(self, varied_image_dir: Path, tmp_path: Path):
        out = tmp_path / "nested" / "subdir" / "buckets.toml"
        result = runner.invoke(
            app,
            [
                "buckets-from-dataset",
                str(varied_image_dir),
                "--num",
                "3",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0
        assert out.exists()
