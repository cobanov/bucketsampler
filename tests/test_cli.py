"""Tests for the bucketsampler CLI scaffolding."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from bucketsampler import __version__
from bucketsampler.cli import app

runner = CliRunner()


class TestHelp:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code != 0 or "Usage" in result.output

    def test_help_flag(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "bucketsampler" in result.output.lower()


class TestVersion:
    def test_prints_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestPresets:
    def test_lists_bundled(self):
        result = runner.invoke(app, ["presets"])
        assert result.exit_code == 0
        for name in ("sdxl", "sd15", "novelai"):
            assert name in result.output

    def test_json_output_is_valid_json(self):
        result = runner.invoke(app, ["presets", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "sdxl" in data
        assert "buckets" in data["sdxl"]
        assert isinstance(data["sdxl"]["buckets"], list)
        assert all(isinstance(p, list) and len(p) == 2 for p in data["sdxl"]["buckets"])
