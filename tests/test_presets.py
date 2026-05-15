"""Tests for bucketsampler.presets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bucketsampler import BucketSet, list_presets, load_from_json, load_from_toml, load_preset
from bucketsampler.exceptions import InvalidPresetError, PresetNotFoundError


class TestListPresets:
    def test_lists_bundled(self):
        names = list_presets()
        assert "sdxl" in names
        assert "sd15" in names
        assert "novelai" in names

    def test_sorted(self):
        names = list_presets()
        assert names == sorted(names)


class TestLoadPreset:
    @pytest.mark.parametrize("name", ["sdxl", "sd15", "novelai"])
    def test_known_presets_load(self, name):
        bs = load_preset(name)
        assert isinstance(bs, BucketSet)
        assert len(bs) >= 7
        assert bs.name == name
        assert bs.vae_factor >= 1

    def test_sdxl_has_1024_square(self):
        bs = load_preset("sdxl")
        assert (1024, 1024) in {b.as_tuple() for b in bs}

    def test_sdxl_buckets_multiple_of_vae_factor(self):
        bs = load_preset("sdxl")
        assert bs.all_multiples_of(bs.vae_factor)

    def test_sd15_buckets_multiple_of_vae_factor(self):
        bs = load_preset("sd15")
        assert bs.all_multiples_of(bs.vae_factor)

    def test_novelai_buckets_multiple_of_vae_factor(self):
        bs = load_preset("novelai")
        assert bs.all_multiples_of(bs.vae_factor)

    def test_missing_preset_raises(self):
        with pytest.raises(PresetNotFoundError) as excinfo:
            load_preset("nonexistent_preset")
        assert "nonexistent_preset" in str(excinfo.value)
        assert "sdxl" in str(excinfo.value)


class TestLoadFromToml:
    def test_round_trip(self, tmp_path):
        content = """
name = "custom"
description = "test"
vae_factor = 8

[[buckets]]
width = 512
height = 512

[[buckets]]
width = 768
height = 512
"""
        p = tmp_path / "custom.toml"
        p.write_text(content, encoding="utf-8")
        bs = load_from_toml(p)
        assert bs.name == "custom"
        assert bs.vae_factor == 8
        assert len(bs) == 2

    def test_missing_buckets_raises(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text('name = "bad"\n', encoding="utf-8")
        with pytest.raises(InvalidPresetError):
            load_from_toml(p)

    def test_negative_width_raises(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text(
            """
name = "bad"
[[buckets]]
width = -1
height = 512
""",
            encoding="utf-8",
        )
        with pytest.raises(InvalidPresetError):
            load_from_toml(p)

    def test_extra_field_raises(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text(
            """
name = "bad"
extra_field = "boom"
[[buckets]]
width = 512
height = 512
""",
            encoding="utf-8",
        )
        with pytest.raises(InvalidPresetError):
            load_from_toml(p)

    def test_str_path_accepted(self, tmp_path):
        p = tmp_path / "ok.toml"
        p.write_text(
            """
[[buckets]]
width = 512
height = 512
""",
            encoding="utf-8",
        )
        bs = load_from_toml(str(p))
        assert len(bs) == 1


class TestLoadFromJson:
    def test_round_trip(self, tmp_path):
        data = {
            "name": "custom",
            "description": "test",
            "vae_factor": 8,
            "buckets": [
                {"width": 512, "height": 512},
                {"width": 768, "height": 512},
            ],
        }
        p = tmp_path / "custom.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        bs = load_from_json(p)
        assert bs.name == "custom"
        assert len(bs) == 2

    def test_missing_buckets_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"name": "bad"}', encoding="utf-8")
        with pytest.raises(InvalidPresetError):
            load_from_json(p)

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_from_json(tmp_path / "missing.json")
