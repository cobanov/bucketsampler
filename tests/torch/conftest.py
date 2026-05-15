"""Shared fixtures for torch-integration tests."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image


def make_synthetic_image(path: Path, width: int, height: int, seed: int = 0) -> None:
    """Write a synthetic RGB JPEG with the given dimensions.

    The pixels are constant per image but vary by seed so different images
    decode to visibly different tensors. Cheap enough for unit tests.
    """
    rng = random.Random(seed)
    color = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
    Image.new("RGB", (width, height), color).save(path, format="JPEG", quality=70)


@pytest.fixture
def image_paths(tmp_path: Path) -> list[Path]:
    """A modest synthetic image set spanning square, wide, and tall ARs."""
    specs = [
        (1024, 1024),
        (1024, 1024),
        (1024, 1024),
        (1024, 1024),
        (2048, 1024),
        (2048, 1024),
        (1024, 2048),
        (1024, 2048),
        (1280, 720),
        (720, 1280),
        (4000, 1000),
        (1000, 4000),
    ]
    paths: list[Path] = []
    for i, (w, h) in enumerate(specs):
        p = tmp_path / f"img_{i:03d}.jpg"
        make_synthetic_image(p, w, h, seed=i)
        paths.append(p)
    return paths
