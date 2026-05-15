"""Caching layer: metadata and precomputed VAE latents.

Two pieces live here:

  - :mod:`bucketsampler.cache.metadata` persists per-image
    ``(path, width, height, mtime)`` rows so subsequent runs skip the
    PIL header read pass. Backed by parquet via pyarrow.
  - :mod:`bucketsampler.cache.latents` precomputes and stores VAE
    latents per bucket, so training does not pay the VAE forward pass
    on every step. Pluggable VAE adapters live alongside.
"""

from __future__ import annotations

from bucketsampler.cache.metadata import (
    MetadataCache,
    MetadataRow,
    build_metadata_cache,
)

__all__ = [
    "MetadataCache",
    "MetadataRow",
    "build_metadata_cache",
]
