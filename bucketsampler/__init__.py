"""bucketsampler: aspect ratio bucketing for diffusion model training.

This is the framework-agnostic core surface. Presets, the CLI, and the
PyTorch / HuggingFace adapters extend this top level in later milestones.
"""

from __future__ import annotations

__version__ = "0.1.0"

from bucketsampler.core.assignment import (
    assign_many,
    assign_many_indices,
    best_bucket,
    crop_loss,
    log_ar_distance,
    resize_to_bucket_dims,
)
from bucketsampler.core.bucket import Bucket, BucketSet
from bucketsampler.core.stats import (
    AspectRatioSummary,
    CropLossSummary,
    aspect_ratio_summary,
    bucket_distribution,
    crop_loss_summary,
    underutilized_buckets,
)
from bucketsampler.core.strategies import FixedBuckets, Strategy
from bucketsampler.exceptions import (
    BucketSamplerError,
    DuplicateBucketError,
    EmptyBucketSetError,
    ImageTooSmallError,
    InvalidBucketError,
    InvalidPresetError,
    PresetNotFoundError,
)

__all__ = [
    "AspectRatioSummary",
    "Bucket",
    "BucketSamplerError",
    "BucketSet",
    "CropLossSummary",
    "DuplicateBucketError",
    "EmptyBucketSetError",
    "FixedBuckets",
    "ImageTooSmallError",
    "InvalidBucketError",
    "InvalidPresetError",
    "PresetNotFoundError",
    "Strategy",
    "__version__",
    "aspect_ratio_summary",
    "assign_many",
    "assign_many_indices",
    "best_bucket",
    "bucket_distribution",
    "crop_loss",
    "crop_loss_summary",
    "log_ar_distance",
    "resize_to_bucket_dims",
    "underutilized_buckets",
]
