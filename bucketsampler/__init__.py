"""bucketsampler: aspect ratio bucketing for diffusion model training.

Public API (importable from this package's top level):

    from bucketsampler import (
        Bucket,
        BucketSet,
        FixedBuckets,
        Strategy,
        best_bucket,
        assign_many,
        load_preset,
        list_presets,
    )

Everything else lives under ``bucketsampler.core``, ``bucketsampler.presets``,
or ``bucketsampler.cli``. PyTorch and HuggingFace integration arrive in later
milestones and will not be importable from this top-level surface until then.
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
from bucketsampler.presets import (
    list_presets,
    load_from_json,
    load_from_toml,
    load_preset,
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
    "list_presets",
    "load_from_json",
    "load_from_toml",
    "load_preset",
    "log_ar_distance",
    "resize_to_bucket_dims",
    "underutilized_buckets",
]
