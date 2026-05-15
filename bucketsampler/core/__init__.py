"""Framework-agnostic bucketing core.

This subpackage must not import torch, HuggingFace datasets, or any other
training-stack dependency. It is pure Python + numpy, so it can be unit-tested
without a GPU and reused outside the PyTorch ecosystem.
"""

from bucketsampler.core.assignment import (
    assign_many,
    assign_many_indices,
    best_bucket,
    crop_loss,
    log_ar_distance,
    resize_to_bucket_dims,
)
from bucketsampler.core.auto_bucket import (
    AutoBucketResult,
    AutoBuckets,
    bucket_set_to_toml,
    generate_buckets,
)
from bucketsampler.core.bucket import Bucket, BucketSet
from bucketsampler.core.stats import (
    aspect_ratio_summary,
    bucket_distribution,
    crop_loss_summary,
    underutilized_buckets,
)
from bucketsampler.core.strategies import FixedBuckets, Strategy

__all__ = [
    "AutoBucketResult",
    "AutoBuckets",
    "Bucket",
    "BucketSet",
    "FixedBuckets",
    "Strategy",
    "aspect_ratio_summary",
    "assign_many",
    "assign_many_indices",
    "best_bucket",
    "bucket_distribution",
    "bucket_set_to_toml",
    "crop_loss",
    "crop_loss_summary",
    "generate_buckets",
    "log_ar_distance",
    "resize_to_bucket_dims",
    "underutilized_buckets",
]
