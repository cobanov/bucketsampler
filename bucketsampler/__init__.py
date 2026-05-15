"""bucketsampler: aspect ratio bucketing for diffusion model training.

Public API (importable from this package's top level):

    from bucketsampler import (
        Bucket,
        BucketSet,
        FixedBuckets,
        Strategy,
        BucketedDataset,
        BucketBatchSampler,
        BucketResize,
        best_bucket,
        assign_many,
        load_preset,
        list_presets,
    )

PyTorch-backed names (``BucketedDataset``, ``BucketBatchSampler``,
``BucketResize``) are imported lazily so that a torch-free install can still
use the core surface. Attempting to access one of those names without torch
installed raises a clear ``ImportError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

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

if TYPE_CHECKING:
    from bucketsampler.torch.dataset import BucketedDataset
    from bucketsampler.torch.sampler import BucketBatchSampler
    from bucketsampler.torch.transforms import BucketResize

_TORCH_LAZY_EXPORTS = {
    "BucketedDataset": ("bucketsampler.torch.dataset", "BucketedDataset"),
    "BucketBatchSampler": ("bucketsampler.torch.sampler", "BucketBatchSampler"),
    "BucketResize": ("bucketsampler.torch.transforms", "BucketResize"),
}


def __getattr__(name: str) -> Any:
    if name in _TORCH_LAZY_EXPORTS:
        module_name, attr_name = _TORCH_LAZY_EXPORTS[name]
        try:
            import importlib

            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"{name!r} requires PyTorch. Install with: pip install bucketsampler[torch]"
            ) from exc
        return getattr(module, attr_name)
    raise AttributeError(f"module 'bucketsampler' has no attribute {name!r}")


__all__ = [
    "AspectRatioSummary",
    "AutoBucketResult",
    "AutoBuckets",
    "Bucket",
    "BucketBatchSampler",
    "BucketResize",
    "BucketSamplerError",
    "BucketSet",
    "BucketedDataset",
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
    "bucket_set_to_toml",
    "crop_loss",
    "crop_loss_summary",
    "generate_buckets",
    "list_presets",
    "load_from_json",
    "load_from_toml",
    "load_preset",
    "log_ar_distance",
    "resize_to_bucket_dims",
    "underutilized_buckets",
]
