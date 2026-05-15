"""PyTorch integration for bucketsampler.

This subpackage imports ``torch`` at module load time. Importing it without
torch installed raises ``ImportError``. The top-level ``bucketsampler``
package guards its re-exports with a lazy ``__getattr__`` so a torch-free
install still works for core/ users.

Public surface:

    from bucketsampler.torch import (
        BucketedDataset,
        BucketBatchSampler,
        BucketResize,
    )

The same names are re-exported from ``bucketsampler`` for callers that
prefer the flat import.
"""

from __future__ import annotations

from bucketsampler.torch.dataset import BucketedDataset
from bucketsampler.torch.sampler import BucketBatchSampler
from bucketsampler.torch.transforms import BucketResize

__all__ = [
    "BucketBatchSampler",
    "BucketResize",
    "BucketedDataset",
]
