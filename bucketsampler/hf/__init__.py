"""HuggingFace ``datasets`` integration.

Importing this subpackage requires the optional ``datasets`` extra. The
top-level ``bucketsampler.BucketedDataset.from_hf`` classmethod wraps the
underlying :class:`_HFSource` from this module so most users never need
to import it directly.
"""

from __future__ import annotations

from bucketsampler.hf.adapter import _HFSource

__all__ = ["_HFSource"]
