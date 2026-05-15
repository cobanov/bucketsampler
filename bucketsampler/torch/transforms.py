"""Resize-and-crop transform that lands an image on exact bucket dims.

Only one transform ships here: :class:`BucketResize`. Per the project
design, augmentations are out of scope, the user should chain their own
normalization, colour jitter, etc. on top of the bucket-fit tensor.
"""

from __future__ import annotations

import torch
from PIL import Image

from bucketsampler.core.assignment import resize_to_bucket_dims
from bucketsampler.core.bucket import Bucket


class BucketResize:
    """Fit an image to a bucket via resize-then-center-crop.

    The image is scaled so it fully covers the bucket on both axes
    (preserving aspect ratio), then the overhang is removed with a
    symmetric center crop. The output is a float32 CHW tensor in
    ``[0, 1]``; downstream code can chain its own normalization.

    Args:
        bucket: Target bucket whose ``(width, height)`` the output matches.
        resample: PIL resampling filter for the resize step. Defaults to
            bicubic, which is the common SD-family training default.

    Example:
        >>> from PIL import Image
        >>> from bucketsampler import Bucket
        >>> from bucketsampler.torch import BucketResize
        >>> img = Image.new("RGB", (2048, 1024))
        >>> t = BucketResize(Bucket(1024, 512))
        >>> t(img).shape
        torch.Size([3, 512, 1024])
    """

    def __init__(
        self,
        bucket: Bucket,
        *,
        resample: Image.Resampling = Image.Resampling.BICUBIC,
    ) -> None:
        self.bucket = bucket
        self.resample = resample

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError(f"BucketResize expects a PIL.Image.Image, got {type(image).__name__}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        (new_w, new_h), (crop_x, crop_y) = resize_to_bucket_dims(
            image.width, image.height, self.bucket
        )
        resized = image.resize((new_w, new_h), resample=self.resample)
        cropped = resized.crop(
            (crop_x, crop_y, crop_x + self.bucket.width, crop_y + self.bucket.height)
        )
        return _pil_to_chw_float(cropped)

    def __repr__(self) -> str:
        return f"BucketResize(bucket={self.bucket})"


def _pil_to_chw_float(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float32 CHW tensor in ``[0, 1]``.

    Kept private and unnormalized; users compose their own normalization.
    """
    import numpy as np

    arr = np.array(image, dtype=np.uint8, copy=True)
    if arr.ndim == 2:
        arr = arr[..., None]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0)
    return tensor
