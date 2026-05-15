"""HuggingFace ``datasets`` adapter.

:class:`_HFSource` implements the
:class:`bucketsampler.torch._source._DataSource` protocol over a
``datasets.Dataset`` (map-style). It supports PIL columns (the common case),
raw bytes columns (``datasets.Image(decode=False)``), and tensor / numpy
columns (typical when the dataset was cast to a torch or numpy format
upstream).

Streaming (``IterableDataset``) is not yet supported: the bucketing
algorithm needs to size each batch from a known per-bucket index queue,
which an unindexable stream cannot provide without a separate buffered
sampler. Pass a map-style :class:`datasets.Dataset` for now.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from bucketsampler.torch._source import _bulk_read_dims

if TYPE_CHECKING:
    import datasets


class _HFSource:
    """Wrap a ``datasets.Dataset`` as a bucketsampler data source.

    Args:
        hf_dataset: A map-style ``datasets.Dataset`` (NOT an IterableDataset).
        image_column: Column name holding the image. Defaults to ``"image"``,
            which is the convention used by ``datasets.Image`` features.
        caption_column: Optional caption column. If ``None``, samples carry
            no ``"caption"`` key.

    Raises:
        ValueError: If a column is missing, or if ``hf_dataset`` does not
            look like a map-style Dataset.
    """

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        *,
        image_column: str = "image",
        caption_column: str | None = None,
    ) -> None:
        if not hasattr(hf_dataset, "__len__") or not hasattr(hf_dataset, "__getitem__"):
            raise ValueError(
                "BucketedDataset.from_hf requires a map-style datasets.Dataset; "
                "IterableDataset (streaming) is not supported yet."
            )
        column_names = getattr(hf_dataset, "column_names", None)
        if column_names is not None:
            if image_column not in column_names:
                raise ValueError(
                    f"image_column {image_column!r} not in dataset columns {column_names}"
                )
            if caption_column is not None and caption_column not in column_names:
                raise ValueError(
                    f"caption_column {caption_column!r} not in dataset columns {column_names}"
                )
        self.ds = hf_dataset
        self.image_col = image_column
        self.caption_col = caption_column

    def __len__(self) -> int:
        return len(self.ds)

    def read_dim(self, idx: int) -> tuple[int, int]:
        return _extract_dim(self.ds[idx][self.image_col])

    def open_image(self, idx: int) -> Image.Image:
        return _to_pil(self.ds[idx][self.image_col])

    def get_caption(self, idx: int) -> str | None:
        if self.caption_col is None:
            return None
        value = self.ds[idx][self.caption_col]
        return None if value is None else str(value)

    def identifier(self, idx: int) -> str:
        return f"hf:{idx}"

    def read_all_dims(self, num_workers: int = 8) -> np.ndarray:
        return _bulk_read_dims(self, num_workers)


def _extract_dim(item: Any) -> tuple[int, int]:
    """Return ``(width, height)`` for a single column value.

    Accepts ``PIL.Image.Image``, a ``{"bytes": ...}`` dict (``datasets.Image
    (decode=False)`` output), ``numpy.ndarray``, or ``torch.Tensor``. For
    multi-channel arrays / tensors, CHW vs HWC is detected by which axis
    looks like a small channel count.
    """
    if isinstance(item, Image.Image):
        return (int(item.width), int(item.height))
    if isinstance(item, dict) and "bytes" in item and item["bytes"] is not None:
        with Image.open(io.BytesIO(item["bytes"])) as img:
            return (int(img.width), int(img.height))
    if isinstance(item, dict) and "path" in item and item["path"] is not None:
        with Image.open(item["path"]) as img:
            return (int(img.width), int(img.height))
    shape = _get_shape(item)
    if shape is not None:
        h, w = _hw_from_shape(shape)
        return (int(w), int(h))
    raise TypeError(
        f"cannot extract dim from image column value of type "
        f"{type(item).__name__!r}; expected PIL.Image, bytes-dict, "
        "numpy.ndarray, or torch.Tensor"
    )


def _to_pil(item: Any) -> Image.Image:
    """Decode a single column value into a PIL image.

    Mirror of :func:`_extract_dim`: same input formats accepted, and tensors
    are converted assuming float values are in ``[0, 1]`` and integer values
    are already in ``[0, 255]``.
    """
    if isinstance(item, Image.Image):
        return item
    if isinstance(item, dict) and "bytes" in item and item["bytes"] is not None:
        return Image.open(io.BytesIO(item["bytes"]))
    if isinstance(item, dict) and "path" in item and item["path"] is not None:
        return Image.open(item["path"])
    arr = _to_numpy(item)
    if arr is None:
        raise TypeError(
            f"cannot convert image column value of type {type(item).__name__!r} to PIL.Image"
        )
    return _array_to_pil(arr)


def _get_shape(item: Any) -> tuple[int, ...] | None:
    shape = getattr(item, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(s) for s in shape)
    except TypeError:
        return None


def _hw_from_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return (int(shape[0]), int(shape[1]))
    if len(shape) == 3:
        if shape[0] <= 4 < shape[2]:
            return (int(shape[1]), int(shape[2]))
        return (int(shape[0]), int(shape[1]))
    raise ValueError(f"unexpected image shape {shape}; expected 2-D or 3-D")


def _to_numpy(item: Any) -> np.ndarray | None:
    if hasattr(item, "detach"):
        item = item.detach().cpu().numpy()  # torch tensor
    elif hasattr(item, "numpy") and not isinstance(item, np.ndarray):
        item = item.numpy()
    if isinstance(item, np.ndarray):
        return item
    return None


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 3 and arr.shape[0] <= 4 < arr.shape[2]:
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    raise ValueError(f"unexpected pixel-array shape {arr.shape}")
