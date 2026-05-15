"""Internal data-source abstraction for ``BucketedDataset``.

``BucketedDataset`` works the same for any source that can (a) report
``(width, height)`` for each item without doing a full decode, and (b)
hand back a PIL image when actually asked for one. The two shipped
implementations are :class:`_PathSource` (filesystem) and
:class:`_HFSource` (HuggingFace ``datasets.Dataset``).

This module is private. Users only see the classmethods
:meth:`BucketedDataset.from_paths` and :meth:`BucketedDataset.from_hf`.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class _DataSource(Protocol):
    """Read-only iteration of (image, optional caption, identifier).

    ``read_dim`` and ``open_image`` are per-item operations. ``read_all_dims``
    is the bulk path the constructor uses; default implementations work
    sequentially or via a thread pool, but a source that knows a cheaper
    bulk method (e.g. a parquet column) can override.
    """

    def __len__(self) -> int: ...

    def read_dim(self, idx: int) -> tuple[int, int]: ...

    def open_image(self, idx: int) -> Image.Image: ...

    def get_caption(self, idx: int) -> str | None: ...

    def identifier(self, idx: int) -> str: ...

    def read_all_dims(self, num_workers: int = 8) -> np.ndarray: ...


def _bulk_read_dims(source: _DataSource, num_workers: int) -> np.ndarray:
    """Default :meth:`_DataSource.read_all_dims` implementation."""
    n = len(source)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if num_workers <= 1:
        return np.asarray([source.read_dim(i) for i in range(n)], dtype=np.int64)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(source.read_dim, range(n)))
    return np.asarray(results, dtype=np.int64)


class _PathSource:
    """Filesystem source: each item is an image file on disk.

    Dim reads use PIL's header-only path (``Image.open`` without
    ``.load()``), so the dataset can be sized in milliseconds per image.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        captions: Sequence[str] | None = None,
    ) -> None:
        if len(paths) == 0:
            raise ValueError("at least one path is required")
        if captions is not None and len(captions) != len(paths):
            raise ValueError(
                f"captions length ({len(captions)}) must match paths length ({len(paths)})"
            )
        self.paths: list[Path] = [Path(p) for p in paths]
        self._captions: list[str] | None = list(captions) if captions is not None else None

    def __len__(self) -> int:
        return len(self.paths)

    def read_dim(self, idx: int) -> tuple[int, int]:
        with Image.open(self.paths[idx]) as img:
            return (int(img.width), int(img.height))

    def open_image(self, idx: int) -> Image.Image:
        return Image.open(self.paths[idx])

    def get_caption(self, idx: int) -> str | None:
        if self._captions is None:
            return None
        return self._captions[idx]

    def identifier(self, idx: int) -> str:
        return str(self.paths[idx])

    def read_all_dims(self, num_workers: int = 8) -> np.ndarray:
        return _bulk_read_dims(self, num_workers)
