"""Metadata cache: persist ``(path, width, height, mtime)`` rows to parquet.

Building a :class:`bucketsampler.torch.BucketedDataset` over a million
images takes a meaningful amount of time even with header-only PIL reads
because each read still pays a syscall + I/O. The cache eliminates that
work on second and subsequent runs: scan once, persist to parquet, and
next time skip straight to bucket assignment.

Cache invalidation is per-row: each entry stores the image's ``mtime``
at scan time, and the loader treats a stale ``mtime`` (or a missing
file) as a cache miss for that row. Files added since the last scan
are detected and re-read; files removed are dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

_PARQUET_SCHEMA_VERSION = 1
_PARQUET_COLUMNS = ("path", "width", "height", "mtime_ns", "size_bytes")


@dataclass(frozen=True, slots=True)
class MetadataRow:
    """One row of the metadata cache.

    Attributes:
        path: Absolute image path, as a string.
        width: Image width in pixels.
        height: Image height in pixels.
        mtime_ns: File modification time in nanoseconds since the epoch.
            Used for cache invalidation, the cache rereads the row when
            the on-disk mtime changes.
        size_bytes: File size in bytes at scan time.
    """

    path: str
    width: int
    height: int
    mtime_ns: int
    size_bytes: int


class MetadataCache:
    """In-memory metadata table, persistable as parquet.

    The cache is path-keyed and append-only in the API: add new rows
    with :meth:`upsert`, fetch dims with :meth:`get_dim`, write to disk
    with :meth:`save`, and reload with :meth:`load`. ``upsert`` overwrites
    any existing row for the same path, which is what callers want
    when re-scanning a directory whose images may have been edited.
    """

    def __init__(self, rows: Iterable[MetadataRow] | None = None) -> None:
        self._rows: dict[str, MetadataRow] = {}
        if rows is not None:
            for r in rows:
                self._rows[r.path] = r

    @classmethod
    def load(cls, path: str | Path) -> MetadataCache:
        """Load a metadata cache from a parquet file.

        Args:
            path: Parquet file path produced by :meth:`save`.

        Raises:
            ImportError: If pyarrow is not installed.
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is missing required columns or its
                schema version is unsupported.
        """
        pq = _import_pyarrow()
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"no such cache file: {p}")
        table = pq.read_table(p)
        for col in _PARQUET_COLUMNS:
            if col not in table.column_names:
                raise ValueError(
                    f"cache file {p} missing column {col!r}; have {table.column_names}"
                )
        version = table.schema.metadata or {}
        v = version.get(b"bucketsampler_schema_version")
        if v is not None and int(v) > _PARQUET_SCHEMA_VERSION:
            raise ValueError(
                f"cache file {p} schema version {int(v)} is newer than "
                f"this build's {_PARQUET_SCHEMA_VERSION}"
            )
        cols = {name: table.column(name).to_pylist() for name in _PARQUET_COLUMNS}
        rows = [
            MetadataRow(
                path=str(cols["path"][i]),
                width=int(cols["width"][i]),
                height=int(cols["height"][i]),
                mtime_ns=int(cols["mtime_ns"][i]),
                size_bytes=int(cols["size_bytes"][i]),
            )
            for i in range(len(cols["path"]))
        ]
        return cls(rows)

    def save(self, path: str | Path) -> None:
        """Write the cache to a parquet file.

        The schema version is stored in the parquet metadata so future
        loaders can refuse files they do not understand.
        """
        pa, pq = _import_pyarrow_full()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "path": pa.array([r.path for r in self._rows.values()], type=pa.string()),
            "width": pa.array([r.width for r in self._rows.values()], type=pa.int32()),
            "height": pa.array([r.height for r in self._rows.values()], type=pa.int32()),
            "mtime_ns": pa.array([r.mtime_ns for r in self._rows.values()], type=pa.int64()),
            "size_bytes": pa.array([r.size_bytes for r in self._rows.values()], type=pa.int64()),
        }
        table = pa.table(
            arrays,
            metadata={
                b"bucketsampler_schema_version": str(_PARQUET_SCHEMA_VERSION).encode("ascii"),
            },
        )
        pq.write_table(table, p)

    def upsert(self, row: MetadataRow) -> None:
        """Add or replace a row keyed by ``row.path``."""
        self._rows[row.path] = row

    def upsert_many(self, rows: Iterable[MetadataRow]) -> None:
        """Bulk version of :meth:`upsert`."""
        for r in rows:
            self._rows[r.path] = r

    def get(self, path: str | Path) -> MetadataRow | None:
        """Return the row for ``path`` or ``None`` if absent."""
        return self._rows.get(str(path))

    def get_dim(self, path: str | Path) -> tuple[int, int] | None:
        """Return ``(width, height)`` for ``path`` or ``None`` if absent."""
        r = self._rows.get(str(path))
        return None if r is None else (r.width, r.height)

    def is_stale(self, path: str | Path) -> bool:
        """Whether the on-disk file has a newer mtime than the cached row.

        Missing files and unknown paths are considered stale (callers
        should retreat to a fresh read).
        """
        p = Path(path)
        cached = self._rows.get(str(p))
        if cached is None:
            return True
        try:
            stat = p.stat()
        except OSError:
            return True
        return int(stat.st_mtime_ns) != cached.mtime_ns

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, key: object) -> bool:
        return str(key) in self._rows

    def __iter__(self) -> Iterator[MetadataRow]:
        return iter(self._rows.values())

    def paths(self) -> list[str]:
        """All cached paths, in insertion order."""
        return list(self._rows.keys())


def build_metadata_cache(
    paths: Sequence[str | Path],
    *,
    existing: MetadataCache | None = None,
    num_workers: int = 8,
) -> MetadataCache:
    """Scan ``paths`` and produce / refresh a :class:`MetadataCache`.

    If ``existing`` is provided, rows whose on-disk ``mtime`` matches the
    cached value are reused (free). Stale rows and never-seen paths get
    fresh PIL header reads. Removed paths are dropped from the returned
    cache.

    Args:
        paths: Image file paths to (re)scan.
        existing: A previously loaded cache to update in place. Pass
            ``None`` to build a cache from scratch.
        num_workers: Threads for the parallel header-read pass. Set to
            ``<= 1`` for serial reads.

    Returns:
        A :class:`MetadataCache` covering exactly the given paths.
        Caller is responsible for :meth:`MetadataCache.save`.
    """
    cache = MetadataCache()
    if existing is not None:
        wanted = {str(Path(p)) for p in paths}
        for r in existing:
            if r.path in wanted:
                cache.upsert(r)
    stale_or_new: list[Path] = []
    for raw in paths:
        p = Path(raw)
        s = str(p)
        cached = cache.get(s)
        if cached is None:
            stale_or_new.append(p)
            continue
        try:
            stat = p.stat()
        except OSError:
            stale_or_new.append(p)
            continue
        if int(stat.st_mtime_ns) != cached.mtime_ns:
            stale_or_new.append(p)

    if stale_or_new:
        rows = _read_rows_parallel(stale_or_new, num_workers=num_workers)
        cache.upsert_many(rows)
    return cache


def _read_one_row(path: Path) -> MetadataRow | None:
    try:
        stat = path.stat()
        with Image.open(path) as img:
            return MetadataRow(
                path=str(path),
                width=int(img.width),
                height=int(img.height),
                mtime_ns=int(stat.st_mtime_ns),
                size_bytes=int(stat.st_size),
            )
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _read_rows_parallel(paths: Sequence[Path], *, num_workers: int) -> list[MetadataRow]:
    if num_workers <= 1:
        return [r for r in (_read_one_row(p) for p in paths) if r is not None]
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        return [r for r in ex.map(_read_one_row, paths) if r is not None]


def dims_from_cache(paths: Sequence[str | Path], cache: MetadataCache) -> np.ndarray:
    """Lookup ``(width, height)`` for each path in the given cache.

    Missing rows raise ``KeyError`` so the caller can decide whether
    to refresh the cache or fall back to a fresh read.

    Returns:
        ``(N, 2)`` int64 array of dims in the same order as ``paths``.
    """
    out = np.empty((len(paths), 2), dtype=np.int64)
    for i, p in enumerate(paths):
        dim = cache.get_dim(p)
        if dim is None:
            raise KeyError(f"cache miss for {p}; refresh the cache first")
        out[i, 0] = dim[0]
        out[i, 1] = dim[1]
    return out


def _import_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "metadata cache requires pyarrow. Install with: pip install bucketsampler[cache]"
        ) from exc
    return pq


def _import_pyarrow_full() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "metadata cache requires pyarrow. Install with: pip install bucketsampler[cache]"
        ) from exc
    return pa, pq
