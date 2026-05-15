"""``bucketsampler build-cache`` command.

Scans an image directory once and writes a parquet metadata cache so
subsequent ``BucketedDataset`` constructions can skip the PIL header
read pass. Optionally refreshes an existing cache: rows whose on-disk
mtime hasn't changed are reused, stale or new files are re-read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from bucketsampler.cache.metadata import MetadataCache, build_metadata_cache
from bucketsampler.cli.analyze import scan_images


def build_cache(
    path: Path = typer.Argument(..., help="Image directory to scan."),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to write the parquet cache file.",
    ),
    workers: int = typer.Option(8, help="Threads for parallel header reads."),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Descend into subdirectories."
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="If --output exists, reuse it and only re-read changed files.",
    ),
) -> None:
    """Build (or refresh) a parquet metadata cache for an image directory."""
    paths = scan_images(path, recursive=recursive)
    if not paths:
        typer.echo(f"No images found under {path}", err=True)
        raise typer.Exit(code=1)
    existing: MetadataCache | None = None
    if refresh and output.exists():
        existing = MetadataCache.load(output)
    cache = build_metadata_cache(paths, existing=existing, num_workers=workers)
    cache.save(output)
    reused = 0 if existing is None else sum(1 for r in cache if r.path in existing)
    new_or_stale = len(cache) - reused
    typer.echo(
        f"Wrote {len(cache)} rows to {output} "
        f"({reused} reused, {new_or_stale} freshly read, "
        f"{len(paths) - len(cache)} broken)."
    )
    sys.stdout.flush()
