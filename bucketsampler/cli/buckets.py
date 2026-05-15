"""``bucketsampler buckets-from-dataset`` command.

Derives a bucket set from an image directory using k-means on the
log-aspect-ratio distribution and (optionally) writes the result to a
TOML file you can later load with :func:`bucketsampler.load_from_toml`
or feed back into ``bucketsampler analyze --bucket-config``.

The command also compares the auto-generated set against a chosen preset
so you can see whether the data-driven set actually improves crop loss
on your dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

from bucketsampler.cli.analyze import read_dims_safe, scan_images
from bucketsampler.core.auto_bucket import bucket_set_to_toml, generate_buckets
from bucketsampler.core.stats import crop_loss_summary
from bucketsampler.presets import load_preset


def buckets_from_dataset(
    path: Path = typer.Argument(..., help="Image directory to scan."),
    num: int = typer.Option(8, "--num", "-n", help="Number of buckets to generate (k in k-means)."),
    target: int = typer.Option(
        1024,
        "--target",
        "-t",
        help="Approximate target side length (pixel budget = target^2).",
    ),
    vae_factor: int = typer.Option(
        64, "--vae-factor", help="Round each dim to a multiple of this."
    ),
    name: str = typer.Option("auto", "--name", help="Name stored in the output bucket set."),
    seed: int = typer.Option(0, "--seed", help="K-means seed."),
    workers: int = typer.Option(8, help="Threads for parallel header reads."),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Descend into subdirectories."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the generated buckets to this TOML path.",
    ),
    compare_to: str | None = typer.Option(
        None,
        "--compare-to",
        help="Preset name to compare crop loss against (e.g. sdxl).",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit a JSON summary instead of plain text."
    ),
) -> None:
    """Generate a bucket set tailored to a dataset."""
    paths = scan_images(path, recursive=recursive)
    if not paths:
        typer.echo(f"No images found under {path}", err=True)
        raise typer.Exit(code=1)
    dims, broken = read_dims_safe(paths, workers=workers)
    if dims.size == 0:
        typer.echo(f"No readable images under {path}", err=True)
        raise typer.Exit(code=1)

    result = generate_buckets(
        dims,
        num_buckets=num,
        target=target,
        vae_factor=vae_factor,
        seed=seed,
        name=name,
        description=f"Auto-generated from {dims.shape[0]} images "
        f"({len(broken)} broken) under {path}",
    )

    compared_loss: float | None = None
    if compare_to is not None:
        compared_loss = float(crop_loss_summary(dims, load_preset(compare_to)).mean)

    written_path: str | None = None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(bucket_set_to_toml(result.bucket_set), encoding="utf-8")
        written_path = str(output)

    if json_output:
        payload = _json_summary(
            path=path,
            result=result,
            broken=len(broken),
            compare_to=compare_to,
            compared_loss=compared_loss,
            written=written_path,
        )
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(
            _text_summary(
                path=path,
                result=result,
                readable=int(dims.shape[0]),
                broken=len(broken),
                compare_to=compare_to,
                compared_loss=compared_loss,
                written=written_path,
            )
        )
    sys.stdout.flush()


def _text_summary(
    *,
    path: Path,
    result: Any,
    readable: int,
    broken: int,
    compare_to: str | None,
    compared_loss: float | None,
    written: str | None,
) -> str:
    lines: list[str] = []
    lines.append(f"Scanned {readable + broken} files under {path}")
    lines.append(f"  readable: {readable}")
    lines.append(f"  broken:   {broken}")
    lines.append("")
    lines.append(
        f"Generated {len(result.bucket_set)} buckets "
        f"(requested {result.requested_k}, k-means converged in "
        f"{result.iterations} iterations):"
    )
    for b, size, center in zip(
        result.bucket_set,
        result.cluster_sizes,
        result.cluster_centers,
        strict=False,
    ):
        ratio = b.aspect_ratio
        lines.append(f"  {b!s:<12}  AR={ratio:6.3f}  cluster log-AR={center:6.3f}  size={size}")
    lines.append("")
    lines.append(f"Mean crop loss (auto):      {result.crop_loss_mean * 100:5.2f}%")
    if compared_loss is not None:
        delta = compared_loss - result.crop_loss_mean
        sign = "-" if delta >= 0 else "+"
        lines.append(
            f"Mean crop loss ({compare_to}):    {compared_loss * 100:5.2f}%  "
            f"(auto is {sign}{abs(delta) * 100:.2f} pts vs {compare_to})"
        )
    if written:
        lines.append("")
        lines.append(f"Wrote bucket set to {written}")
    return "\n".join(lines)


def _json_summary(
    *,
    path: Path,
    result: Any,
    broken: int,
    compare_to: str | None,
    compared_loss: float | None,
    written: str | None,
) -> dict[str, Any]:
    return {
        "root": str(path),
        "readable": sum(result.cluster_sizes),
        "broken": broken,
        "requested_k": result.requested_k,
        "iterations": result.iterations,
        "crop_loss_mean": result.crop_loss_mean,
        "compare_to": compare_to,
        "compare_to_crop_loss_mean": compared_loss,
        "buckets": [
            {
                "width": b.width,
                "height": b.height,
                "aspect_ratio": b.aspect_ratio,
                "cluster_log_ar": center,
                "cluster_size": size,
            }
            for b, size, center in zip(
                result.bucket_set,
                result.cluster_sizes,
                result.cluster_centers,
                strict=False,
            )
        ],
        "output_path": written,
    }
