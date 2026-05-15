"""Command-line interface entry point.

Each subcommand lives in its own module under :mod:`bucketsampler.cli`
and is registered against the shared typer app below.
"""

from __future__ import annotations

import json
from typing import Any

import typer

from bucketsampler import __version__
from bucketsampler.cli.analyze import analyze as _analyze_command
from bucketsampler.cli.buckets import buckets_from_dataset as _buckets_from_dataset
from bucketsampler.cli.cache import build_cache as _build_cache_command
from bucketsampler.cli.precompute import precompute as _precompute_command
from bucketsampler.presets import list_presets, load_preset

app = typer.Typer(
    name="bucketsampler",
    help="Aspect ratio bucketing for diffusion model training.",
    no_args_is_help=True,
    add_completion=False,
)

app.command(name="analyze")(_analyze_command)
app.command(name="buckets-from-dataset")(_buckets_from_dataset)
app.command(name="build-cache")(_build_cache_command)
app.command(name="precompute")(_precompute_command)


@app.command()
def version() -> None:
    """Print the installed bucketsampler version."""
    typer.echo(__version__)


@app.command()
def presets(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a table."
    ),
) -> None:
    """List bundled bucket-set presets and their bucket dimensions."""
    names = list_presets()
    payload: dict[str, dict[str, Any]] = {}
    for name in names:
        bs = load_preset(name)
        payload[name] = {
            "description": bs.description,
            "vae_factor": bs.vae_factor,
            "n_buckets": len(bs),
            "buckets": [[b.width, b.height] for b in bs],
        }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    for name, info in payload.items():
        n = info["n_buckets"]
        desc = info["description"] or ""
        typer.echo(f"{name}  ({n} buckets)  {desc}")
        for w, h in info["buckets"]:
            typer.echo(f"    {w}x{h}")


def main() -> None:
    """Entry point used by the ``bucketsampler`` console script."""
    app()


__all__ = ["app", "main"]
