"""Command-line interface scaffolding.

Top-level commands are registered against a single typer app. Each command
lives in its own submodule so they can grow independently (analyzer, cache
builder, latent precompute, etc., per the milestone plan). For M1, only
``presets`` and ``version`` are implemented; running with no args prints help.
"""

from __future__ import annotations

import json
from typing import Any

import typer

from bucketsampler import __version__
from bucketsampler.presets import list_presets, load_preset

app = typer.Typer(
    name="bucketsampler",
    help="Aspect ratio bucketing for diffusion model training.",
    no_args_is_help=True,
    add_completion=False,
)


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
