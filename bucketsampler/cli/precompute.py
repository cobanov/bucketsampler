"""``bucketsampler precompute`` command.

Walks an image directory, builds a :class:`BucketedDataset`, encodes
everything through a VAE, and writes per-bucket latents to disk. The
training script then reads from the latent dataset and skips the VAE
forward pass entirely.

Only SD-family VAEs are wired up by default (via diffusers). Custom
encoders can be used programmatically by calling
:func:`bucketsampler.cache.latents.precompute_latents` directly.

Torch and torch-dependent helpers (BucketedDataset, precompute_latents,
SDVAEEncoder) are imported inside the command body so importing this
module does not require ``pip install bucketsampler[torch]``; only
running the command does.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from bucketsampler.cli.analyze import scan_images
from bucketsampler.presets import list_presets, load_from_json, load_from_toml, load_preset


def precompute(
    path: Path = typer.Argument(..., help="Image directory to encode."),
    vae: str = typer.Option(
        ...,
        "--vae",
        help="VAE model id (e.g. stabilityai/sdxl-vae) or local path.",
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Directory to write latents and manifest."
    ),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help="Bundled bucket preset (sdxl, sd15, novelai)."
    ),
    bucket_config: Path | None = typer.Option(
        None, "--bucket-config", "-c", help="Custom bucket TOML/JSON path."
    ),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="VAE batch size."),
    dtype: str = typer.Option(
        "bfloat16",
        "--dtype",
        help="Storage dtype: float32, float16, or bfloat16.",
    ),
    device: str | None = typer.Option(
        None, "--device", help="Torch device override (e.g. cuda:0)."
    ),
    workers: int = typer.Option(8, help="Threads for header reads."),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Descend into subdirectories."
    ),
    captions_file: Path | None = typer.Option(
        None,
        "--captions-file",
        help="Optional newline-delimited captions, one per image, same order as scan.",
    ),
) -> None:
    """Encode an image directory through a VAE and persist per-bucket latents."""
    try:
        import torch

        from bucketsampler import FixedBuckets
        from bucketsampler.cache.latents import precompute_latents
        from bucketsampler.cache.vae_adapters.sd import SDVAEEncoder
        from bucketsampler.torch import BucketedDataset
    except ImportError as exc:
        raise typer.BadParameter(
            "the precompute command requires torch and the [vae] extra; "
            "install with: pip install 'bucketsampler[vae]'"
        ) from exc

    if bool(preset) == bool(bucket_config):
        raise typer.BadParameter(
            "pass exactly one of --preset "
            f"(available: {', '.join(list_presets())}) or --bucket-config"
        )
    if preset is not None:
        bucket_set = load_preset(preset)
    else:
        assert bucket_config is not None
        bucket_set = (
            load_from_json(bucket_config)
            if bucket_config.suffix.lower() == ".json"
            else load_from_toml(bucket_config)
        )

    paths = scan_images(path, recursive=recursive)
    if not paths:
        typer.echo(f"No images found under {path}", err=True)
        raise typer.Exit(code=1)

    captions: list[str] | None = None
    if captions_file is not None:
        captions = captions_file.read_text(encoding="utf-8").splitlines()
        if len(captions) != len(paths):
            typer.echo(
                f"captions file has {len(captions)} lines but scan found {len(paths)} images",
                err=True,
            )
        captions = captions[: len(paths)]

    strategy = FixedBuckets(bucket_set)
    dataset = BucketedDataset(
        paths=paths,
        strategy=strategy,
        captions=captions,
        num_workers=workers,
    )

    encoder = SDVAEEncoder(
        model_id=vae,
        device=device,
        dtype=_parse_dtype(dtype, torch),
    )

    typer.echo(
        f"Encoding {len(dataset)} images with {vae} "
        f"into {len(set(dataset.bucket_indices.tolist()))} buckets..."
    )
    manifest = precompute_latents(
        dataset,
        encoder,
        output_dir=output,
        batch_size=batch_size,
        dtype=_parse_dtype(dtype, torch),
        progress_callback=_progress_callback(),
    )
    typer.echo("")
    typer.echo(
        f"Wrote {len(manifest.buckets)} bucket files + manifest to {output} "
        f"(downsample={manifest.downsample_factor}x, "
        f"latent_channels={manifest.latent_channels})"
    )
    sys.stdout.flush()


def _parse_dtype(name: str, torch_mod: Any) -> Any:
    mapping = {
        "float32": torch_mod.float32,
        "fp32": torch_mod.float32,
        "float16": torch_mod.float16,
        "fp16": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
        "bf16": torch_mod.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise typer.BadParameter(f"unknown dtype {name!r}; choose one of {sorted(mapping)}")
    return mapping[key]


def _progress_callback() -> Callable[[int, int], None]:
    last: dict[str, int] = {"shown": -1}

    def cb(done: int, total: int) -> None:
        pct = int(100 * done / max(total, 1))
        if pct == last["shown"]:
            return
        last["shown"] = pct
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        typer.echo(f"\r  [{bar}] {pct}% ({done}/{total})", nl=False)

    return cb
