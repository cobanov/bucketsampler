"""Precompute and serve VAE latents per bucket.

The training-time win is that the VAE forward pass is moved off the GPU
hot path. Latents are computed once with :func:`precompute_latents` and
stored on disk; training reads them with :class:`BucketedLatentDataset`
and only needs to run the U-Net (or whatever downstream model).

Storage layout (one safetensors file per bucket):

::

    <root>/
      manifest.json       # bucket set, dtype, scale_factor, counts
      bucket_0000.safetensors   # tensor "latents" shape (N0, C, H/f, W/f)
      bucket_0001.safetensors
      ...

The manifest records the exact bucket dimensions so the consumer can
verify the latents match the training configuration. Captions, when
provided, are stored as JSON next to the latents.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from bucketsampler.core.bucket import Bucket, BucketSet

if TYPE_CHECKING:
    from bucketsampler.cache.vae_adapters.protocol import VAEEncoder
    from bucketsampler.torch.dataset import BucketedDataset


_MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class _BucketManifestRow:
    bucket_idx: int
    width: int
    height: int
    count: int
    file: str


@dataclass(frozen=True, slots=True)
class LatentManifest:
    """Top-level manifest stored alongside the per-bucket safetensors files."""

    version: int
    dtype: str
    scale_factor: float
    downsample_factor: int
    latent_channels: int
    buckets: list[_BucketManifestRow]
    has_captions: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["buckets"] = [asdict(b) for b in self.buckets]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatentManifest:
        rows = [_BucketManifestRow(**b) for b in data["buckets"]]
        return cls(
            version=int(data["version"]),
            dtype=str(data["dtype"]),
            scale_factor=float(data["scale_factor"]),
            downsample_factor=int(data["downsample_factor"]),
            latent_channels=int(data["latent_channels"]),
            buckets=rows,
            has_captions=bool(data["has_captions"]),
        )


def precompute_latents(
    dataset: BucketedDataset,
    encoder: VAEEncoder,
    output_dir: str | Path,
    *,
    batch_size: int = 8,
    dtype: torch.dtype = torch.float32,
    pixel_normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> LatentManifest:
    """Encode every image in ``dataset`` with ``encoder`` and persist latents.

    Iterates the dataset in per-bucket batches (so VAE forward passes
    can stack), encodes them, and writes one safetensors file per bucket
    to ``output_dir``. The manifest records bucket dims, dtype, and the
    encoder's scale factor so the consumer can sanity-check at load time.

    Args:
        dataset: A :class:`BucketedDataset`. The dataset's
            ``__getitem__`` is called once per image.
        encoder: Any :class:`VAEEncoder`-conformant object.
        output_dir: Directory to receive the manifest and per-bucket
            files. Created if missing.
        batch_size: Encoder batch size. Lower if VRAM is tight; higher
            for throughput.
        dtype: Storage dtype for the latents. ``torch.bfloat16`` saves
            half the disk space versus float32.
        pixel_normalize: Optional callable mapping the dataset's tensor
            output (CHW float in ``[0, 1]``) to the encoder's expected
            input range. Defaults to ``2x - 1`` (``[0, 1]`` -> ``[-1, 1]``),
            which is the SD / SDXL convention.
        progress_callback: Optional ``(done, total)`` reporter, called
            once after each batch. ``done`` counts images, not batches.

    Returns:
        The :class:`LatentManifest` that was written.

    Raises:
        ImportError: If safetensors is not installed.
    """
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "precompute_latents requires safetensors. Install with: pip install bucketsampler[vae]"
        ) from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bucket_set: BucketSet = dataset.bucket_set
    normalize = pixel_normalize if pixel_normalize is not None else _default_normalize

    indices_by_bucket: dict[int, list[int]] = {}
    for i, b in enumerate(dataset.bucket_indices.tolist()):
        indices_by_bucket.setdefault(int(b), []).append(i)

    has_captions = False
    captions_by_bucket: dict[int, list[str]] = {}
    total = sum(len(v) for v in indices_by_bucket.values())
    done = 0

    rows: list[_BucketManifestRow] = []
    for bucket_idx in sorted(indices_by_bucket.keys()):
        indices = indices_by_bucket[bucket_idx]
        bucket: Bucket = bucket_set[bucket_idx]
        on_progress = _make_on_progress(progress_callback, done, total)
        latents = _encode_bucket(
            dataset=dataset,
            indices=indices,
            encoder=encoder,
            normalize=normalize,
            batch_size=batch_size,
            dtype=dtype,
            on_progress=on_progress,
        )
        done += len(indices)
        first_caption = _get_caption(dataset, indices[0])
        if first_caption is not None:
            has_captions = True
            captions_by_bucket[bucket_idx] = [str(_get_caption(dataset, i) or "") for i in indices]
        filename = f"bucket_{bucket_idx:04d}.safetensors"
        save_file(
            {"latents": latents.contiguous()},
            str(out / filename),
            metadata={
                "bucket_idx": str(bucket_idx),
                "width": str(bucket.width),
                "height": str(bucket.height),
            },
        )
        rows.append(
            _BucketManifestRow(
                bucket_idx=bucket_idx,
                width=bucket.width,
                height=bucket.height,
                count=len(indices),
                file=filename,
            )
        )

    manifest = LatentManifest(
        version=_MANIFEST_VERSION,
        dtype=str(dtype).replace("torch.", ""),
        scale_factor=float(encoder.scale_factor),
        downsample_factor=int(encoder.downsample_factor),
        latent_channels=int(encoder.latent_channels),
        buckets=rows,
        has_captions=has_captions,
    )
    (out / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    if has_captions:
        (out / "captions.json").write_text(
            json.dumps({str(k): v for k, v in captions_by_bucket.items()}, indent=2),
            encoding="utf-8",
        )
    if progress_callback is not None:
        progress_callback(total, total)
    return manifest


def _default_normalize(x: torch.Tensor) -> torch.Tensor:
    """``[0, 1]`` to ``[-1, 1]`` (SD / SDXL VAE convention)."""
    return x * 2.0 - 1.0


def _make_on_progress(
    cb: Callable[[int, int], None] | None,
    base: int,
    total: int,
) -> Callable[[int], None] | None:
    if cb is None:
        return None

    def inner(n: int) -> None:
        cb(base + n, total)

    return inner


def _get_caption(dataset: BucketedDataset, idx: int) -> str | None:
    source = getattr(dataset, "_source", None)
    if source is None:
        return None
    getter = getattr(source, "get_caption", None)
    if getter is None:
        return None
    value = getter(idx)
    return None if value is None else str(value)


def _encode_bucket(
    *,
    dataset: BucketedDataset,
    indices: list[int],
    encoder: VAEEncoder,
    normalize: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
    dtype: torch.dtype,
    on_progress: Callable[[int], None] | None,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        pixels = torch.stack([dataset[i]["image"] for i in batch_indices])
        pixels = normalize(pixels)
        with torch.inference_mode():
            latents = encoder.encode(pixels)
        chunks.append(latents.to(dtype=dtype, device="cpu"))
        if on_progress is not None:
            on_progress(start + len(batch_indices))
    return torch.cat(chunks, dim=0)


class BucketedLatentDataset(Dataset[dict[str, Any]]):
    """Read precomputed latents instead of raw images.

    Each item returns a dict with keys:

      - ``"latents"``: ``(latent_channels, H/f, W/f)`` tensor, dtype as
        stored on disk (typically bfloat16 or float32)
      - ``"bucket"``: the :class:`Bucket` the latent corresponds to
      - ``"bucket_idx"``: the bucket's index
      - ``"caption"``: optional, only present if precompute was given
        captions

    The dataset is fast to construct, latents are memory-mapped via
    safetensors so opening a directory of 100K precomputed latents is
    nearly free. Pair it with :class:`BucketBatchSampler` (built from
    the dataset's ``bucket_indices``) to keep batches single-bucket.

    Args:
        path: Directory produced by :func:`precompute_latents`.
        map_location: Device for the returned tensors. Default ``"cpu"``;
            tensors are moved per-call so this is cheap.

    Raises:
        FileNotFoundError: If ``path`` lacks a ``manifest.json``.
        ImportError: If safetensors is not installed.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> None:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "BucketedLatentDataset requires safetensors. "
                "Install with: pip install bucketsampler[vae]"
            ) from exc
        root = Path(path)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing manifest.json under {root}; "
                "was this directory produced by precompute_latents?"
            )
        self._root = root
        self.manifest = LatentManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        self._safe_open = safe_open
        self._map_location = map_location

        captions_path = root / "captions.json"
        self._captions: dict[int, list[str]] | None = None
        if self.manifest.has_captions and captions_path.is_file():
            raw = json.loads(captions_path.read_text(encoding="utf-8"))
            self._captions = {int(k): list(v) for k, v in raw.items()}

        bucket_idx: list[int] = []
        local_idx: list[int] = []
        for row in self.manifest.buckets:
            for j in range(row.count):
                bucket_idx.append(row.bucket_idx)
                local_idx.append(j)
        self.bucket_indices: np.ndarray = np.asarray(bucket_idx, dtype=np.int64)
        self._local_indices = np.asarray(local_idx, dtype=np.int64)
        self._by_idx = {row.bucket_idx: row for row in self.manifest.buckets}

    def __len__(self) -> int:
        return int(self.bucket_indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        bucket_idx = int(self.bucket_indices[idx])
        local = int(self._local_indices[idx])
        row = self._by_idx[bucket_idx]
        file_path = self._root / row.file
        opener: Any = self._safe_open
        with opener(str(file_path), framework="pt", device=str(self._map_location)) as f:
            tensor = f.get_slice("latents")[local : local + 1, :, :, :][0]
        sample: dict[str, Any] = {
            "latents": tensor,
            "bucket": Bucket(row.width, row.height),
            "bucket_idx": bucket_idx,
        }
        if self._captions is not None and bucket_idx in self._captions:
            sample["caption"] = self._captions[bucket_idx][local]
        return sample

    @property
    def bucket_set(self) -> BucketSet:
        """Reconstruct the bucket set from the manifest."""
        buckets = tuple(Bucket(row.width, row.height) for row in self.manifest.buckets)
        return BucketSet(
            buckets=buckets,
            name="precomputed",
            vae_factor=self.manifest.downsample_factor,
        )


def iter_latent_files(path: str | Path) -> Iterable[Path]:
    """Yield each per-bucket safetensors file under ``path``."""
    root = Path(path)
    yield from sorted(root.glob("bucket_*.safetensors"))
