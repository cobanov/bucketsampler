"""Duck-typed VAE encoder interface.

Any object that exposes :attr:`downsample_factor`, :attr:`latent_channels`,
:attr:`scale_factor`, and :meth:`encode` is a valid encoder. The bundled
SD adapter wraps ``diffusers.AutoencoderKL``; tests use a lightweight
``FakeVAE`` that produces deterministic latents without loading any
real model.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class VAEEncoder(Protocol):
    """Encodes pixel batches into latent batches.

    Attributes (or read-only properties) on conforming objects:

      - ``downsample_factor``: how much each spatial dim shrinks. SD /
        SDXL VAEs downsample by 8 (``128x128`` latent from ``1024x1024``
        pixels). Used to derive per-bucket latent shapes ahead of time
        so storage can be preallocated.
      - ``latent_channels``: number of channels in the latent tensor.
        SD's VAE produces 4, SDXL's 4, Flux's 16, etc.
      - ``scale_factor``: the diffusers-convention scaling constant
        applied after encoding so latents live in ``~N(0, 1)``. Stored
        alongside the latents so the trainer can undo / reapply it.
    """

    @property
    def downsample_factor(self) -> int: ...

    @property
    def latent_channels(self) -> int: ...

    @property
    def scale_factor(self) -> float: ...

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode a pixel batch into latents.

        Args:
            pixels: ``(B, 3, H, W)`` float tensor in ``[-1, 1]``.

        Returns:
            ``(B, latent_channels, H // downsample_factor,
            W // downsample_factor)`` float tensor on the same device
            as ``pixels``, after the scale factor has been applied.
        """
        ...
