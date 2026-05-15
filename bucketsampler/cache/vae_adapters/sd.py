"""SD / SDXL VAE adapter on top of :mod:`diffusers`.

Wraps ``diffusers.AutoencoderKL`` so that any SD-family model identifier
(``stabilityai/sdxl-vae``, ``runwayml/stable-diffusion-v1-5``,
``madebyollin/sdxl-vae-fp16-fix``, etc.) can be used as a
:class:`VAEEncoder`. The model is loaded lazily on first use so the
import-time cost is just discovering the class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass


class SDVAEEncoder:
    """Diffusers-backed VAE encoder for the SD / SDXL family.

    Args:
        model_id: HuggingFace repo id (e.g. ``"stabilityai/sdxl-vae"``)
            or local path that ``AutoencoderKL.from_pretrained`` accepts.
        device: Target torch device. Defaults to CUDA if available else CPU.
        dtype: Computation dtype. ``torch.bfloat16`` halves storage and
            usually preserves quality on SDXL-class models.
        scale_factor: Override the model's default scaling constant. The
            SD VAE stores its own ``config.scaling_factor`` (0.18215 for
            SD1.5, 0.13025 for SDXL); pass ``None`` to use that.

    Raises:
        ImportError: If diffusers is not installed.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        scale_factor: float | None = None,
    ) -> None:
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise ImportError(
                "SDVAEEncoder requires diffusers. Install with: pip install bucketsampler[vae]"
            ) from exc
        self.model_id = model_id
        self._model_cls = AutoencoderKL
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = dtype
        self._override_scale = scale_factor
        self._vae: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._vae is None:
            vae = self._model_cls.from_pretrained(self.model_id, torch_dtype=self._dtype)
            vae.eval()
            vae.requires_grad_(False)
            vae.to(self._device)
            self._vae = vae
        return self._vae

    @property
    def downsample_factor(self) -> int:
        return 8

    @property
    def latent_channels(self) -> int:
        vae = self._ensure_loaded()
        return int(vae.config.latent_channels)

    @property
    def scale_factor(self) -> float:
        if self._override_scale is not None:
            return float(self._override_scale)
        vae = self._ensure_loaded()
        return float(vae.config.scaling_factor)

    @torch.inference_mode()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        vae = self._ensure_loaded()
        pixels = pixels.to(device=self._device, dtype=self._dtype)
        out: torch.Tensor = vae.encode(pixels).latent_dist.mean
        return out * self.scale_factor
