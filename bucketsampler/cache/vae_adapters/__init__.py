"""VAE encoder protocol and reference adapters.

A :class:`VAEEncoder` is anything that maps a CHW pixel batch in
``[-1, 1]`` to a latent tensor. The protocol is intentionally tiny so
custom VAEs (Flux, exotic upstream models, in-house fine-tunes) plug
in by writing a single class. The bundled adapters live alongside but
are lazy-imported so the diffusers dependency stays optional.
"""

from __future__ import annotations

from bucketsampler.cache.vae_adapters.protocol import VAEEncoder

__all__ = ["VAEEncoder"]
