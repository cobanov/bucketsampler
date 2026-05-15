# bucketsampler

Aspect ratio bucketing toolkit for diffusion model training.

> Note: "bucket sampler" also exists in NLP for sequence-length bucketing. This
> project is specifically for image / diffusion training (SDXL-style multi-AR
> batches), not sequence data.

## Status

Milestone M1 (core bucketing logic) is implemented. PyTorch, HuggingFace,
caching, CLI analyzer, and VAE adapters are planned (see `PLAN.md`).

## Install

```bash
pip install bucketsampler           # core only
pip install bucketsampler[torch]    # adds PyTorch integration (M2+)
```

## Quickstart

```python
from bucketsampler import FixedBuckets, load_preset

strategy = FixedBuckets(load_preset("sdxl"))
bucket = strategy.assign(width=1280, height=720)
print(bucket)  # Bucket(width=1344, height=768)
```

## CLI

```bash
bucketsampler --help
bucketsampler presets       # list bundled presets
bucketsampler version
```

## License

MIT
