# bucketsampler

> Aspect ratio bucketing for diffusion model training (SDXL-style multi-AR
> batches). PyTorch native, DDP correct, zero training-framework lock-in.

> Heads up: "bucket sampler" also names a length-bucketing pattern in NLP.
> This is the image / diffusion variant, not the sequence one.

## Why

Diffusion U-Nets want a fixed `(H, W)` per batch. Real datasets do not. The
naive options either distort (squeeze every image to a square) or throw data
away (center-crop to the smallest common size). Bucketing splits images into a
small set of `(W, H)` targets and draws each batch from a single bucket, so
nothing gets squished and nothing gets dropped.

`bucketsampler` ships the plumbing: assignment, dataset wrapper, DDP-correct
sampler, presets for SDXL / SD1.5 / NovelAI, and a CLI to inspect your data
before you start training.

## Install

```bash
pip install bucketsampler             # core (no torch)
pip install "bucketsampler[torch]"    # + PyTorch integration
pip install "bucketsampler[analyze]"  # + HTML reports from the analyzer
```

## 30-second quickstart

```python
from pathlib import Path
from torch.utils.data import DataLoader
from bucketsampler import (
    BucketBatchSampler,
    BucketedDataset,
    FixedBuckets,
    load_preset,
)

paths = sorted(Path("data/").glob("*.jpg"))
strategy = FixedBuckets(load_preset("sdxl"))

dataset = BucketedDataset(paths=paths, strategy=strategy)
sampler = BucketBatchSampler(dataset, batch_size=4)
loader = DataLoader(dataset, batch_sampler=sampler)

for batch in loader:
    images = batch["image"]   # [4, 3, H, W], same (H, W) within a batch
    buckets = batch["bucket"] # list[Bucket], one per sample
    # ... feed images to your VAE / U-Net / etc.
```

## Inspect your dataset

Before you commit to a bucket set, see how your images actually distribute:

```bash
bucketsampler analyze data/ --preset sdxl
bucketsampler analyze data/ --preset sdxl --json > report.json
bucketsampler analyze data/ --preset sdxl --html report.html
```

The report shows readable / broken counts, AR distribution, per-bucket counts,
underutilized buckets (so you know what to drop), and outliers (extreme ARs
that match no bucket well, often a sign of bad data).

## DDP

Same sampler, two extra kwargs:

```python
sampler = BucketBatchSampler(
    dataset,
    batch_size=4,
    num_replicas=world_size,
    rank=rank,
)
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)   # required, reseeds the per-bucket shuffle
    for batch in DataLoader(dataset, batch_sampler=sampler):
        ...
```

All ranks yield the same number of batches per epoch and see disjoint
indices, so gradient sync stays happy.

## Auto-generate buckets from your dataset

Presets are reasonable defaults, but a bucket set picked from *your*
distribution always crops less. Let `bucketsampler` derive one for you:

```bash
bucketsampler buckets-from-dataset data/ \
    --num 8 --target 1024 \
    --compare-to sdxl \
    --output my_buckets.toml
```

Sample output:

```
Generated 8 buckets (requested 8, k-means converged in 6 iterations):
  640x1536      AR= 0.417  cluster log-AR=-0.875  size=120
  768x1280      AR= 0.600  cluster log-AR=-0.511  size=210
  ...
Mean crop loss (auto):       3.42%
Mean crop loss (sdxl):       7.18%  (auto is -3.76 pts vs sdxl)

Wrote bucket set to my_buckets.toml
```

Then feed the TOML straight back into your training script:

```python
from bucketsampler import FixedBuckets, load_from_toml

strategy = FixedBuckets(load_from_toml("my_buckets.toml"))
```

Or do it inline (no file involved) with the `AutoBuckets` convenience:

```python
import numpy as np
from bucketsampler import AutoBuckets

# dims = (N, 2) int array of (width, height); analyzer's scan returns this
strategy = AutoBuckets.from_dims(dims, num_buckets=8, target=1024)
```

The algorithm is 1-D k-means on `log(width / height)`, then snaps each
cluster center to a `(w, h)` whose product is close to `target^2` and
whose dims are multiples of `vae_factor` (default 64). Fully deterministic
for a given seed.

## Custom buckets

Drop a TOML file anywhere on disk:

```toml
# my_buckets.toml
name = "my-budget"
vae_factor = 8

[[buckets]]
width  = 768
height = 768

[[buckets]]
width  = 896
height = 640

[[buckets]]
width  = 640
height = 896
```

```python
from bucketsampler import FixedBuckets, load_from_toml

strategy = FixedBuckets(load_from_toml("my_buckets.toml"))
```

JSON is also supported via `load_from_json`. The bundled presets
(`sdxl`, `sd15`, `novelai`) live in the same format.

## CLI cheatsheet

```bash
bucketsampler --help
bucketsampler version
bucketsampler presets [--json]
bucketsampler analyze <path> --preset sdxl [--json | --html report.html]
bucketsampler buckets-from-dataset <path> --num 8 --target 1024 [--output buckets.toml] [--compare-to sdxl]
```

## Status

- [x] **M1** Core bucketing (`Bucket`, `BucketSet`, assignment, presets)
- [x] **M2** PyTorch integration (`BucketedDataset`, `BucketBatchSampler`, DDP)
- [x] **M3** Dataset analyzer CLI (`bucketsampler analyze`)
- [x] **M4** Auto-bucket generation (`AutoBuckets`, `buckets-from-dataset`)
- [ ] **M5** HuggingFace datasets adapter
- [ ] **M6** Metadata cache (parquet / sqlite)
- [ ] **M7** VAE latent precomputation
- [ ] **M8** Polish, examples, PyPI release

See [`PLAN.md`](PLAN.md) for the full roadmap.

## License

MIT
