# bucketsampler - Project Plan

> Aspect ratio bucketing toolkit for diffusion model training. PyTorch native, HuggingFace datasets compatible.

## Vision

A production-ready, well-tested Python package that handles every aspect of aspect-ratio bucketing for diffusion training. Drop-in replacement for naive `DataLoader` workflows. Used by researchers and indie trainers who want SDXL-style multi-AR training without writing the plumbing themselves.

## Goals

1. **Correctness first**: bucketing logic should never silently distort or lose data
2. **Performance**: handle 1M+ image datasets without choking
3. **Developer ergonomics**: 3-line integration into existing training scripts
4. **Reproducibility**: deterministic given a seed, exportable bucket assignments
5. **Observability**: rich dataset analyzer to inspect AR distribution, bucket utilization, edge cases

## Non-goals

- Image augmentation library (use albumentations/torchvision)
- Training framework (this is a data utility)
- Storage backend (uses filesystem, not S3/GCS, though latent cache could be remote)

## Architecture

```
bucketsampler/
├── core/                    # framework-agnostic
│   ├── bucket.py            # Bucket, BucketSet dataclasses
│   ├── strategies.py        # FixedBuckets, AutoBuckets (k-means)
│   ├── assignment.py        # image -> bucket matching logic
│   └── stats.py             # AR distribution analysis
├── torch/                   # PyTorch integration
│   ├── dataset.py           # BucketedDataset wrapper
│   ├── sampler.py           # BucketBatchSampler (per-bucket batches)
│   └── transforms.py        # resize+crop to exact bucket dims
├── hf/                      # HuggingFace integration
│   └── adapter.py           # works with datasets.Dataset
├── cache/                   # caching layer
│   ├── metadata.py          # dimensions, AR, bucket assignments
│   └── latents.py           # VAE latent precomputation
├── cli/                     # command-line tools
│   ├── analyze.py           # bucketsampler analyze <path>
│   ├── precompute.py        # bucketsampler precompute --vae ...
│   └── inspect.py           # bucketsampler inspect --bucket 768x512
└── tests/
```

## Core concepts

### Bucket
A single (width, height) target. Has a target pixel count and aspect ratio.

```python
@dataclass(frozen=True)
class Bucket:
    width: int
    height: int

    @property
    def aspect_ratio(self) -> float: ...
    @property
    def pixel_count(self) -> int: ...
```

### BucketSet
A collection of buckets with assignment logic. Two flavors:
- `FixedBucketSet`: user-defined list (SDXL preset, NovelAI preset, custom)
- `AutoBucketSet`: derived from dataset via k-means on log-AR, constrained to multiples of N (usually 64 for VAE-friendly dims)

### Assignment
Given an image's native dims, pick the closest bucket. "Closest" by minimum AR distance (in log space, since AR is multiplicative). Resize so the longer dim fits, then minimal center crop.

### BucketBatchSampler
Custom PyTorch sampler. Maintains per-bucket index queues. Each batch comes entirely from one bucket. Shuffles within bucket per epoch. Handles DDP via `set_epoch()` + rank-aware slicing.

## Milestones

### M1: Core bucketing logic (week 1)

- [ ] `Bucket`, `BucketSet` dataclasses
- [ ] Fixed presets: SDXL, NovelAI, SD1.5, custom (TOML/JSON loader)
- [ ] Assignment algorithm with unit tests (edge cases: square, ultrawide, tall)
- [ ] CLI scaffolding: `bucketsampler --help`

**Done when**: can take a list of (w, h) tuples and a bucket set, return assignments. 100% test coverage on core/.

### M2: PyTorch integration (week 1-2)

- [ ] `BucketedDataset`: wraps any Dataset returning (image, ...), reads dims lazily or from cache
- [ ] `BucketBatchSampler`: per-bucket batches, shuffle support, drop_last semantics
- [ ] `BucketResize` transform: resize-then-crop to exact bucket dims
- [ ] DDP support: rank slicing inside sampler
- [ ] Integration test with synthetic dataset

**Done when**: minimal training loop runs end-to-end on synthetic data, each batch has consistent dims, DDP works on 2 GPUs.

### M3: Dataset analyzer (week 2)

- [ ] `bucketsampler analyze <path>` CLI command
- [ ] Reports:
  - Total images, broken/unreadable count
  - AR histogram (log-scale)
  - Per-bucket count (using a given BucketSet)
  - Underutilized buckets (warn if < N images)
  - Outliers (extreme ARs that match no bucket well)
- [ ] HTML report option with matplotlib plots
- [ ] JSON output for piping into other tools

**Done when**: pointed at a folder of 10K images, produces a clear report in under 60s.

### M4: Auto-bucket generation (week 2-3)

- [ ] k-means on log-AR distribution from dataset
- [ ] Constraint: dims must be multiples of N (64 default for VAE)
- [ ] Constraint: target pixel budget (e.g. ~512² or ~1024²)
- [ ] Output as TOML config that can be loaded back
- [ ] CLI: `bucketsampler buckets-from-dataset <path> --target 512 --num 8`

**Done when**: given a dataset, produces a bucket set that's measurably better (less crop loss) than generic presets.

### M5: HuggingFace adapter (week 3)

- [ ] `BucketedDataset.from_hf(hf_dataset, image_column="image")`
- [ ] Streaming dataset support (for large remote datasets)
- [ ] Works with both PIL and tensor columns
- [ ] Example notebook: train on a HF diffusion dataset

**Done when**: a HF example dataset (e.g. lambdalabs/pokemon-blip-captions) works without manual conversion.

### M6: Metadata caching (week 3-4)

- [ ] Precompute (width, height, bucket_id) for all images, store in parquet/sqlite
- [ ] Skip re-reading images for dim lookup (huge speedup)
- [ ] Cache invalidation on dataset changes (hash file paths + mtimes)
- [ ] CLI: `bucketsampler build-cache <path>`

**Done when**: a 100K image dataset loads buckets in seconds instead of minutes.

### M7: Latent precomputation (week 4)

- [ ] Generic VAE interface (encoder callable)
- [ ] Adapters for SD/SDXL/Flux VAE via `diffusers`
- [ ] Multi-GPU latent precompute with progress bar
- [ ] Store latents per-bucket in `.safetensors` or memory-mapped arrays
- [ ] `BucketedLatentDataset`: returns precomputed latents directly
- [ ] CLI: `bucketsampler precompute --vae stabilityai/sdxl-vae --dataset <path>`

**Done when**: training speed measurably improves (no VAE forward in train loop), pointed at 50K images runs in reasonable time on multi-GPU.

### M8: Polish & release (week 4-5)

- [ ] README with quickstart
- [ ] Examples directory: minimal training, SDXL fine-tune, custom bucket set
- [ ] Benchmark suite (vs naive DataLoader, vs diffusers' default)
- [ ] PyPI release as `bucketsampler`
- [ ] GitHub Actions: CI tests on Python 3.10/3.11/3.12, lint, type-check

## Technical decisions

### Bucket matching algorithm
Match by minimum **log-aspect-ratio distance**, not by pixel count. AR is the visual property that matters; pixel budget is enforced by bucket design, not assignment.

```python
def best_bucket(img_w, img_h, buckets):
    img_log_ar = math.log(img_w / img_h)
    return min(buckets, key=lambda b: abs(math.log(b.width / b.height) - img_log_ar))
```

### Resize strategy
**Fit-longer-dim, then crop**. Resize so that the image is at least as large as the bucket on both axes, then center-crop the excess. Never upscale-then-crop (introduces blur) unless image is genuinely smaller than bucket (warn the user).

### Bucket dimensions
Always multiples of **vae_factor** (default 64 for SD-family VAEs, 16 for some new ones). Total pixels target with `latent_size² × vae_factor²` math; e.g. SDXL targets ~128² latents × 8² VAE = ~1024² pixels.

### Sampling fairness
Default: weighted by bucket size (a bucket with 10K images is seen more than one with 100). Optional `balanced=True` for uniform bucket sampling, useful for rare-AR generation.

### DDP correctness
Each rank gets a disjoint slice of each bucket. `set_epoch(epoch)` reseeds the per-bucket shuffles. drop_last=True per bucket per rank to keep batches uniform.

## Naming disambiguation

The PyTorch ecosystem already has the term "bucket sampler" in NLP contexts (length-based bucketing for variable sequence length, e.g. torchnlp). To differentiate:

- README must lead with "for diffusion / image training" framing
- Top-level class is `BucketBatchSampler` (not just `BucketSampler`)
- All examples should use image data, not sequence data
- Tagline emphasizes aspect ratio specifically: "Aspect ratio bucketing for diffusion training"

## Risks and unknowns

- **Name collision with NLP bucket samplers**: handled via positioning and naming above
- **Streaming datasets + bucketing**: can't bucket what you can't measure ahead of time. Solution: lazy bucketing with a small in-memory queue per bucket, accept some warmup cost
- **Very long-tail AR distributions**: real-world data has weird outliers (1px wide images, etc). Need robust outlier filtering
- **Mixed precision interactions**: latent cache should store bf16 (4x smaller than fp32), need to validate quality impact
- **Auto-bucket quality**: k-means in log-AR may not be optimal, consider weighted variants or median-based clustering

## Success metrics

1. **Adoption signal**: ≥ 50 PyPI installs/week within 2 months of release
2. **Correctness**: 100% test coverage on core/, integration tests for sampler + DDP
3. **Performance**: 100K-image dataset analysis under 60s, latent precompute saturates GPU
4. **DX**: integration into existing diffusers training script takes under 10 LOC change

## Future ideas (post-v1)

- Adaptive bucket reassignment during training (rare ARs get promoted)
- Conditional bucketing (per-class buckets for class-conditional models)
- Multi-resolution training schedule (start low-res, ramp up)
- Direct integration with kohya_ss, OneTrainer, etc.
- Web UI for dataset inspection
- Cloud-native: S3/GCS latent storage with smart prefetch