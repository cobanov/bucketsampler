# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working on this repository.

## Project context

`bucketsampler` is an aspect ratio bucketing toolkit for diffusion model training. It bridges raw image datasets and PyTorch/HuggingFace training pipelines, ensuring batches contain images with matching dimensions without distortion or data loss.

The core problem: diffusion U-Nets require fixed (H, W) per batch, but real datasets have varied aspect ratios. Naive solutions (center-crop, squeeze) lose information or distort. Bucketing groups images by AR into pre-defined targets, with each batch drawn from one bucket.

**Naming note**: "bucket sampler" also exists in NLP for sequence-length bucketing. This project is specifically for image / diffusion training. Keep that framing in docs and examples.

## Tech stack

- **Python**: 3.10+ (use PEP 604 union syntax `X | Y`, structural pattern matching where it helps)
- **Core deps**: `torch>=2.1`, `numpy`, `pillow`, `pydantic>=2` for config models
- **Optional deps** (extras): `datasets` for HF, `diffusers` for VAE adapters, `safetensors` for latent storage, `pyarrow` for parquet caches
- **Dev tools**: `ruff` (lint + format), `mypy` (strict), `pytest`, `pytest-cov`
- **Build**: `uv` for env management, `hatchling` for packaging

## Code style

### Formatting
- Line length: 100
- Ruff with rules: E, F, I, N, UP, B, SIM, RUF
- Run `ruff format` and `ruff check --fix` before committing
- No em dashes anywhere in code, docstrings, or comments

### Typing
- Type hints on **all** public APIs (functions, methods, dataclass fields)
- Internal helpers: type hints encouraged, not enforced
- `mypy --strict` must pass on `bucketsampler/` (tests can be looser)
- Use `Protocol` for duck-typed interfaces (e.g. VAE encoder)
- `Annotated` for constrained types where useful

### Docstrings
- Google style
- Public APIs: required, with examples for non-trivial ones
- Private (`_prefixed`): optional, only if non-obvious
- No filler ("This function returns the result")

### Naming
- Snake_case for functions, variables, modules
- PascalCase for classes
- SCREAMING_SNAKE for module-level constants
- Single-letter vars only for math (`x`, `y`, `w`, `h`) or loop indices

## Architecture principles

### Framework separation
`core/` must not import torch. It's pure Python + numpy. PyTorch-specific code lives in `torch/`. HuggingFace-specific in `hf/`. This makes the core testable without GPU and reusable.

### Lazy evaluation
Don't read image bytes for dimension lookup. Use PIL's lazy header read (`Image.open(path).size` without `.load()`). For repeat access, use the metadata cache.

### Determinism
Same seed + same dataset + same bucket set = same batches. Verified by tests. Critical for reproducibility in research.

### Immutability for value types
`Bucket` and `BucketSet` are frozen dataclasses. Mutation happens at the dataset/cache layer, not in core types.

### Errors with context
Custom exceptions in `bucketsampler/exceptions.py`. When raising, include actionable info: image path, expected bucket, AR distance, etc. Don't just `raise ValueError("bad image")`.

```python
# bad
raise ValueError("image too small")

# good
raise ImageTooSmallError(
    path=path,
    actual=(w, h),
    required=(bucket.width, bucket.height),
    suggestion="Filter dataset or add smaller bucket to BucketSet",
)
```

## Testing conventions

- `pytest` with `pytest-cov`
- Tests mirror source layout: `bucketsampler/core/bucket.py` → `tests/core/test_bucket.py`
- Use `pytest.fixture` for shared setup, `tmp_path` for filesystem tests
- Synthetic images via PIL for unit tests (don't ship test images in repo)
- Integration tests in `tests/integration/`, marked with `@pytest.mark.integration`
- Coverage target: 100% on `core/`, 80%+ on the rest
- DDP tests use `torch.multiprocessing.spawn` with 2 fake ranks on CPU

## Performance discipline

This is a data utility on the hot path of training. Performance matters.

- **Profile before optimizing**: use `py-spy` or `cProfile`, don't guess
- **Avoid per-image Python overhead in tight loops**: batch operations, vectorize with numpy
- **I/O parallelism**: use `concurrent.futures.ThreadPoolExecutor` for image dim reads (I/O bound)
- **Memory**: a 1M-image metadata cache should fit in under 200MB (use int32 for dims, not int64)
- **Benchmarks**: `tests/benchmarks/` directory with `pytest-benchmark`; flag regressions in CI

## Common workflows for Claude Code

### Adding a new bucket strategy
1. Implement in `bucketsampler/core/strategies.py` (or a sibling module for data-derived strategies) as a class with `assign(width, height) -> Bucket` and `assign_many_indices(dims) -> np.ndarray`
2. Add to the `Strategy` protocol if introducing a new interface method
3. Unit tests for assignment correctness + edge cases

### Adding a CLI command
1. Module under `bucketsampler/cli/` (one command per file)
2. Register in `bucketsampler/cli/__init__.py` via the dispatcher
3. Use `typer` for argument parsing (we prefer it over argparse for typing)
4. Always support `--json` output for machine readability
5. Integration test under `tests/cli/`

### Adding a VAE adapter
1. Create `bucketsampler/cache/vae_adapters/<name>.py`
2. Implement the `VAEEncoder` protocol (`encode(images) -> latents`)
3. Document the model identifier and expected scale_factor in module docstring
4. Don't require the heavy dep at import time (lazy import inside `__init__`)

## DDP gotchas to watch for

When working on `BucketBatchSampler` or anything distributed:

- `set_epoch(epoch)` must reseed deterministically; without it, all epochs see same order
- Each rank's batches must be disjoint across all buckets
- `drop_last` semantics: per-bucket-per-rank, not global
- World size 1 must still work (no DDP) without special-casing the user's code
- Don't call `dist.barrier()` inside the sampler (deadlocks); barriers belong in training scripts

## What NOT to do

- Don't add training logic to this repo (no train loops, no model definitions)
- Don't depend on a specific diffusion framework as a hard dep (diffusers is optional)
- Don't write image augmentations (resize-and-crop for bucketing is the only transform)
- Don't add network I/O to core/ (filesystem only; cloud storage as optional extra later)
- Don't use `print()` for user output; use `logging` or the CLI's rich output layer
- Don't catch broad exceptions silently; either handle specifically or let propagate
- Don't add em dashes in any text output, error messages, or docstrings
- Don't conflate this with NLP sequence-length bucket samplers in docs or examples

## File organization

```
bucketsampler/
├── __init__.py              # public API re-exports
├── exceptions.py            # all custom exceptions
├── core/
│   ├── __init__.py
│   ├── bucket.py            # Bucket, BucketSet
│   ├── strategies.py        # FixedBuckets, AutoBuckets
│   ├── assignment.py        # matching algorithm
│   └── stats.py             # distribution analysis
├── torch/
│   ├── __init__.py
│   ├── dataset.py
│   ├── sampler.py
│   └── transforms.py
├── hf/
├── cache/
├── cli/
└── presets/                 # built-in bucket sets as TOML
    ├── sdxl.toml
    ├── sd15.toml
    └── novelai.toml
```

Public API surface (importable from top level):
```python
from bucketsampler import (
    Bucket,
    BucketSet,
    FixedBuckets,
    AutoBuckets,
    BucketedDataset,
    BucketBatchSampler,
    load_preset,
)
```

Everything else is `from bucketsampler.core.assignment import ...` style.

## Commit conventions

Conventional commits with these scopes:
- `feat(core):` new functionality in framework-agnostic layer
- `feat(torch):` PyTorch-specific features
- `feat(hf):` HuggingFace integration
- `feat(cli):` CLI changes
- `fix(...):` bug fixes
- `perf(...):` performance improvements with benchmark numbers in body
- `docs:` README, docstrings, examples
- `test:` test-only changes
- `chore:` deps, CI, build config

Body should explain **why**, not just what. Performance commits must include benchmark deltas.

## When in doubt

1. Check whether the feature matches the project's non-goals (no training loops, no augmentations, no network I/O in core)
2. Prefer simple, well-tested code over clever optimization
3. Ask before introducing new dependencies (especially heavy ones)
4. Frame error messages from the user's perspective, not the implementation's