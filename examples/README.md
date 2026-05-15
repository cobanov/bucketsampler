# Examples

Short, runnable snippets that show common bucketsampler workflows.

| Script | What it shows |
|--------|---------------|
| [`minimal_training_loop.py`](minimal_training_loop.py) | End-to-end loop: paths → `BucketedDataset` → `BucketBatchSampler` → DataLoader. Replace the fake U-Net with your model. |
| [`auto_buckets_inline.py`](auto_buckets_inline.py) | Derive buckets from your dataset's actual aspect-ratio distribution without writing any TOML. |
| [`hf_dataset.py`](hf_dataset.py) | Pull from a `datasets.Dataset` (PIL or tensor columns). |
| [`precompute_latents.py`](precompute_latents.py) | Cache VAE latents once, train without paying the VAE forward pass every step. |
| [`ddp_training.py`](ddp_training.py) | Same sampler with `num_replicas` / `rank` / `set_epoch` for DDP. |
