"""Tests for bucketsampler.torch.sampler.

Most tests construct the sampler with a hand-rolled ``bucket_indices`` array
so they don't have to materialize images. Real-image integration runs in
``tests/torch/test_dataset.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from bucketsampler.torch import BucketBatchSampler


def _flatten(batches):
    out = []
    for b in batches:
        out.extend(b)
    return out


class TestValidation:
    def test_batch_size_must_be_positive(self):
        with pytest.raises(ValueError):
            BucketBatchSampler(np.array([0, 0, 1]), batch_size=0)

    def test_num_replicas_must_be_positive(self):
        with pytest.raises(ValueError):
            BucketBatchSampler(np.array([0, 0, 1]), batch_size=2, num_replicas=0)

    def test_rank_must_be_in_range(self):
        with pytest.raises(ValueError):
            BucketBatchSampler(np.array([0, 0, 1]), batch_size=2, num_replicas=2, rank=2)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            BucketBatchSampler(np.array([[0, 1], [2, 3]]), batch_size=2)

    def test_negative_indices_raise(self):
        with pytest.raises(ValueError):
            BucketBatchSampler(np.array([0, -1, 1]), batch_size=2)

    def test_accepts_sequence(self):
        s = BucketBatchSampler([0, 0, 1, 1], batch_size=2)
        assert s.bucket_indices.dtype == np.int64

    def test_accepts_dataset_like(self):
        class FakeDataset:
            bucket_indices = np.array([0, 0, 1, 1], dtype=np.int64)

        s = BucketBatchSampler(FakeDataset(), batch_size=2)
        assert s.bucket_indices.tolist() == [0, 0, 1, 1]


class TestBatching:
    def test_each_batch_same_bucket(self):
        # 10 in bucket 0, 6 in bucket 1
        idx = np.array([0] * 10 + [1] * 6)
        s = BucketBatchSampler(idx, batch_size=2, shuffle=False, drop_last=False)
        for batch in s:
            assigned = {int(idx[i]) for i in batch}
            assert len(assigned) == 1

    def test_drop_last_drops_partial(self):
        # 7 items in bucket 0, batch_size 3, drop_last=True -> 2 batches (6 items)
        idx = np.zeros(7, dtype=np.int64)
        s = BucketBatchSampler(idx, batch_size=3, shuffle=False, drop_last=True)
        batches = list(s)
        assert len(batches) == 2
        assert all(len(b) == 3 for b in batches)

    def test_no_drop_last_yields_partial(self):
        idx = np.zeros(7, dtype=np.int64)
        s = BucketBatchSampler(idx, batch_size=3, shuffle=False, drop_last=False)
        batches = list(s)
        assert len(batches) == 3
        assert sorted(len(b) for b in batches) == [1, 3, 3]

    def test_len_matches_iter(self):
        idx = np.array([0] * 10 + [1] * 6 + [2] * 4)
        for drop in (True, False):
            s = BucketBatchSampler(idx, batch_size=3, shuffle=True, drop_last=drop)
            assert len(s) == len(list(s))

    def test_all_indices_appear_when_not_dropping(self):
        idx = np.array([0] * 4 + [1] * 5)
        s = BucketBatchSampler(idx, batch_size=2, shuffle=False, drop_last=False)
        seen = _flatten(list(s))
        assert sorted(seen) == list(range(len(idx)))

    def test_empty_indices(self):
        s = BucketBatchSampler(np.array([], dtype=np.int64), batch_size=2)
        assert len(s) == 0
        assert list(s) == []

    def test_repr(self):
        s = BucketBatchSampler(np.array([0, 0, 1, 1]), batch_size=2)
        r = repr(s)
        assert "BucketBatchSampler" in r
        assert "batch_size=2" in r


class TestDeterminism:
    def test_same_seed_same_epoch_same_batches(self):
        idx = np.array([0] * 10 + [1] * 8 + [2] * 6)
        s1 = BucketBatchSampler(idx, batch_size=2, shuffle=True, seed=123)
        s2 = BucketBatchSampler(idx, batch_size=2, shuffle=True, seed=123)
        assert list(s1) == list(s2)

    def test_different_epoch_different_order(self):
        idx = np.array([0] * 10 + [1] * 8)
        s = BucketBatchSampler(idx, batch_size=2, shuffle=True, seed=0)
        s.set_epoch(0)
        first = list(s)
        s.set_epoch(1)
        second = list(s)
        # Order should change (overwhelmingly likely with 5+ batches)
        assert first != second
        # But the multiset of indices should be the same (no losses without DDP)
        assert sorted(_flatten(first)) == sorted(_flatten(second))

    def test_set_epoch_is_deterministic(self):
        idx = np.array([0] * 12 + [1] * 6)
        s = BucketBatchSampler(idx, batch_size=3, shuffle=True, seed=42)
        s.set_epoch(7)
        out1 = list(s)
        s.set_epoch(7)
        out2 = list(s)
        assert out1 == out2


class TestDDPSlicing:
    def test_ranks_yield_disjoint_indices(self):
        idx = np.array([0] * 12 + [1] * 8 + [2] * 6)
        # 3 ranks, batch_size 2, drop_last True
        ranks = []
        for r in range(3):
            s = BucketBatchSampler(
                idx,
                batch_size=2,
                shuffle=True,
                drop_last=True,
                num_replicas=3,
                rank=r,
                seed=0,
            )
            ranks.append(set(_flatten(list(s))))
        # No overlap between ranks
        assert ranks[0].isdisjoint(ranks[1])
        assert ranks[0].isdisjoint(ranks[2])
        assert ranks[1].isdisjoint(ranks[2])

    def test_ranks_yield_equal_batch_count(self):
        # Equal batch count is required for DDP gradient sync
        idx = np.array([0] * 11 + [1] * 9 + [2] * 7)  # not divisible by 3
        counts = []
        for r in range(3):
            s = BucketBatchSampler(
                idx,
                batch_size=2,
                shuffle=True,
                drop_last=True,
                num_replicas=3,
                rank=r,
                seed=42,
            )
            counts.append(len(list(s)))
        assert len(set(counts)) == 1

    def test_world_size_one_equivalent_to_no_ddp(self):
        idx = np.array([0] * 10 + [1] * 5)
        s_ddp = BucketBatchSampler(idx, batch_size=2, num_replicas=1, rank=0, seed=99)
        s_plain = BucketBatchSampler(idx, batch_size=2, seed=99)
        assert list(s_ddp) == list(s_plain)

    def test_len_reflects_per_rank(self):
        idx = np.array([0] * 11 + [1] * 9 + [2] * 7)
        for r in range(3):
            s = BucketBatchSampler(
                idx,
                batch_size=2,
                drop_last=True,
                num_replicas=3,
                rank=r,
                seed=0,
            )
            assert len(s) == len(list(s))
