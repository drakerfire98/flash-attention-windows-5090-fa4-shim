"""Importable callable definitions for persistent native-plan probes."""

from __future__ import annotations


def striped_mask(batch_idx, head_idx, q_idx, kv_idx):
    del batch_idx
    return (kv_idx == 0) | ((kv_idx <= q_idx) & (((q_idx + kv_idx + head_idx) % 3) != 1))


def dense_score_bias(scores, batch_idx, head_idx, q_idx, kv_idx):
    del batch_idx
    return scores + (head_idx.to(scores.dtype) * 0.02) + ((q_idx - kv_idx).to(scores.dtype) * 0.01)
