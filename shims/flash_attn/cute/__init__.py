"""Stable Windows shim for the `flash_attn.cute` public API.

This does not implement native FA4 kernels. It provides a numerically stable
fallback for the public forward entrypoints using PyTorch attention ops while
keeping the same import path.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

__version__ = "0.0.0-windows-shim"


def _normalize_window_size(
    window_size: Tuple[Optional[int], Optional[int]],
) -> Tuple[Optional[int], Optional[int]]:
    left, right = window_size
    left = None if left is None or left < 0 else int(left)
    right = None if right is None or right < 0 else int(right)
    return left, right


def _expand_kv_for_q(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if q_heads == kv_heads:
        return k, v
    if q_heads % kv_heads != 0:
        raise ValueError(
            f"q heads ({q_heads}) must equal or be divisible by kv heads ({kv_heads})"
        )
    repeat = q_heads // kv_heads
    return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)


def _build_attention_mask(
    seqlen_q: int,
    seqlen_k: int,
    causal: bool,
    window_size: Tuple[Optional[int], Optional[int]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    left, right = _normalize_window_size(window_size)
    if not causal and left is None and right is None:
        return None

    q_idx = torch.arange(seqlen_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seqlen_k, device=device).unsqueeze(0)
    relative = k_idx - (q_idx + seqlen_k - seqlen_q)

    allowed = torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=device)
    if causal:
        allowed &= relative <= 0
    if left is not None:
        allowed &= relative >= -left
    if right is not None:
        allowed &= relative <= right
    return allowed


def _check_supported_common(
    learnable_sink: Optional[torch.Tensor],
    mask_mod: Optional[Callable],
    softcap: float,
) -> None:
    if mask_mod is not None:
        raise NotImplementedError("mask_mod/score_mod is not supported by the Windows shim")
    if softcap < 0:
        raise ValueError("softcap must be >= 0")


def _normalize_learnable_sink(
    learnable_sink: Optional[torch.Tensor],
    num_heads: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if learnable_sink is None:
        return None
    if learnable_sink.ndim != 1 or learnable_sink.shape[0] != num_heads:
        raise ValueError(f"learnable_sink must have shape ({num_heads},)")
    if learnable_sink.device != device:
        learnable_sink = learnable_sink.to(device=device)
    return learnable_sink.to(dtype=torch.float32).view(1, num_heads, 1, 1)


def _attention_forward_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float],
    causal: bool,
    window_size: Tuple[Optional[int], Optional[int]],
    learnable_sink: Optional[torch.Tensor],
    softcap: float,
    return_lse: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("dense flash_attn_func expects q, k, v shaped (batch, seqlen, heads, dim)")

    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    k_t, v_t = _expand_kv_for_q(q_t, k_t, v_t)
    sink = _normalize_learnable_sink(learnable_sink, q_t.shape[1], q.device)

    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    attn_mask = _build_attention_mask(
        seqlen_q=q.shape[1],
        seqlen_k=k.shape[1],
        causal=causal,
        window_size=window_size,
        device=q.device,
    )

    if softcap == 0.0 and not return_lse and sink is None:
        if causal and _normalize_window_size(window_size) == (None, None):
            use_native_causal = True
            attn_mask_for_sdpa = None
        else:
            use_native_causal = False
            attn_mask_for_sdpa = None if attn_mask is None else attn_mask.view(1, 1, q.shape[1], k.shape[1])

        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask_for_sdpa,
            dropout_p=0.0,
            is_causal=use_native_causal,
            scale=scale,
        )
        return out.permute(0, 2, 1, 3).contiguous(), None

    scores = torch.matmul(q_t.float(), k_t.float().transpose(-1, -2)) * scale
    if softcap > 0.0:
        scores = torch.tanh(scores / softcap) * softcap
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.view(1, 1, q.shape[1], k.shape[1]), float("-inf"))
    if sink is None:
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        lse = torch.logsumexp(scores, dim=-1).permute(0, 2, 1).contiguous() if return_lse else None
    else:
        logits_max = torch.amax(scores, dim=-1, keepdim=True)
        logits_or_sinks_max = torch.maximum(logits_max, sink)
        unnormalized_scores = torch.exp(scores - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + torch.exp(
            sink - logits_or_sinks_max
        )
        probs = unnormalized_scores / normalizer
        lse = (
            (torch.log(normalizer) + logits_or_sinks_max).squeeze(-1).permute(0, 2, 1).contiguous()
            if return_lse
            else None
        )
    out = torch.matmul(probs, v_t.float()).to(q.dtype)
    return out.permute(0, 2, 1, 3).contiguous(), lse


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    mask_mod: Optional[Callable] = None,
    full_block_cnt: Optional[torch.Tensor] = None,
    full_block_idx: Optional[torch.Tensor] = None,
    mask_block_cnt: Optional[torch.Tensor] = None,
    mask_block_idx: Optional[torch.Tensor] = None,
    block_size: Optional[Tuple[int, int]] = None,
    return_lse: bool = False,
):
    del deterministic, num_splits, pack_gqa, block_size
    if any(
        tensor is not None
        for tensor in (full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx)
    ):
        raise NotImplementedError("block-sparse inputs are not supported by the Windows shim")
    _check_supported_common(learnable_sink=learnable_sink, mask_mod=mask_mod, softcap=softcap)
    return _attention_forward_dense(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        learnable_sink=learnable_sink,
        softcap=softcap,
        return_lse=return_lse,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_lse: bool = False,
):
    del deterministic, num_splits, pack_gqa, max_seqlen_q, max_seqlen_k, aux_tensors
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError("cu_seqlens_q and cu_seqlens_k are required for varlen shim")
    if page_table is not None:
        raise NotImplementedError("paged KV is not supported by the Windows shim")
    _check_supported_common(learnable_sink=learnable_sink, mask_mod=score_mod, softcap=softcap)

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("varlen flash_attn_varlen_func expects q, k, v shaped (total, heads, dim)")

    batch = cu_seqlens_q.numel() - 1
    outputs = []
    lse_chunks = []

    for batch_idx in range(batch):
        q_start = int(cu_seqlens_q[batch_idx].item())
        q_end = int(cu_seqlens_q[batch_idx + 1].item())
        k_start = int(cu_seqlens_k[batch_idx].item())
        k_end = int(cu_seqlens_k[batch_idx + 1].item())

        if seqused_q is not None:
            q_end = min(q_end, q_start + int(seqused_q[batch_idx].item()))
        if seqused_k is not None:
            k_end = min(k_end, k_start + int(seqused_k[batch_idx].item()))

        q_chunk = q[q_start:q_end].unsqueeze(0)
        k_chunk = k[k_start:k_end].unsqueeze(0)
        v_chunk = v[k_start:k_end].unsqueeze(0)

        out_chunk, lse_chunk = _attention_forward_dense(
            q_chunk,
            k_chunk,
            v_chunk,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            learnable_sink=learnable_sink,
            softcap=softcap,
            return_lse=return_lse,
        )
        outputs.append(out_chunk.squeeze(0))
        if return_lse:
            lse_chunks.append(lse_chunk.squeeze(0))

    out = torch.cat(outputs, dim=0) if outputs else q.new_empty((0, q.shape[1], q.shape[2]))
    lse = torch.cat(lse_chunks, dim=0) if return_lse and lse_chunks else None
    return out, lse


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
