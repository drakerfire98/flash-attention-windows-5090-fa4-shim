"""Stable Windows shim for the `flash_attn.cute` public API.

This does not implement native FA4 kernels. It provides a numerically stable
fallback for the public forward entrypoints using PyTorch attention ops while
keeping the same import path.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

__version__ = "0.0.0-windows-shim"


@dataclass(frozen=True)
class _SeqLenInfo:
    offset_q: torch.Tensor
    offset_k: torch.Tensor
    seqlen_q: int
    seqlen_k: int


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


def _make_seqlen_info(
    *,
    device: torch.device,
    offset_q: int,
    offset_k: int,
    seqlen_q: int,
    seqlen_k: int,
) -> _SeqLenInfo:
    return _SeqLenInfo(
        offset_q=torch.tensor(int(offset_q), device=device, dtype=torch.long),
        offset_k=torch.tensor(int(offset_k), device=device, dtype=torch.long),
        seqlen_q=int(seqlen_q),
        seqlen_k=int(seqlen_k),
    )


def _resolve_mod_signature(mod: Callable, score_first: bool) -> str:
    try:
        signature = inspect.signature(mod)
    except (TypeError, ValueError):
        return "full"

    positional_names = []
    has_varargs = False
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_names.append(parameter.name.lower())
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True

    base_arity = 5 if score_first else 4
    if has_varargs or len(positional_names) >= base_arity + 2:
        return "full"
    if len(positional_names) == base_arity + 1:
        last_name = positional_names[-1]
        if "seq" in last_name or "offset" in last_name or "info" in last_name:
            return "seqlen"
        return "aux"
    if len(positional_names) == base_arity:
        return "minimal"
    raise TypeError(
        "Unsupported custom modifier signature. Expected 4/5/6 args for mask_mod "
        "or 5/6/7 args for score_mod."
    )


def _coerce_mod_output(
    value: torch.Tensor | float | bool,
    *,
    like: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.to(device=like.device)
    else:
        tensor = torch.as_tensor(value, device=like.device)
    tensor = tensor.to(dtype=dtype)
    if tensor.shape != like.shape:
        tensor = torch.broadcast_to(tensor, like.shape)
    return tensor


def _call_score_mod(
    score_mod: Callable,
    scores: torch.Tensor,
    *,
    batch_idx: torch.Tensor,
    head_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    seqlen_info: _SeqLenInfo,
    aux_tensors: Optional[list],
) -> torch.Tensor:
    call_mode = _resolve_mod_signature(score_mod, score_first=True)
    if call_mode == "full":
        result = score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors)
    elif call_mode == "aux":
        result = score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, aux_tensors)
    elif call_mode == "seqlen":
        result = score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info)
    else:
        result = score_mod(scores, batch_idx, head_idx, q_idx, kv_idx)
    return _coerce_mod_output(result, like=scores, dtype=torch.float32)


def _call_mask_mod(
    mask_mod: Callable,
    *,
    like: torch.Tensor,
    batch_idx: torch.Tensor,
    head_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    seqlen_info: _SeqLenInfo,
    aux_tensors: Optional[list],
) -> torch.Tensor:
    call_mode = _resolve_mod_signature(mask_mod, score_first=False)
    if call_mode == "full":
        result = mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors)
    elif call_mode == "aux":
        result = mask_mod(batch_idx, head_idx, q_idx, kv_idx, aux_tensors)
    elif call_mode == "seqlen":
        result = mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info)
    else:
        result = mask_mod(batch_idx, head_idx, q_idx, kv_idx)
    return _coerce_mod_output(result, like=like, dtype=torch.bool)


def _apply_score_mod(
    scores: torch.Tensor,
    score_mod: Callable,
    *,
    batch_start_index: int,
    offset_q: int,
    offset_k: int,
    aux_tensors: Optional[list],
) -> torch.Tensor:
    batch, heads, seqlen_q, seqlen_k = scores.shape
    q_idx = torch.arange(seqlen_q, device=scores.device, dtype=torch.long).view(seqlen_q, 1)
    kv_idx = torch.arange(seqlen_k, device=scores.device, dtype=torch.long).view(1, seqlen_k)
    seqlen_info = _make_seqlen_info(
        device=scores.device,
        offset_q=offset_q,
        offset_k=offset_k,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
    )
    modified = scores.clone()
    for batch_offset in range(batch):
        batch_idx = torch.tensor(batch_start_index + batch_offset, device=scores.device, dtype=torch.long)
        for head in range(heads):
            head_idx = torch.tensor(head, device=scores.device, dtype=torch.long)
            modified[batch_offset, head] = _call_score_mod(
                score_mod,
                modified[batch_offset, head],
                batch_idx=batch_idx,
                head_idx=head_idx,
                q_idx=q_idx,
                kv_idx=kv_idx,
                seqlen_info=seqlen_info,
                aux_tensors=aux_tensors,
            )
    return modified


def _apply_mask_mod(
    scores: torch.Tensor,
    mask_mod: Callable,
    *,
    batch_start_index: int,
    offset_q: int,
    offset_k: int,
    aux_tensors: Optional[list],
) -> torch.Tensor:
    batch, heads, seqlen_q, seqlen_k = scores.shape
    q_idx = torch.arange(seqlen_q, device=scores.device, dtype=torch.long).view(seqlen_q, 1)
    kv_idx = torch.arange(seqlen_k, device=scores.device, dtype=torch.long).view(1, seqlen_k)
    seqlen_info = _make_seqlen_info(
        device=scores.device,
        offset_q=offset_q,
        offset_k=offset_k,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
    )
    keep_mask = torch.empty_like(scores, dtype=torch.bool)
    for batch_offset in range(batch):
        batch_idx = torch.tensor(batch_start_index + batch_offset, device=scores.device, dtype=torch.long)
        for head in range(heads):
            head_idx = torch.tensor(head, device=scores.device, dtype=torch.long)
            keep_mask[batch_offset, head] = _call_mask_mod(
                mask_mod,
                like=scores[batch_offset, head],
                batch_idx=batch_idx,
                head_idx=head_idx,
                q_idx=q_idx,
                kv_idx=kv_idx,
                seqlen_info=seqlen_info,
                aux_tensors=aux_tensors,
            )
    return keep_mask


def _safe_softmax(
    scores: torch.Tensor,
    *,
    return_lse: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    valid = ~torch.isneginf(scores)
    all_masked = ~valid.any(dim=-1, keepdim=True)
    row_max = torch.amax(scores, dim=-1, keepdim=True)
    safe_row_max = torch.where(all_masked, torch.zeros_like(row_max), row_max)
    exp_scores = torch.where(valid, torch.exp(scores - safe_row_max), torch.zeros_like(scores))
    normalizer = exp_scores.sum(dim=-1, keepdim=True)
    probs = torch.where(normalizer > 0, exp_scores / normalizer, torch.zeros_like(exp_scores))
    lse = None
    if return_lse:
        lse = torch.where(
            normalizer > 0,
            torch.log(normalizer) + safe_row_max,
            torch.full_like(safe_row_max, float("-inf")),
        ).squeeze(-1)
    return probs, lse


def _check_supported_common(
    learnable_sink: Optional[torch.Tensor],
    mask_mod: Optional[Callable],
    softcap: float,
    score_mod: Optional[Callable] = None,
) -> None:
    del mask_mod
    if softcap < 0:
        raise ValueError("softcap must be >= 0")
    if softcap > 0.0 and score_mod is not None:
        raise NotImplementedError("softcap and score_mod are mutually exclusive in the Windows shim")


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
    score_mod: Optional[Callable],
    mask_mod: Optional[Callable],
    aux_tensors: Optional[list],
    batch_start_index: int,
    offset_q: int,
    offset_k: int,
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
    attn_mask = None
    if mask_mod is None:
        attn_mask = _build_attention_mask(
            seqlen_q=q.shape[1],
            seqlen_k=k.shape[1],
            causal=causal,
            window_size=window_size,
            device=q.device,
        )

    if softcap == 0.0 and not return_lse and sink is None and score_mod is None and mask_mod is None:
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
    if score_mod is not None:
        scores = _apply_score_mod(
            scores,
            score_mod,
            batch_start_index=batch_start_index,
            offset_q=offset_q,
            offset_k=offset_k,
            aux_tensors=aux_tensors,
        )
    if softcap > 0.0:
        scores = torch.tanh(scores / softcap) * softcap
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.view(1, 1, q.shape[1], k.shape[1]), float("-inf"))
    if mask_mod is not None:
        keep_mask = _apply_mask_mod(
            scores,
            mask_mod,
            batch_start_index=batch_start_index,
            offset_q=offset_q,
            offset_k=offset_k,
            aux_tensors=aux_tensors,
        )
        scores = scores.masked_fill(~keep_mask, float("-inf"))
    if sink is None:
        probs, lse = _safe_softmax(scores, return_lse=return_lse)
        if lse is not None:
            lse = lse.permute(0, 2, 1).contiguous()
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
        score_mod=None,
        mask_mod=mask_mod,
        aux_tensors=None,
        batch_start_index=0,
        offset_q=0,
        offset_k=0,
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
    del deterministic, num_splits, pack_gqa, max_seqlen_q, max_seqlen_k
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError("cu_seqlens_q and cu_seqlens_k are required for varlen shim")
    if page_table is not None:
        raise NotImplementedError("paged KV is not supported by the Windows shim")
    _check_supported_common(
        learnable_sink=learnable_sink,
        mask_mod=None,
        softcap=softcap,
        score_mod=score_mod,
    )

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
            score_mod=score_mod,
            mask_mod=None,
            aux_tensors=aux_tensors,
            batch_start_index=batch_idx,
            offset_q=q_start,
            offset_k=k_start,
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
