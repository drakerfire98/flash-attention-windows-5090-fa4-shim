"""Stable Windows shim for the `flash_attn.cute` public API.

This does not implement native FA4 kernels. It provides a numerically stable
fallback for the public forward entrypoints using PyTorch attention ops while
keeping the same import path.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

__version__ = "0.0.0-windows-shim"


class BlockSparseTensors(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None
    full_block_idx: torch.Tensor | None


class BlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None = None
    full_block_idx: torch.Tensor | None = None
    block_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class _SeqLenInfo:
    offset_q: torch.Tensor
    offset_k: torch.Tensor
    seqlen_q: int
    seqlen_k: int


@dataclass(frozen=True)
class _VarlenLayout:
    packed: bool
    batch: int
    logical_starts: tuple[int, ...]
    used_lengths: tuple[int, ...]
    padded_length: Optional[int] = None


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


def _build_varlen_layout(
    tensor: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    name: str,
) -> _VarlenLayout:
    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1:
            raise ValueError(f"{name} cu_seqlens must be 1D")
        if tensor.ndim != 3:
            raise ValueError(
                f"varlen flash_attn_varlen_func expects packed {name} shaped (total, heads, dim) "
                "when cu_seqlens are provided"
            )
        batch = cu_seqlens.numel() - 1
        if batch < 0:
            raise ValueError(f"{name} cu_seqlens must have at least one element")
        if seqused is not None:
            if seqused.ndim != 1 or seqused.numel() != batch:
                raise ValueError(f"{name} seqused must be 1D with one length per batch item")
        logical_starts = []
        used_lengths = []
        for batch_idx in range(batch):
            start = int(cu_seqlens[batch_idx].item())
            end = int(cu_seqlens[batch_idx + 1].item())
            if end < start:
                raise ValueError(f"{name} cu_seqlens must be nondecreasing")
            used = end - start
            if seqused is not None:
                used = min(used, max(0, int(seqused[batch_idx].item())))
            logical_starts.append(start)
            used_lengths.append(used)
        return _VarlenLayout(
            packed=True,
            batch=batch,
            logical_starts=tuple(logical_starts),
            used_lengths=tuple(used_lengths),
        )

    if tensor.ndim != 4:
        raise ValueError(
            f"varlen flash_attn_varlen_func expects padded {name} shaped (batch, seqlen, heads, dim) "
            "when cu_seqlens are not provided"
        )
    batch = tensor.shape[0]
    padded_length = tensor.shape[1]
    if seqused is not None:
        if seqused.ndim != 1 or seqused.numel() != batch:
            raise ValueError(f"{name} seqused must be 1D with one length per batch item")
        used_lengths = tuple(
            min(padded_length, max(0, int(seqused[batch_idx].item()))) for batch_idx in range(batch)
        )
    else:
        used_lengths = tuple([padded_length] * batch)
    logical_starts = []
    running = 0
    for used in used_lengths:
        logical_starts.append(running)
        running += used
    return _VarlenLayout(
        packed=False,
        batch=batch,
        logical_starts=tuple(logical_starts),
        used_lengths=tuple(used_lengths),
        padded_length=padded_length,
    )


def _slice_varlen_batch(
    tensor: torch.Tensor,
    layout: _VarlenLayout,
    batch_idx: int,
) -> torch.Tensor:
    used = layout.used_lengths[batch_idx]
    if layout.packed:
        start = layout.logical_starts[batch_idx]
        return tensor[start : start + used].unsqueeze(0)
    return tensor[batch_idx : batch_idx + 1, :used]


def _zero_attention_chunk(
    q_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    *,
    return_lse: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    out = q_chunk.new_zeros((1, q_chunk.shape[1], q_chunk.shape[2], v_chunk.shape[-1]))
    lse = None
    if return_lse:
        lse = torch.full(
            (1, q_chunk.shape[1], q_chunk.shape[2]),
            float("-inf"),
            device=q_chunk.device,
            dtype=torch.float32,
        )
    return out, lse


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
    if page_table is not None:
        raise NotImplementedError("paged KV is not supported by the Windows shim")
    _check_supported_common(
        learnable_sink=learnable_sink,
        mask_mod=None,
        softcap=softcap,
        score_mod=score_mod,
    )

    q_layout = _build_varlen_layout(q, cu_seqlens=cu_seqlens_q, seqused=seqused_q, name="q")
    k_layout = _build_varlen_layout(k, cu_seqlens=cu_seqlens_k, seqused=seqused_k, name="k")
    v_layout = _build_varlen_layout(v, cu_seqlens=cu_seqlens_k, seqused=seqused_k, name="v")
    if k_layout != v_layout:
        raise ValueError("k and v must use the same packed/padded layout and effective sequence lengths")
    if q_layout.batch != k_layout.batch:
        raise ValueError("q and kv must describe the same batch size")

    outputs = []
    lse_chunks = []

    for batch_idx in range(q_layout.batch):
        q_chunk = _slice_varlen_batch(q, q_layout, batch_idx)
        k_chunk = _slice_varlen_batch(k, k_layout, batch_idx)
        v_chunk = _slice_varlen_batch(v, v_layout, batch_idx)
        q_used = q_layout.used_lengths[batch_idx]
        k_used = k_layout.used_lengths[batch_idx]

        if q_used == 0 or k_used == 0:
            out_chunk, lse_chunk = _zero_attention_chunk(q_chunk, v_chunk, return_lse=return_lse)
        else:
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
                offset_q=q_layout.logical_starts[batch_idx],
                offset_k=k_layout.logical_starts[batch_idx],
                return_lse=return_lse,
            )

        out_piece = out_chunk.squeeze(0)
        lse_piece = lse_chunk.squeeze(0) if return_lse else None
        if q_layout.packed:
            outputs.append(out_piece)
            if return_lse:
                lse_chunks.append(lse_piece)
            continue

        pad_q = q_layout.padded_length - q_used
        if pad_q > 0:
            out_piece = torch.cat(
                [out_piece, out_piece.new_zeros((pad_q, out_piece.shape[1], out_piece.shape[2]))],
                dim=0,
            )
            if return_lse:
                lse_piece = torch.cat(
                    [
                        lse_piece,
                        torch.full(
                            (pad_q, lse_piece.shape[1]),
                            float("-inf"),
                            device=lse_piece.device,
                            dtype=lse_piece.dtype,
                        ),
                    ],
                    dim=0,
                )
        outputs.append(out_piece)
        if return_lse:
            lse_chunks.append(lse_piece)

    if q_layout.packed:
        out = torch.cat(outputs, dim=0) if outputs else q.new_empty((0, q.shape[1], v.shape[-1]))
        lse = torch.cat(lse_chunks, dim=0) if return_lse and lse_chunks else None
    else:
        out = torch.stack(outputs, dim=0)
        lse = torch.stack(lse_chunks, dim=0) if return_lse else None
    return out, lse


def _combine_split_attention(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out_partial.ndim not in (4, 5):
        raise ValueError("out_partial must have 4 or 5 dimensions")
    if lse_partial.ndim != out_partial.ndim - 1:
        raise ValueError("lse_partial rank must be exactly one less than out_partial rank")
    if out_partial.shape[:-1] != lse_partial.shape:
        raise ValueError("out_partial and lse_partial shapes are incompatible")
    if out_partial.shape[0] == 0:
        raise ValueError("out_partial must include at least one split")

    lse_float = lse_partial.float()
    valid = ~torch.isneginf(lse_float)
    any_valid = valid.any(dim=0)
    lse_max = torch.amax(lse_float, dim=0)
    safe_lse_max = torch.where(any_valid, lse_max, torch.zeros_like(lse_max))

    weights = torch.where(
        valid,
        torch.exp(lse_float - safe_lse_max.unsqueeze(0)),
        torch.zeros_like(lse_float),
    )
    denom = weights.sum(dim=0)
    numerator = (out_partial.float() * weights.unsqueeze(-1)).sum(dim=0)
    out = torch.where(
        denom.unsqueeze(-1) > 0,
        numerator / denom.unsqueeze(-1),
        torch.zeros_like(numerator),
    )
    lse = torch.where(
        denom > 0,
        torch.log(denom) + safe_lse_max,
        torch.full_like(safe_lse_max, float("-inf")),
    )
    return out, lse


def _coerce_python_int(value: int | torch.Tensor) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _mask_mod_scalar(
    mask_mod: Callable,
    *,
    batch_idx: int,
    head_idx: int,
    q_idx: int,
    kv_idx: int,
    seqlen_q: int,
    seqlen_k: int,
    aux_tensors: Optional[list],
    device: torch.device,
) -> bool:
    like = torch.zeros((), device=device, dtype=torch.bool)
    seqlen_info = _make_seqlen_info(
        device=device,
        offset_q=0,
        offset_k=0,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
    )
    result = _call_mask_mod(
        mask_mod,
        like=like,
        batch_idx=torch.tensor(batch_idx, device=device, dtype=torch.long),
        head_idx=torch.tensor(head_idx, device=device, dtype=torch.long),
        q_idx=torch.tensor(q_idx, device=device, dtype=torch.long),
        kv_idx=torch.tensor(kv_idx, device=device, dtype=torch.long),
        seqlen_info=seqlen_info,
        aux_tensors=aux_tensors,
    )
    return bool(result.item())


def _block_sampling_points(
    m_base: int,
    n_base: int,
    *,
    tile_m: int,
    tile_n: int,
    seqlen_q: int,
    seqlen_k: int,
) -> list[tuple[int, int]]:
    if m_base >= seqlen_q or n_base >= seqlen_k:
        return []
    q_last = min(m_base + tile_m - 1, seqlen_q - 1)
    k_last = min(n_base + tile_n - 1, seqlen_k - 1)
    q_mid = m_base + min(seqlen_q - m_base, tile_m) // 2
    k_mid = n_base + min(seqlen_k - n_base, tile_n) // 2
    return [
        (m_base, n_base),
        (m_base, k_last),
        (q_last, n_base),
        (q_last, k_last),
        (q_mid, k_mid),
    ]


def _classify_block(
    mask_mod: Callable,
    *,
    batch_idx: int,
    head_idx: int,
    m_base: int,
    n_base: int,
    tile_m: int,
    tile_n: int,
    seqlen_q: int,
    seqlen_k: int,
    aux_tensors: Optional[list],
    device: torch.device,
    use_fast_sampling: bool,
) -> tuple[bool, bool]:
    has_unmasked = False
    has_masked = False

    if use_fast_sampling:
        points = _block_sampling_points(
            m_base,
            n_base,
            tile_m=tile_m,
            tile_n=tile_n,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
        )
        for q_idx, kv_idx in points:
            allowed = _mask_mod_scalar(
                mask_mod,
                batch_idx=batch_idx,
                head_idx=head_idx,
                q_idx=q_idx,
                kv_idx=kv_idx,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                aux_tensors=aux_tensors,
                device=device,
            )
            has_unmasked |= allowed
            has_masked |= not allowed
            if has_unmasked and has_masked:
                break
        return has_unmasked, has_masked

    for row_offset in range(tile_m):
        q_idx = m_base + row_offset
        if q_idx >= seqlen_q:
            break
        for col_offset in range(tile_n):
            kv_idx = n_base + col_offset
            if kv_idx >= seqlen_k:
                break
            allowed = _mask_mod_scalar(
                mask_mod,
                batch_idx=batch_idx,
                head_idx=head_idx,
                q_idx=q_idx,
                kv_idx=kv_idx,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                aux_tensors=aux_tensors,
                device=device,
            )
            has_unmasked |= allowed
            has_masked |= not allowed
            if has_unmasked and has_masked:
                return has_unmasked, has_masked
    return has_unmasked, has_masked


def _fill_block_sparse_tensors(
    *,
    tile_m: int,
    tile_n: int,
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    mask_mod: Callable,
    aux_tensors: Optional[list],
    device: torch.device,
    compute_full_blocks: bool,
    use_fast_sampling: bool,
    mask_block_cnt: torch.Tensor,
    mask_block_idx: torch.Tensor,
    full_block_cnt: torch.Tensor | None,
    full_block_idx: torch.Tensor | None,
) -> None:
    num_m_blocks = (seqlen_q + tile_m - 1) // tile_m
    num_n_blocks = (seqlen_k + tile_n - 1) // tile_n
    mask_block_cnt.zero_()
    mask_block_idx.zero_()
    if full_block_cnt is not None:
        full_block_cnt.zero_()
    if full_block_idx is not None:
        full_block_idx.zero_()

    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            for m_block in range(num_m_blocks):
                m_base = m_block * tile_m
                mask_count = 0
                full_count = 0
                for n_block in range(num_n_blocks):
                    n_base = n_block * tile_n
                    has_unmasked, has_masked = _classify_block(
                        mask_mod,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        m_base=m_base,
                        n_base=n_base,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        seqlen_q=seqlen_q,
                        seqlen_k=seqlen_k,
                        aux_tensors=aux_tensors,
                        device=device,
                        use_fast_sampling=use_fast_sampling,
                    )
                    if has_unmasked and has_masked:
                        mask_block_idx[batch_idx, head_idx, m_block, mask_count] = n_block
                        mask_count += 1
                    elif has_unmasked and compute_full_blocks and full_block_idx is not None:
                        full_block_idx[batch_idx, head_idx, m_block, full_count] = n_block
                        full_count += 1
                mask_block_cnt[batch_idx, head_idx, m_block] = mask_count
                if compute_full_blocks and full_block_cnt is not None:
                    full_block_cnt[batch_idx, head_idx, m_block] = full_count


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    del cu_seqlens, seqused, varlen_batch_idx

    combined_out, combined_lse = _combine_split_attention(out_partial, lse_partial)
    if out is None:
        target_dtype = out_dtype if out_dtype is not None else out_partial.dtype
        out = combined_out.to(dtype=target_dtype)
    else:
        if out.shape != combined_out.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} does not match combined output shape {tuple(combined_out.shape)}"
            )
        out.copy_(combined_out.to(dtype=out.dtype))

    if not return_lse:
        return out, None
    return out, combined_lse


def fast_sampling(mask_mod: Callable) -> Callable:
    mask_mod.use_fast_sampling = True
    return mask_mod


def compute_block_sparsity(
    tile_m: int,
    tile_n: int,
    batch_size: int,
    num_heads: int,
    seqlen_q: int | torch.Tensor,
    seqlen_k: int | torch.Tensor,
    mask_mod: Callable,
    aux_tensors: Optional[list],
    device,
    compute_full_blocks: bool = True,
    use_fast_sampling: bool = False,
) -> tuple[BlockSparseTensors, BlockSparseTensorsTorch]:
    seqlen_q_int = _coerce_python_int(seqlen_q)
    seqlen_k_int = _coerce_python_int(seqlen_k)
    tile_m = int(tile_m)
    tile_n = int(tile_n)
    batch_size = int(batch_size)
    num_heads = int(num_heads)
    device = torch.device(device)
    use_fast_sampling = bool(getattr(mask_mod, "use_fast_sampling", use_fast_sampling))

    num_m_blocks = (seqlen_q_int + tile_m - 1) // tile_m
    num_n_blocks = (seqlen_k_int + tile_n - 1) // tile_n
    mask_block_cnt = torch.zeros(
        (batch_size, num_heads, num_m_blocks),
        device=device,
        dtype=torch.int32,
    )
    mask_block_idx = torch.zeros(
        (batch_size, num_heads, num_m_blocks, num_n_blocks),
        device=device,
        dtype=torch.int32,
    )
    full_block_cnt = (
        torch.zeros((batch_size, num_heads, num_m_blocks), device=device, dtype=torch.int32)
        if compute_full_blocks
        else None
    )
    full_block_idx = (
        torch.zeros((batch_size, num_heads, num_m_blocks, num_n_blocks), device=device, dtype=torch.int32)
        if compute_full_blocks
        else None
    )

    _fill_block_sparse_tensors(
        tile_m=tile_m,
        tile_n=tile_n,
        batch_size=batch_size,
        num_heads=num_heads,
        seqlen_q=seqlen_q_int,
        seqlen_k=seqlen_k_int,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        device=device,
        compute_full_blocks=compute_full_blocks,
        use_fast_sampling=use_fast_sampling,
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
    )

    torch_tensors = BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(tile_m, tile_n),
    )
    cute_tensors = BlockSparseTensors(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
    )
    return cute_tensors, torch_tensors


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_combine",
    "compute_block_sparsity",
    "fast_sampling",
    "BlockSparseTensors",
    "BlockSparseTensorsTorch",
]
