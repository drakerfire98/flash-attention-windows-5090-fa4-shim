"""Runtime-owned local attention core for the Windows FA4 bridge path."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SeqLenInfo:
    offset_q: torch.Tensor
    offset_k: torch.Tensor
    seqlen_q: int
    seqlen_k: int


@dataclass(frozen=True)
class VarlenLayout:
    packed: bool
    batch: int
    logical_starts: tuple[int, ...]
    used_lengths: tuple[int, ...]
    padded_length: Optional[int] = None


def _coerce_python_int(value: int | torch.Tensor) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def normalize_window_size(
    window_size: tuple[int | None, int | None],
) -> tuple[int | None, int | None]:
    left, right = window_size
    left = None if left is None or left < 0 else int(left)
    right = None if right is None or right < 0 else int(right)
    return left, right


def _expand_kv_for_q(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    window_size: tuple[int | None, int | None],
    device: torch.device,
) -> Optional[torch.Tensor]:
    left, right = normalize_window_size(window_size)
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
) -> SeqLenInfo:
    return SeqLenInfo(
        offset_q=torch.tensor(int(offset_q), device=device, dtype=torch.long),
        offset_k=torch.tensor(int(offset_k), device=device, dtype=torch.long),
        seqlen_q=int(seqlen_q),
        seqlen_k=int(seqlen_k),
    )


def build_varlen_layout(
    tensor: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    name: str,
) -> VarlenLayout:
    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1:
            raise ValueError(f"{name} cu_seqlens must be 1D")
        if tensor.ndim != 3:
            raise ValueError(
                f"packed varlen {name} must be shaped (total, heads, dim) when cu_seqlens are provided"
            )
        batch = cu_seqlens.numel() - 1
        if batch < 0:
            raise ValueError(f"{name} cu_seqlens must have at least one element")
        if seqused is not None and (seqused.ndim != 1 or seqused.numel() != batch):
            raise ValueError(f"{name} seqused must be 1D with one length per batch item")
        logical_starts: list[int] = []
        used_lengths: list[int] = []
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
        return VarlenLayout(
            packed=True,
            batch=batch,
            logical_starts=tuple(logical_starts),
            used_lengths=tuple(used_lengths),
        )

    if tensor.ndim != 4:
        raise ValueError(
            f"padded varlen {name} must be shaped (batch, seqlen, heads, dim) when cu_seqlens are absent"
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
    return VarlenLayout(
        packed=False,
        batch=batch,
        logical_starts=tuple(logical_starts),
        used_lengths=tuple(used_lengths),
        padded_length=padded_length,
    )


def _slice_varlen_batch(
    tensor: torch.Tensor,
    layout: VarlenLayout,
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
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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

    positional_names: list[str] = []
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
    seqlen_info: SeqLenInfo,
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
    seqlen_info: SeqLenInfo,
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


def apply_score_mod(
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


def apply_mask_mod(
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
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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


def _normalize_block_sparse_pair(
    name: str,
    cnt: Optional[torch.Tensor],
    idx: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (cnt is None) != (idx is None):
        raise ValueError(f"{name}_block_cnt and {name}_block_idx must both be provided or both be omitted")
    if cnt is None or idx is None:
        return None, None
    if cnt.ndim != 3 or idx.ndim != 4:
        raise ValueError(f"{name}_block tensors must have shapes (B, H, M) and (B, H, M, N)")
    if cnt.dtype != torch.int32 or idx.dtype != torch.int32:
        raise ValueError(f"{name}_block tensors must use dtype torch.int32")
    if cnt.device != idx.device:
        raise ValueError(f"{name}_block tensors must live on the same device")
    if cnt.shape[:3] != idx.shape[:3]:
        raise ValueError(f"{name}_block tensors must share batch/head/m-block dimensions")
    return cnt, idx


def _resolve_block_sparse_block_size(
    *,
    block_size: Optional[tuple[int, int]],
    seqlen_q: int,
    seqlen_k: int,
    mask_block_idx: Optional[torch.Tensor],
    full_block_idx: Optional[torch.Tensor],
) -> tuple[int, int]:
    if block_size is not None:
        block_q, block_k = block_size
        if block_q <= 0 or block_k <= 0:
            raise ValueError("block_size entries must be positive")
        return int(block_q), int(block_k)

    index_tensor = mask_block_idx if mask_block_idx is not None else full_block_idx
    if index_tensor is None:
        raise ValueError("block-sparse inputs require at least one block index tensor")
    num_m_blocks = int(index_tensor.shape[2])
    num_n_blocks = int(index_tensor.shape[3])
    if num_m_blocks <= 0 or num_n_blocks <= 0:
        raise ValueError("block-sparse tensors must include at least one block in each dimension")
    return (
        max(1, (int(seqlen_q) + num_m_blocks - 1) // num_m_blocks),
        max(1, (int(seqlen_k) + num_n_blocks - 1) // num_n_blocks),
    )


def _select_block_sparse_batch_tensor(
    tensor: Optional[torch.Tensor],
    batch_idx: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    batch_sel = 0 if tensor.shape[0] == 1 else batch_idx
    return tensor[batch_sel : batch_sel + 1]


def build_block_sparse_keep_mask(
    *,
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    mask_block_cnt: Optional[torch.Tensor],
    mask_block_idx: Optional[torch.Tensor],
    full_block_cnt: Optional[torch.Tensor],
    full_block_idx: Optional[torch.Tensor],
    block_size: Optional[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    mask_block_cnt, mask_block_idx = _normalize_block_sparse_pair("mask", mask_block_cnt, mask_block_idx)
    full_block_cnt, full_block_idx = _normalize_block_sparse_pair("full", full_block_cnt, full_block_idx)
    if mask_block_idx is None and full_block_idx is None:
        raise ValueError("block-sparse inputs require at least one of mask/full block tensors")

    if mask_block_idx is None:
        assert full_block_cnt is not None and full_block_idx is not None
        mask_block_cnt = torch.zeros_like(full_block_cnt)
        mask_block_idx = torch.zeros_like(full_block_idx)

    assert mask_block_cnt is not None and mask_block_idx is not None
    block_q, block_k = _resolve_block_sparse_block_size(
        block_size=block_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        mask_block_idx=mask_block_idx,
        full_block_idx=full_block_idx,
    )

    keep_mask = torch.zeros((batch_size, num_heads, seqlen_q, seqlen_k), device=device, dtype=torch.bool)
    num_sparse_m_blocks = mask_block_cnt.shape[2]
    for batch_idx in range(batch_size):
        batch_sel = 0 if mask_block_cnt.shape[0] == 1 else batch_idx
        full_batch_sel = 0 if full_block_cnt is None or full_block_cnt.shape[0] == 1 else batch_idx
        for head_idx in range(num_heads):
            head_sel = 0 if mask_block_cnt.shape[1] == 1 else head_idx
            full_head_sel = 0 if full_block_cnt is None or full_block_cnt.shape[1] == 1 else head_idx
            for sparse_m_block in range(num_sparse_m_blocks):
                q_start = sparse_m_block * block_q
                q_end = min(q_start + block_q, seqlen_q)
                if q_start >= q_end:
                    continue
                mask_count = int(mask_block_cnt[batch_sel, head_sel, sparse_m_block].item())
                for offset in range(mask_count):
                    n_block = int(mask_block_idx[batch_sel, head_sel, sparse_m_block, offset].item())
                    k_start = n_block * block_k
                    k_end = min(k_start + block_k, seqlen_k)
                    if k_start < k_end:
                        keep_mask[batch_idx, head_idx, q_start:q_end, k_start:k_end] = True
                if full_block_cnt is not None and full_block_idx is not None:
                    full_count = int(full_block_cnt[full_batch_sel, full_head_sel, sparse_m_block].item())
                    for offset in range(full_count):
                        n_block = int(full_block_idx[full_batch_sel, full_head_sel, sparse_m_block, offset].item())
                        k_start = n_block * block_k
                        k_end = min(k_start + block_k, seqlen_k)
                        if k_start < k_end:
                            keep_mask[batch_idx, head_idx, q_start:q_end, k_start:k_end] = True
    return keep_mask


def materialize_dense_keep_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    mask_mod: Optional[Callable],
    aux_tensors: Optional[list],
    batch_start_index: int,
    offset_q: int,
    offset_k: int,
    block_sparse_tensors: Optional[tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    block_size: Optional[tuple[int, int]] = None,
) -> Optional[torch.Tensor]:
    keep_mask: Optional[torch.Tensor] = None
    if block_sparse_tensors is not None:
        mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = block_sparse_tensors
        keep_mask = build_block_sparse_keep_mask(
            batch_size=int(q.shape[0]),
            num_heads=int(q.shape[2]),
            seqlen_q=int(q.shape[1]),
            seqlen_k=int(k.shape[1]),
            mask_block_cnt=mask_block_cnt,
            mask_block_idx=mask_block_idx,
            full_block_cnt=full_block_cnt,
            full_block_idx=full_block_idx,
            block_size=block_size,
            device=q.device,
        )
    if mask_mod is None:
        return keep_mask
    mod_like = torch.empty(
        (int(q.shape[0]), int(q.shape[2]), int(q.shape[1]), int(k.shape[1])),
        device=q.device,
        dtype=torch.float32,
    )
    mod_mask = apply_mask_mod(
        mod_like,
        mask_mod,
        batch_start_index=batch_start_index,
        offset_q=offset_q,
        offset_k=offset_k,
        aux_tensors=aux_tensors,
    )
    if keep_mask is None:
        return mod_mask
    return keep_mask & mod_mask


def _materialize_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if page_table.ndim != 2:
        raise ValueError("page_table must be shaped (batch, max_num_pages_per_seq)")
    if page_table.dtype != torch.int32:
        raise ValueError("page_table must use dtype torch.int32")
    if page_table.device != k.device:
        page_table = page_table.to(device=k.device)
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("paged KV expects k and v shaped (num_pages, page_size, heads, dim)")
    if k.shape[:3] != v.shape[:3]:
        raise ValueError("paged KV expects k and v to share page geometry and KV heads")

    num_pages, page_size, num_heads = k.shape[:3]
    table = page_table.to(dtype=torch.long)
    if table.numel() > 0:
        table_min = int(table.min().item())
        table_max = int(table.max().item())
        if table_min < 0 or table_max >= num_pages:
            raise ValueError("page_table contains page indices outside the available KV cache range")

    batch_size, max_num_pages = table.shape
    gathered_k = k.index_select(0, table.reshape(-1)).reshape(
        batch_size, max_num_pages * page_size, num_heads, k.shape[-1]
    )
    gathered_v = v.index_select(0, table.reshape(-1)).reshape(
        batch_size, max_num_pages * page_size, num_heads, v.shape[-1]
    )
    return gathered_k, gathered_v


def prepare_varlen_kv_inputs(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if page_table is None:
        return k, v, cu_seqlens_k, seqused_k
    if cu_seqlens_k is not None:
        raise ValueError("page_table is not supported together with cu_seqlens_k")
    k_dense, v_dense = _materialize_paged_kv_cache(k, v, page_table)
    return k_dense, v_dense, None, seqused_k


def materialize_varlen_keep_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor],
    mask_mod: Optional[Callable],
    aux_tensors: Optional[list],
    block_sparse_tensors: Optional[tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    block_size: Optional[tuple[int, int]] = None,
) -> Optional[torch.Tensor]:
    if mask_mod is None and block_sparse_tensors is None:
        return None

    k_input, _, cu_seqlens_k_input, seqused_k_input = prepare_varlen_kv_inputs(
        k,
        k,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        page_table=page_table,
    )
    q_layout = build_varlen_layout(q, cu_seqlens=cu_seqlens_q, seqused=seqused_q, name="q")
    k_layout = build_varlen_layout(
        k_input,
        cu_seqlens=cu_seqlens_k_input,
        seqused=seqused_k_input,
        name="k",
    )
    if q_layout.batch != k_layout.batch:
        raise ValueError("q and kv must describe the same batch size")

    num_heads = int(q.shape[1] if q_layout.packed else q.shape[2])
    max_q = max(q_layout.used_lengths, default=0)
    max_k = max(k_layout.used_lengths, default=0)
    keep_mask = torch.zeros((q_layout.batch, num_heads, max_q, max_k), device=q.device, dtype=torch.bool)

    for batch_idx in range(q_layout.batch):
        q_len = q_layout.used_lengths[batch_idx]
        k_len = k_layout.used_lengths[batch_idx]
        if q_len == 0 or k_len == 0:
            continue

        batch_mask = torch.ones((1, num_heads, q_len, k_len), device=q.device, dtype=torch.bool)
        if block_sparse_tensors is not None:
            mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = block_sparse_tensors
            batch_mask &= build_block_sparse_keep_mask(
                batch_size=1,
                num_heads=num_heads,
                seqlen_q=q_len,
                seqlen_k=k_len,
                mask_block_cnt=_select_block_sparse_batch_tensor(mask_block_cnt, batch_idx),
                mask_block_idx=_select_block_sparse_batch_tensor(mask_block_idx, batch_idx),
                full_block_cnt=_select_block_sparse_batch_tensor(full_block_cnt, batch_idx),
                full_block_idx=_select_block_sparse_batch_tensor(full_block_idx, batch_idx),
                block_size=block_size,
                device=q.device,
            )
        if mask_mod is not None:
            mod_like = torch.empty((1, num_heads, q_len, k_len), device=q.device, dtype=torch.float32)
            batch_mask &= apply_mask_mod(
                mod_like,
                mask_mod,
                batch_start_index=batch_idx,
                offset_q=q_layout.logical_starts[batch_idx],
                offset_k=k_layout.logical_starts[batch_idx],
                aux_tensors=aux_tensors,
            )
        keep_mask[batch_idx, :, :q_len, :k_len] = batch_mask[0]
    return keep_mask


def attention_forward_dense_local(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float],
    causal: bool,
    window_size: tuple[int | None, int | None],
    learnable_sink: Optional[torch.Tensor],
    softcap: float,
    score_mod: Optional[Callable],
    mask_mod: Optional[Callable],
    aux_tensors: Optional[list],
    batch_start_index: int,
    offset_q: int,
    offset_k: int,
    extra_keep_mask: Optional[torch.Tensor],
    return_lse: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("dense flash attention expects q, k, v shaped (batch, seqlen, heads, dim)")

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

    if (
        softcap == 0.0
        and not return_lse
        and sink is None
        and score_mod is None
        and mask_mod is None
        and extra_keep_mask is None
    ):
        if causal and normalize_window_size(window_size) == (None, None):
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
        scores = apply_score_mod(
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
    if extra_keep_mask is not None:
        scores = scores.masked_fill(~extra_keep_mask, float("-inf"))
    if mask_mod is not None:
        keep_mask = apply_mask_mod(
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
        logits_or_sink_max = torch.maximum(logits_max, sink)
        unnormalized_scores = torch.exp(scores - logits_or_sink_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - logits_or_sink_max)
        probs = unnormalized_scores / normalizer
        lse = (
            (torch.log(normalizer) + logits_or_sink_max).squeeze(-1).permute(0, 2, 1).contiguous()
            if return_lse
            else None
        )
    out = torch.matmul(probs, v_t.float()).to(q.dtype)
    return out.permute(0, 2, 1, 3).contiguous(), lse


def _finalize_varlen_output_chunk(
    *,
    outputs: list[torch.Tensor],
    lse_chunks: list[torch.Tensor],
    out_chunk: torch.Tensor,
    lse_chunk: Optional[torch.Tensor],
    q_layout: VarlenLayout,
    batch_idx: int,
    return_lse: bool,
) -> None:
    out_piece = out_chunk.squeeze(0)
    lse_piece = lse_chunk.squeeze(0) if return_lse and lse_chunk is not None else None
    if q_layout.packed:
        outputs.append(out_piece)
        if return_lse and lse_piece is not None:
            lse_chunks.append(lse_piece)
        return

    pad_q = q_layout.padded_length - q_layout.used_lengths[batch_idx]
    if pad_q > 0:
        out_piece = torch.cat(
            [out_piece, out_piece.new_zeros((pad_q, out_piece.shape[1], out_piece.shape[2]))],
            dim=0,
        )
        if return_lse and lse_piece is not None:
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
    if return_lse and lse_piece is not None:
        lse_chunks.append(lse_piece)


def _assemble_varlen_outputs(
    *,
    q_layout: VarlenLayout,
    outputs: list[torch.Tensor],
    lse_chunks: list[torch.Tensor],
    q: torch.Tensor,
    v: torch.Tensor,
    return_lse: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if q_layout.packed:
        out = torch.cat(outputs, dim=0) if outputs else q.new_empty((0, q.shape[1], v.shape[-1]))
        lse = torch.cat(lse_chunks, dim=0) if return_lse and lse_chunks else None
    else:
        out = torch.stack(outputs, dim=0)
        lse = torch.stack(lse_chunks, dim=0) if return_lse else None
    return out, lse


def attention_forward_varlen_local(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float],
    causal: bool,
    window_size: tuple[int | None, int | None],
    learnable_sink: Optional[torch.Tensor],
    softcap: float,
    score_mod: Optional[Callable],
    mask_mod: Optional[Callable],
    aux_tensors: Optional[list],
    block_sparse_tensors: Optional[tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    block_size: Optional[tuple[int, int]] = None,
    return_lse: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    k_input, v_input, cu_seqlens_k_input, seqused_k_input = prepare_varlen_kv_inputs(
        k,
        v,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        page_table=page_table,
    )

    q_layout = build_varlen_layout(q, cu_seqlens=cu_seqlens_q, seqused=seqused_q, name="q")
    k_layout = build_varlen_layout(
        k_input,
        cu_seqlens=cu_seqlens_k_input,
        seqused=seqused_k_input,
        name="k",
    )
    v_layout = build_varlen_layout(
        v_input,
        cu_seqlens=cu_seqlens_k_input,
        seqused=seqused_k_input,
        name="v",
    )
    if k_layout != v_layout:
        raise ValueError("k and v must use the same packed/padded layout and effective sequence lengths")
    if q_layout.batch != k_layout.batch:
        raise ValueError("q and kv must describe the same batch size")

    outputs: list[torch.Tensor] = []
    lse_chunks: list[torch.Tensor] = []
    for batch_idx in range(q_layout.batch):
        q_chunk = _slice_varlen_batch(q, q_layout, batch_idx)
        k_chunk = _slice_varlen_batch(k_input, k_layout, batch_idx)
        v_chunk = _slice_varlen_batch(v_input, v_layout, batch_idx)
        q_used = q_layout.used_lengths[batch_idx]
        k_used = k_layout.used_lengths[batch_idx]
        if q_used == 0 or k_used == 0:
            out_chunk, lse_chunk = _zero_attention_chunk(q_chunk, v_chunk, return_lse=return_lse)
        else:
            local_block_sparse_tensors = None
            if block_sparse_tensors is not None:
                local_block_sparse_tensors = (
                    _select_block_sparse_batch_tensor(block_sparse_tensors[0], batch_idx),
                    _select_block_sparse_batch_tensor(block_sparse_tensors[1], batch_idx),
                    _select_block_sparse_batch_tensor(block_sparse_tensors[2], batch_idx),
                    _select_block_sparse_batch_tensor(block_sparse_tensors[3], batch_idx),
                )
            extra_keep_mask = materialize_varlen_keep_mask(
                q_chunk,
                k_chunk,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                seqused_q=None,
                seqused_k=None,
                page_table=None,
                mask_mod=mask_mod,
                aux_tensors=aux_tensors,
                block_sparse_tensors=local_block_sparse_tensors,
                block_size=block_size,
            )
            if extra_keep_mask is not None:
                extra_keep_mask = extra_keep_mask[:, :, :q_used, :k_used]
            out_chunk, lse_chunk = attention_forward_dense_local(
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
                extra_keep_mask=extra_keep_mask,
                return_lse=return_lse,
            )
        _finalize_varlen_output_chunk(
            outputs=outputs,
            lse_chunks=lse_chunks,
            out_chunk=out_chunk,
            lse_chunk=lse_chunk,
            q_layout=q_layout,
            batch_idx=batch_idx,
            return_lse=return_lse,
        )

    return _assemble_varlen_outputs(
        q_layout=q_layout,
        outputs=outputs,
        lse_chunks=lse_chunks,
        q=q,
        v=v_input,
        return_lse=return_lse,
    )


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
    like = torch.empty((1, 1), device=device, dtype=torch.bool)
    result = _call_mask_mod(
        mask_mod,
        like=like,
        batch_idx=torch.tensor(batch_idx, device=device, dtype=torch.long),
        head_idx=torch.tensor(head_idx, device=device, dtype=torch.long),
        q_idx=torch.tensor([[q_idx]], device=device, dtype=torch.long),
        kv_idx=torch.tensor([[kv_idx]], device=device, dtype=torch.long),
        seqlen_info=_make_seqlen_info(
            device=device,
            offset_q=0,
            offset_k=0,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
        ),
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
    q_last = min(m_base + tile_m - 1, seqlen_q - 1)
    k_last = min(n_base + tile_n - 1, seqlen_k - 1)
    q_mid = m_base + min(seqlen_q - m_base, tile_m) // 2
    k_mid = n_base + min(seqlen_k - n_base, tile_n) // 2
    return [(m_base, n_base), (m_base, k_last), (q_last, n_base), (q_last, k_last), (q_mid, k_mid)]


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
    points = _block_sampling_points(
        m_base,
        n_base,
        tile_m=tile_m,
        tile_n=tile_n,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
    ) if use_fast_sampling else None
    if points is None:
        points = []
        for row_offset in range(tile_m):
            q_idx = m_base + row_offset
            if q_idx >= seqlen_q:
                break
            for col_offset in range(tile_n):
                kv_idx = n_base + col_offset
                if kv_idx >= seqlen_k:
                    break
                points.append((q_idx, kv_idx))
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


def fill_block_sparse_tensors(
    *,
    tile_m: int,
    tile_n: int,
    batch_size: int,
    num_heads: int,
    seqlen_q: int | torch.Tensor,
    seqlen_k: int | torch.Tensor,
    mask_mod: Callable,
    aux_tensors: Optional[list],
    device: torch.device,
    compute_full_blocks: bool,
    use_fast_sampling: bool,
    mask_block_cnt: torch.Tensor,
    mask_block_idx: torch.Tensor,
    full_block_cnt: Optional[torch.Tensor],
    full_block_idx: Optional[torch.Tensor],
) -> None:
    seqlen_q_int = _coerce_python_int(seqlen_q)
    seqlen_k_int = _coerce_python_int(seqlen_k)
    use_fast_sampling = bool(getattr(mask_mod, "use_fast_sampling", use_fast_sampling))
    num_m_blocks = (seqlen_q_int + tile_m - 1) // tile_m
    num_n_blocks = (seqlen_k_int + tile_n - 1) // tile_n
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
                        seqlen_q=seqlen_q_int,
                        seqlen_k=seqlen_k_int,
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
