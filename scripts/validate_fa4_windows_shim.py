"""Run a broader validation matrix for the Windows FA4 shim."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_shim_path() -> None:
    sys.path.insert(0, str(_repo_root() / "shims"))


def _manual_seed() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    diff = (actual.float() - expected.float()).abs()
    finite_diff = diff[torch.isfinite(diff)]
    max_diff = finite_diff.max().item() if finite_diff.numel() > 0 else 0.0
    print(f"{name}_max_diff={max_diff}")
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def _assert_grad_close(
    prefix: str,
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> None:
    _assert_close(f"{prefix}_dq", actual[0], expected[0], atol=atol, rtol=rtol)
    _assert_close(f"{prefix}_dk", actual[1], expected[1], atol=atol, rtol=rtol)
    _assert_close(f"{prefix}_dv", actual[2], expected[2], atol=atol, rtol=rtol)


def _dense_sdpa_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        is_causal=causal,
    ).permute(0, 2, 1, 3)


def _manual_local_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    left: int,
    right: int,
    scale: float | None = None,
) -> torch.Tensor:
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    batch, heads, seqlen_q, dim = qh.shape
    seqlen_k = kh.shape[2]
    if kh.shape[1] != heads:
        repeat = heads // kh.shape[1]
        kh = kh.repeat_interleave(repeat, dim=1)
        vh = vh.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2))
    scores *= (scale if scale is not None else dim ** -0.5)
    rel = torch.arange(seqlen_k, device=q.device).view(1, 1, 1, seqlen_k) - (
        torch.arange(seqlen_q, device=q.device).view(1, 1, seqlen_q, 1) + seqlen_k - seqlen_q
    )
    allowed = (rel >= -left) & (rel <= right)
    scores = scores.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vh.float()).to(q.dtype).permute(0, 2, 1, 3)


def _manual_dense_softcap_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    scores = torch.tanh(scores / softcap) * softcap
    if causal:
        mask = torch.triu(
            torch.ones((q.shape[1], k.shape[1]), device=q.device, dtype=torch.bool),
            diagonal=1 + k.shape[1] - q.shape[1],
        )
        scores = scores.masked_fill(mask.view(1, 1, q.shape[1], k.shape[1]), float("-inf"))
    probs, lse = _manual_safe_probs_and_lse(scores)
    out = torch.matmul(probs, vh.float()).to(q.dtype).permute(0, 2, 1, 3)
    return out, lse.permute(0, 2, 1).contiguous()


def _manual_safe_probs_and_lse(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = ~torch.isneginf(scores)
    all_masked = ~valid.any(dim=-1, keepdim=True)
    row_max = torch.amax(scores, dim=-1, keepdim=True)
    safe_row_max = torch.where(all_masked, torch.zeros_like(row_max), row_max)
    exp_scores = torch.where(valid, torch.exp(scores - safe_row_max), torch.zeros_like(scores))
    normalizer = exp_scores.sum(dim=-1, keepdim=True)
    probs = torch.where(normalizer > 0, exp_scores / normalizer, torch.zeros_like(exp_scores))
    lse = torch.where(
        normalizer > 0,
        torch.log(normalizer) + safe_row_max,
        torch.full_like(safe_row_max, float("-inf")),
    ).squeeze(-1)
    return probs, lse


def _manual_combine_ref(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    ).to(out_partial.dtype)
    lse = torch.where(
        denom > 0,
        torch.log(denom) + safe_lse_max,
        torch.full_like(safe_lse_max, float("-inf")),
    )
    return out, lse


def _manual_mask_mod_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    q_idx = torch.arange(q.shape[1], device=q.device, dtype=torch.long).view(1, 1, q.shape[1], 1)
    kv_idx = torch.arange(k.shape[1], device=k.device, dtype=torch.long).view(1, 1, 1, k.shape[1])
    offset = k.shape[1] - q.shape[1]
    keep = (kv_idx <= (q_idx + offset)) & (q_idx > 0)
    scores = scores.masked_fill(~keep, float("-inf"))
    probs, lse = _manual_safe_probs_and_lse(scores)
    out = torch.matmul(probs, vh.float()).to(q.dtype).permute(0, 2, 1, 3)
    return out, lse.permute(0, 2, 1).contiguous()


def _manual_learnable_sink_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    learnable_sink: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    if kh.shape[1] != qh.shape[1]:
        repeat = qh.shape[1] // kh.shape[1]
        kh = kh.repeat_interleave(repeat, dim=1)
        vh = vh.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    if causal:
        mask = torch.triu(
            torch.ones((q.shape[1], k.shape[1]), device=q.device, dtype=torch.bool),
            diagonal=1 + k.shape[1] - q.shape[1],
        )
        scores = scores.masked_fill(mask.view(1, 1, q.shape[1], k.shape[1]), float("-inf"))
    sink = learnable_sink.float().view(1, qh.shape[1], 1, 1)
    logits_max = torch.amax(scores, dim=-1, keepdim=True)
    logits_or_sinks_max = torch.maximum(sink, logits_max)
    unnormalized = torch.exp(scores - logits_or_sinks_max)
    normalizer = unnormalized.sum(dim=-1, keepdim=True) + torch.exp(sink - logits_or_sinks_max)
    probs = unnormalized / normalizer
    out = torch.matmul(probs, vh.float()).to(q.dtype).permute(0, 2, 1, 3)
    lse = (torch.log(normalizer) + logits_or_sinks_max).squeeze(-1).permute(0, 2, 1).contiguous()
    return out, lse


def _manual_varlen_ref(
    flash_attn_varlen_func,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = flash_attn_varlen_func(q, k, v, cu_q, cu_k, causal=True, return_lse=True)
    return out, lse


def _manual_varlen_score_mod_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    token_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    lse_chunks = []
    for batch_idx in range(cu_q.numel() - 1):
        qs, qe = int(cu_q[batch_idx].item()), int(cu_q[batch_idx + 1].item())
        ks, ke = int(cu_k[batch_idx].item()), int(cu_k[batch_idx + 1].item())
        q_chunk = q[qs:qe].unsqueeze(0)
        k_chunk = k[ks:ke].unsqueeze(0)
        v_chunk = v[ks:ke].unsqueeze(0)
        qh = q_chunk.permute(0, 2, 1, 3)
        kh = k_chunk.permute(0, 2, 1, 3)
        vh = v_chunk.permute(0, 2, 1, 3)
        scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        bias = token_bias[ks:ke].to(torch.float32).view(1, 1, 1, ke - ks)
        scores = scores + bias
        probs, lse = _manual_safe_probs_and_lse(scores)
        out = torch.matmul(probs, vh.float()).to(q.dtype).permute(0, 2, 1, 3)
        outputs.append(out.squeeze(0))
        lse_chunks.append(lse.permute(0, 2, 1).squeeze(0).contiguous())
    return torch.cat(outputs, dim=0), torch.cat(lse_chunks, dim=0)


def _manual_varlen_softcap_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    causal: bool,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    lse_chunks = []
    for batch_idx in range(cu_q.numel() - 1):
        qs, qe = int(cu_q[batch_idx].item()), int(cu_q[batch_idx + 1].item())
        ks, ke = int(cu_k[batch_idx].item()), int(cu_k[batch_idx + 1].item())
        out, lse = _manual_dense_softcap_ref(
            q[qs:qe].unsqueeze(0),
            k[ks:ke].unsqueeze(0),
            v[ks:ke].unsqueeze(0),
            causal=causal,
            softcap=softcap,
        )
        outputs.append(out.squeeze(0))
        lse_chunks.append(lse.squeeze(0))
    return torch.cat(outputs, dim=0), torch.cat(lse_chunks, dim=0)


def _manual_varlen_aux_score_mod_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    local_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    lse_chunks = []
    for batch_idx in range(cu_q.numel() - 1):
        qs, qe = int(cu_q[batch_idx].item()), int(cu_q[batch_idx + 1].item())
        ks, ke = int(cu_k[batch_idx].item()), int(cu_k[batch_idx + 1].item())
        q_chunk = q[qs:qe].unsqueeze(0)
        k_chunk = k[ks:ke].unsqueeze(0)
        v_chunk = v[ks:ke].unsqueeze(0)
        qh = q_chunk.permute(0, 2, 1, 3)
        kh = k_chunk.permute(0, 2, 1, 3)
        vh = v_chunk.permute(0, 2, 1, 3)
        scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        bias = local_bias[: ke - ks].to(torch.float32).view(1, 1, 1, ke - ks)
        scores = scores + bias
        probs, lse = _manual_safe_probs_and_lse(scores)
        out = torch.matmul(probs, vh.float()).to(q.dtype).permute(0, 2, 1, 3)
        outputs.append(out.squeeze(0))
        lse_chunks.append(lse.permute(0, 2, 1).squeeze(0).contiguous())
    return torch.cat(outputs, dim=0), torch.cat(lse_chunks, dim=0)


def _manual_varlen_seqused_k_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    seqused_k: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    lse_chunks = []
    for batch_idx in range(cu_q.numel() - 1):
        qs, qe = int(cu_q[batch_idx].item()), int(cu_q[batch_idx + 1].item())
        ks = int(cu_k[batch_idx].item())
        ke = ks + int(seqused_k[batch_idx].item())
        q_chunk = q[qs:qe].unsqueeze(0)
        k_chunk = k[ks:ke].unsqueeze(0)
        v_chunk = v[ks:ke].unsqueeze(0)
        qh = q_chunk.permute(0, 2, 1, 3)
        kh = k_chunk.permute(0, 2, 1, 3)
        vh = v_chunk.permute(0, 2, 1, 3)
        scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        if causal:
            mask = torch.triu(
                torch.ones((q_chunk.shape[1], k_chunk.shape[1]), device=q.device, dtype=torch.bool),
                diagonal=1 + k_chunk.shape[1] - q_chunk.shape[1],
            )
            scores = scores.masked_fill(
                mask.view(1, 1, q_chunk.shape[1], k_chunk.shape[1]),
                float("-inf"),
            )
        probs, lse = _manual_safe_probs_and_lse(scores)
        out = torch.matmul(probs, vh.float()).to(q.dtype).permute(0, 2, 1, 3)
        outputs.append(out.squeeze(0))
        lse_chunks.append(lse.permute(0, 2, 1).squeeze(0).contiguous())
    return torch.cat(outputs, dim=0), torch.cat(lse_chunks, dim=0)


def _manual_varlen_layout_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_q: torch.Tensor | None = None,
    cu_k: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    causal: bool,
    q_global_bias: torch.Tensor | None = None,
    kv_global_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_q is not None:
        batch = cu_q.numel() - 1
        q_packed = True
    else:
        batch = q.shape[0]
        q_packed = False
    if cu_k is not None and cu_k.numel() - 1 != batch:
        raise ValueError("q and k layouts must describe the same batch")
    if cu_k is None and k.shape[0] != batch:
        raise ValueError("q and k layouts must describe the same batch")

    outputs = []
    lse_chunks = []
    q_running = 0
    k_running = 0
    q_full_len = q.shape[1] if not q_packed else None

    for batch_idx in range(batch):
        if cu_q is not None:
            q_offset = int(cu_q[batch_idx].item())
            q_used = int(cu_q[batch_idx + 1].item()) - q_offset
            if seqused_q is not None:
                q_used = min(q_used, int(seqused_q[batch_idx].item()))
            q_chunk = q[q_offset : q_offset + q_used].unsqueeze(0)
        else:
            q_offset = q_running
            q_used = int(seqused_q[batch_idx].item()) if seqused_q is not None else q.shape[1]
            q_used = min(q_used, q.shape[1])
            q_chunk = q[batch_idx : batch_idx + 1, :q_used]
            q_running += q_used

        if cu_k is not None:
            k_offset = int(cu_k[batch_idx].item())
            k_used = int(cu_k[batch_idx + 1].item()) - k_offset
            if seqused_k is not None:
                k_used = min(k_used, int(seqused_k[batch_idx].item()))
            k_chunk = k[k_offset : k_offset + k_used].unsqueeze(0)
            v_chunk = v[k_offset : k_offset + k_used].unsqueeze(0)
        else:
            k_offset = k_running
            k_used = int(seqused_k[batch_idx].item()) if seqused_k is not None else k.shape[1]
            k_used = min(k_used, k.shape[1])
            k_chunk = k[batch_idx : batch_idx + 1, :k_used]
            v_chunk = v[batch_idx : batch_idx + 1, :k_used]
            k_running += k_used

        if q_used == 0 or k_used == 0:
            out_piece = q_chunk.new_zeros((q_used, q.shape[-2] if q_packed else q.shape[2], v.shape[-1]))
            lse_piece = torch.full(
                (q_used, q.shape[-2] if q_packed else q.shape[2]),
                float("-inf"),
                device=q.device,
                dtype=torch.float32,
            )
        else:
            qh = q_chunk.permute(0, 2, 1, 3)
            kh = k_chunk.permute(0, 2, 1, 3)
            vh = v_chunk.permute(0, 2, 1, 3)
            if kh.shape[1] != qh.shape[1]:
                repeat = qh.shape[1] // kh.shape[1]
                kh = kh.repeat_interleave(repeat, dim=1)
                vh = vh.repeat_interleave(repeat, dim=1)
            scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
            if q_global_bias is not None:
                scores = scores + q_global_bias[q_offset : q_offset + q_used].to(torch.float32).view(
                    1, 1, q_used, 1
                )
            if kv_global_bias is not None:
                scores = scores + kv_global_bias[k_offset : k_offset + k_used].to(torch.float32).view(
                    1, 1, 1, k_used
                )
            if causal:
                mask = torch.triu(
                    torch.ones((q_used, k_used), device=q.device, dtype=torch.bool),
                    diagonal=1 + k_used - q_used,
                )
                scores = scores.masked_fill(mask.view(1, 1, q_used, k_used), float("-inf"))
            probs, lse = _manual_safe_probs_and_lse(scores)
            out_piece = torch.matmul(probs, vh.float()).to(q.dtype).permute(0, 2, 1, 3).squeeze(0)
            lse_piece = lse.permute(0, 2, 1).squeeze(0).contiguous()

        if q_packed:
            outputs.append(out_piece)
            lse_chunks.append(lse_piece)
            continue

        pad_q = q_full_len - q_used
        if pad_q > 0:
            out_piece = torch.cat(
                [out_piece, out_piece.new_zeros((pad_q, out_piece.shape[1], out_piece.shape[2]))],
                dim=0,
            )
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
        lse_chunks.append(lse_piece)

    if q_packed:
        return torch.cat(outputs, dim=0), torch.cat(lse_chunks, dim=0)
    return torch.stack(outputs, dim=0), torch.stack(lse_chunks, dim=0)


def main() -> int:
    _add_shim_path()
    from flash_attn.cute import (
        compute_block_sparsity,
        fast_sampling,
        flash_attn_combine,
        flash_attn_func,
        flash_attn_varlen_func,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation script")

    _manual_seed()

    out_partial = torch.randn(3, 2, 4, 3, 8, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.randn(3, 2, 4, 3, device="cuda", dtype=torch.float32)
    lse_partial[2, 1, 3, :] = float("-inf")
    out, lse = flash_attn_combine(out_partial, lse_partial, return_lse=True)
    ref_out, ref_lse = _manual_combine_ref(out_partial, lse_partial)
    _assert_close("combine_batched_out", out, ref_out, atol=0.0, rtol=0.0)
    _assert_close("combine_batched_lse", lse, ref_lse, atol=0.0, rtol=0.0)

    out_partial = torch.randn(4, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.randn(4, 7, 2, device="cuda", dtype=torch.float32)
    lse_partial[:, 5, 1] = float("-inf")
    out, lse = flash_attn_combine(out_partial, lse_partial, return_lse=True)
    ref_out, ref_lse = _manual_combine_ref(out_partial, lse_partial)
    _assert_close("combine_varlen_out", out, ref_out, atol=0.0, rtol=0.0)
    _assert_close("combine_varlen_lse", lse, ref_lse, atol=0.0, rtol=0.0)

    def causal_block_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del batch_idx, head_idx, seqlen_info, aux_tensors
        return kv_idx <= q_idx

    _, sparse_tensors = compute_block_sparsity(
        4,
        4,
        1,
        1,
        8,
        8,
        causal_block_mask,
        None,
        torch.device("cuda"),
        compute_full_blocks=True,
    )
    expected_mask_cnt = torch.tensor([[[1, 1]]], device="cuda", dtype=torch.int32)
    expected_mask_idx = torch.tensor([[[[0, 0], [1, 0]]]], device="cuda", dtype=torch.int32)
    expected_full_cnt = torch.tensor([[[0, 1]]], device="cuda", dtype=torch.int32)
    expected_full_idx = torch.tensor([[[[0, 0], [0, 0]]]], device="cuda", dtype=torch.int32)
    _assert_close("block_sparse_mask_cnt", sparse_tensors.mask_block_cnt, expected_mask_cnt, atol=0.0, rtol=0.0)
    _assert_close("block_sparse_mask_idx", sparse_tensors.mask_block_idx, expected_mask_idx, atol=0.0, rtol=0.0)
    _assert_close("block_sparse_full_cnt", sparse_tensors.full_block_cnt, expected_full_cnt, atol=0.0, rtol=0.0)
    _assert_close("block_sparse_full_idx", sparse_tensors.full_block_idx, expected_full_idx, atol=0.0, rtol=0.0)

    @fast_sampling
    def head_bias_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del batch_idx, seqlen_info
        return kv_idx <= (q_idx + aux_tensors[0][head_idx])

    head_bias = torch.tensor([0, 1], device="cuda", dtype=torch.long)
    _, sparse_tensors = compute_block_sparsity(
        2,
        2,
        1,
        2,
        4,
        4,
        head_bias_mask,
        [head_bias],
        torch.device("cuda"),
        compute_full_blocks=True,
        use_fast_sampling=True,
    )
    expected_mask_cnt = torch.tensor([[[1, 1], [1, 0]]], device="cuda", dtype=torch.int32)
    expected_mask_idx = torch.tensor(
        [[[[0, 0], [1, 0]], [[1, 0], [0, 0]]]],
        device="cuda",
        dtype=torch.int32,
    )
    expected_full_cnt = torch.tensor([[[0, 1], [1, 2]]], device="cuda", dtype=torch.int32)
    expected_full_idx = torch.tensor(
        [[[[0, 0], [0, 0]], [[0, 0], [0, 1]]]],
        device="cuda",
        dtype=torch.int32,
    )
    _assert_close("block_sparse_fast_mask_cnt", sparse_tensors.mask_block_cnt, expected_mask_cnt, atol=0.0, rtol=0.0)
    _assert_close("block_sparse_fast_mask_idx", sparse_tensors.mask_block_idx, expected_mask_idx, atol=0.0, rtol=0.0)
    _assert_close("block_sparse_fast_full_cnt", sparse_tensors.full_block_cnt, expected_full_cnt, atol=0.0, rtol=0.0)
    _assert_close("block_sparse_fast_full_idx", sparse_tensors.full_block_idx, expected_full_idx, atol=0.0, rtol=0.0)

    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = flash_attn_func(q, k, v, causal=True)
    ref = _dense_sdpa_ref(q, k, v, causal=True)
    _assert_close("dense_forward", out, ref, atol=0.0, rtol=0.0)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    shim_loss = out.float().sum()
    shim_loss.backward()
    ref_out = _dense_sdpa_ref(q_ref, k_ref, v_ref, causal=True)
    ref_out.float().sum().backward()
    _assert_grad_close(
        "dense",
        (q.grad, k.grad, v.grad),
        (q_ref.grad, k_ref.grad, v_ref.grad),
        atol=0.0,
        rtol=0.0,
    )

    _manual_seed()
    q = torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.bfloat16)
    out, _ = flash_attn_func(q, k, v, window_size=(2, 1))
    ref = _manual_local_ref(q, k, v, left=2, right=1)
    _assert_close("local_window", out, ref, atol=0.02, rtol=0.0)

    _manual_seed()
    q = torch.randn(1, 13, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 13, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 13, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    softcap = 3.5
    out, lse = flash_attn_func(q, k, v, causal=True, softcap=softcap, return_lse=True)
    ref, ref_lse = _manual_dense_softcap_ref(q, k, v, causal=True, softcap=softcap)
    _assert_close("softcap_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("softcap_lse", lse, ref_lse, atol=1e-4, rtol=0.0)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out.float().sum().backward()
    ref_out, _ = _manual_dense_softcap_ref(q_ref, k_ref, v_ref, causal=True, softcap=softcap)
    ref_out.float().sum().backward()
    _assert_grad_close(
        "softcap",
        (q.grad, k.grad, v.grad),
        (q_ref.grad, k_ref.grad, v_ref.grad),
        atol=0.0,
        rtol=0.0,
    )

    _manual_seed()
    q = torch.randn(1, 7, 3, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 10, 3, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 10, 3, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def dense_mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del batch_idx, head_idx, aux_tensors
        offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        return (kv_idx <= (q_idx + offset)) & (q_idx > 0)

    out, lse = flash_attn_func(q, k, v, mask_mod=dense_mask_mod, return_lse=True)
    ref, ref_lse = _manual_mask_mod_ref(q, k, v)
    _assert_close("mask_mod_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("mask_mod_lse", lse, ref_lse, atol=0.0, rtol=0.0)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out.float().sum().backward()
    ref_out, _ = _manual_mask_mod_ref(q_ref, k_ref, v_ref)
    ref_out.float().sum().backward()
    _assert_grad_close(
        "mask_mod",
        (q.grad, k.grad, v.grad),
        (q_ref.grad, k_ref.grad, v_ref.grad),
        atol=0.0,
        rtol=0.0,
    )

    _manual_seed()
    q = torch.randn(1, 7, 3, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 10, 3, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 10, 3, 32, device="cuda", dtype=torch.bfloat16)

    class UnhashableDenseMaskMod:
        def __call__(self, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
            del batch_idx, head_idx, aux_tensors
            offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
            return (kv_idx <= (q_idx + offset)) & (q_idx > 0)

        def __eq__(self, other):
            return self is other

    out, lse = flash_attn_func(q, k, v, mask_mod=UnhashableDenseMaskMod(), return_lse=True)
    ref, ref_lse = _manual_mask_mod_ref(q, k, v)
    _assert_close("unhashable_mask_mod_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("unhashable_mask_mod_lse", lse, ref_lse, atol=0.0, rtol=0.0)

    _manual_seed()
    q = torch.randn(1, 7, 3, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 10, 3, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 10, 3, 32, device="cuda", dtype=torch.bfloat16)

    def info_named_mask_mod(batch_idx, head_idx, q_idx, kv_idx, info):
        del batch_idx, head_idx
        offset = info.seqlen_k - info.seqlen_q
        return (kv_idx <= (q_idx + offset)) & (q_idx > 0)

    out, lse = flash_attn_func(q, k, v, mask_mod=info_named_mask_mod, return_lse=True)
    ref, ref_lse = _manual_mask_mod_ref(q, k, v)
    _assert_close("info_mask_mod_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("info_mask_mod_lse", lse, ref_lse, atol=0.0, rtol=0.0)

    _manual_seed()
    q = torch.randn(1, 12, 8, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 12, 2, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 12, 2, 16, device="cuda", dtype=torch.bfloat16)
    out, _ = flash_attn_func(q, k, v, causal=True)
    ref = _dense_sdpa_ref(q, k.repeat_interleave(4, dim=2), v.repeat_interleave(4, dim=2), causal=True)
    _assert_close("mqa_gqa", out, ref, atol=0.0, rtol=0.0)

    _manual_seed()
    q = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(4, device="cuda", dtype=torch.bfloat16)
    out, lse = flash_attn_func(q, k, v, causal=True, learnable_sink=sink, return_lse=True)
    ref_out, ref_lse = _manual_learnable_sink_ref(q, k, v, sink, causal=True)
    _assert_close("learnable_sink_out", out, ref_out, atol=0.02, rtol=0.0)
    _assert_close("learnable_sink_lse", lse, ref_lse, atol=1e-4, rtol=0.0)

    _manual_seed()
    lengths_q = [3, 5]
    lengths_k = [4, 5]
    heads = 2
    dim = 8
    q = torch.randn(sum(lengths_q), heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, lengths_q[0], sum(lengths_q)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, lengths_k[0], sum(lengths_k)], device="cuda", dtype=torch.int32)
    out, lse = _manual_varlen_ref(flash_attn_varlen_func, q, k, v, cu_q, cu_k)

    ref_chunks = []
    ref_lse_chunks = []
    for batch_idx in range(len(lengths_q)):
        qs, qe = int(cu_q[batch_idx].item()), int(cu_q[batch_idx + 1].item())
        ks, ke = int(cu_k[batch_idx].item()), int(cu_k[batch_idx + 1].item())
        out_chunk, lse_chunk = flash_attn_func(
            q[qs:qe].unsqueeze(0),
            k[ks:ke].unsqueeze(0),
            v[ks:ke].unsqueeze(0),
            causal=True,
            return_lse=True,
        )
        ref_chunks.append(out_chunk.squeeze(0))
        ref_lse_chunks.append(lse_chunk.squeeze(0))
    ref = torch.cat(ref_chunks, dim=0)
    ref_lse = torch.cat(ref_lse_chunks, dim=0)
    _assert_close("varlen_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("varlen_lse", lse, ref_lse, atol=0.0, rtol=0.0)

    _manual_seed()
    lengths_q = [3, 4]
    lengths_k = [5, 6]
    heads = 2
    dim = 16
    q = torch.randn(sum(lengths_q), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu_q = torch.tensor([0, lengths_q[0], sum(lengths_q)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, lengths_k[0], sum(lengths_k)], device="cuda", dtype=torch.int32)
    softcap = 2.25
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        causal=True,
        softcap=softcap,
        return_lse=True,
    )
    ref, ref_lse = _manual_varlen_softcap_ref(q, k, v, cu_q, cu_k, causal=True, softcap=softcap)
    _assert_close("varlen_softcap_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("varlen_softcap_lse", lse, ref_lse, atol=1e-4, rtol=0.0)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out.float().sum().backward()
    ref_out, _ = _manual_varlen_softcap_ref(
        q_ref, k_ref, v_ref, cu_q, cu_k, causal=True, softcap=softcap
    )
    ref_out.float().sum().backward()
    _assert_grad_close(
        "varlen_softcap",
        (q.grad, k.grad, v.grad),
        (q_ref.grad, k_ref.grad, v_ref.grad),
        atol=0.0,
        rtol=0.0,
    )

    _manual_seed()
    lengths_q = [3, 5]
    lengths_k = [5, 6]
    heads = 2
    dim = 8
    q = torch.randn(sum(lengths_q), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu_q = torch.tensor([0, lengths_q[0], sum(lengths_q)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, lengths_k[0], sum(lengths_k)], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor([3, 4], device="cuda", dtype=torch.int32)
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        seqused_k=seqused_k,
        causal=True,
        return_lse=True,
    )
    ref, ref_lse = _manual_varlen_seqused_k_ref(q, k, v, cu_q, cu_k, seqused_k, causal=True)
    _assert_close("varlen_seqused_k_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("varlen_seqused_k_lse", lse, ref_lse, atol=0.0, rtol=0.0)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out.float().sum().backward()
    ref_out, _ = _manual_varlen_seqused_k_ref(
        q_ref, k_ref, v_ref, cu_q, cu_k, seqused_k, causal=True
    )
    ref_out.float().sum().backward()
    _assert_grad_close(
        "varlen_seqused_k",
        (q.grad, k.grad, v.grad),
        (q_ref.grad, k_ref.grad, v_ref.grad),
        atol=0.0,
        rtol=0.0,
    )

    _manual_seed()
    lengths_q = [4, 3]
    physical_k = 6
    used_k = torch.tensor([5, 2], device="cuda", dtype=torch.int32)
    heads = 2
    dim = 16
    q = torch.randn(sum(lengths_q), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, physical_k, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, physical_k, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu_q = torch.tensor([0, lengths_q[0], sum(lengths_q)], device="cuda", dtype=torch.int32)
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        seqused_k=used_k,
        causal=True,
        return_lse=True,
    )
    ref, ref_lse = _manual_varlen_layout_ref(
        q,
        k,
        v,
        cu_q=cu_q,
        seqused_k=used_k,
        causal=True,
    )
    _assert_close("mixed_q_packed_kv_padded_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("mixed_q_packed_kv_padded_lse", lse, ref_lse, atol=0.0, rtol=0.0)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out.float().sum().backward()
    ref_out, _ = _manual_varlen_layout_ref(
        q_ref,
        k_ref,
        v_ref,
        cu_q=cu_q,
        seqused_k=used_k,
        causal=True,
    )
    ref_out.float().sum().backward()
    _assert_grad_close(
        "mixed_q_packed_kv_padded",
        (q.grad, k.grad, v.grad),
        (q_ref.grad, k_ref.grad, v_ref.grad),
        atol=0.0,
        rtol=0.0,
    )

    _manual_seed()
    physical_q = 6
    lengths_k = [5, 4]
    used_q = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    heads = 2
    dim = 16
    q = torch.randn(2, physical_q, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu_k = torch.tensor([0, lengths_k[0], sum(lengths_k)], device="cuda", dtype=torch.int32)
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_k=cu_k,
        seqused_q=used_q,
        causal=True,
        return_lse=True,
    )
    ref, ref_lse = _manual_varlen_layout_ref(
        q,
        k,
        v,
        cu_k=cu_k,
        seqused_q=used_q,
        causal=True,
    )
    _assert_close("mixed_q_padded_kv_packed_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("mixed_q_padded_kv_packed_lse", lse, ref_lse, atol=0.0, rtol=0.0)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out.float().sum().backward()
    ref_out, _ = _manual_varlen_layout_ref(
        q_ref,
        k_ref,
        v_ref,
        cu_k=cu_k,
        seqused_q=used_q,
        causal=True,
    )
    ref_out.float().sum().backward()
    _assert_grad_close(
        "mixed_q_padded_kv_packed",
        (q.grad, k.grad, v.grad),
        (q_ref.grad, k_ref.grad, v_ref.grad),
        atol=0.0,
        rtol=0.0,
    )

    _manual_seed()
    used_q = torch.tensor([3, 4], device="cuda", dtype=torch.int32)
    used_k = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
    physical_q = 5
    physical_k = 6
    heads = 2
    dim = 8
    q = torch.randn(2, physical_q, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, physical_k, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, physical_k, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q_bias = torch.linspace(-0.3, 0.4, steps=int(used_q.sum().item()), device="cuda", dtype=torch.float32)
    kv_bias = torch.linspace(-0.2, 0.5, steps=int(used_k.sum().item()), device="cuda", dtype=torch.float32)

    def mixed_info_score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, info):
        del batch_idx, head_idx
        return (
            scores
            + q_bias[q_idx + info.offset_q].to(torch.float32)
            + kv_bias[kv_idx + info.offset_k].to(torch.float32)
        )

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        seqused_q=used_q,
        seqused_k=used_k,
        causal=True,
        score_mod=mixed_info_score_mod,
        return_lse=True,
    )
    ref, ref_lse = _manual_varlen_layout_ref(
        q,
        k,
        v,
        seqused_q=used_q,
        seqused_k=used_k,
        causal=True,
        q_global_bias=q_bias,
        kv_global_bias=kv_bias,
    )
    _assert_close("mixed_seqused_qk_score_mod_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("mixed_seqused_qk_score_mod_lse", lse, ref_lse, atol=0.0, rtol=0.0)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out.float().sum().backward()
    ref_out, _ = _manual_varlen_layout_ref(
        q_ref,
        k_ref,
        v_ref,
        seqused_q=used_q,
        seqused_k=used_k,
        causal=True,
        q_global_bias=q_bias,
        kv_global_bias=kv_bias,
    )
    ref_out.float().sum().backward()
    _assert_grad_close(
        "mixed_seqused_qk_score_mod",
        (q.grad, k.grad, v.grad),
        (q_ref.grad, k_ref.grad, v_ref.grad),
        atol=0.0,
        rtol=0.0,
    )

    _manual_seed()
    lengths_q = [2, 3]
    lengths_k = [3, 4]
    heads = 2
    dim = 16
    q = torch.randn(sum(lengths_q), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu_q = torch.tensor([0, lengths_q[0], sum(lengths_q)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, lengths_k[0], sum(lengths_k)], device="cuda", dtype=torch.int32)
    token_bias = torch.linspace(-0.5, 0.5, steps=sum(lengths_k), device="cuda", dtype=torch.float32)

    def global_kv_score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del batch_idx, head_idx, q_idx
        bias = aux_tensors[0][kv_idx + seqlen_info.offset_k].to(torch.float32)
        return scores + bias

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        score_mod=global_kv_score_mod,
        aux_tensors=[token_bias],
        return_lse=True,
    )
    ref, ref_lse = _manual_varlen_score_mod_ref(q, k, v, cu_q, cu_k, token_bias)
    _assert_close("varlen_score_mod_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("varlen_score_mod_lse", lse, ref_lse, atol=0.0, rtol=0.0)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out.float().sum().backward()
    ref_out, _ = _manual_varlen_score_mod_ref(q_ref, k_ref, v_ref, cu_q, cu_k, token_bias)
    ref_out.float().sum().backward()
    _assert_grad_close(
        "varlen_score_mod",
        (q.grad, k.grad, v.grad),
        (q_ref.grad, k_ref.grad, v_ref.grad),
        atol=0.0,
        rtol=0.0,
    )

    _manual_seed()
    lengths_q = [2, 3]
    lengths_k = [3, 4]
    heads = 2
    dim = 16
    q = torch.randn(sum(lengths_q), heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(sum(lengths_k), heads, dim, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, lengths_q[0], sum(lengths_q)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, lengths_k[0], sum(lengths_k)], device="cuda", dtype=torch.int32)
    local_bias = torch.linspace(-0.25, 0.25, steps=max(lengths_k), device="cuda", dtype=torch.float32)

    def aux_only_score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
        del batch_idx, head_idx, q_idx
        return scores + aux_tensors[0][kv_idx].to(torch.float32)

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        score_mod=aux_only_score_mod,
        aux_tensors=[local_bias],
        return_lse=True,
    )
    ref_aux, ref_aux_lse = _manual_varlen_aux_score_mod_ref(q, k, v, cu_q, cu_k, local_bias)
    _assert_close("varlen_aux_score_mod_out", out, ref_aux, atol=0.0, rtol=0.0)
    _assert_close("varlen_aux_score_mod_lse", lse, ref_aux_lse, atol=0.0, rtol=0.0)

    token_bias_info = torch.linspace(-0.5, 0.5, steps=sum(lengths_k), device="cuda", dtype=torch.float32)

    def info_named_score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, info):
        del batch_idx, head_idx, q_idx
        return scores + token_bias_info[kv_idx + info.offset_k].to(torch.float32)

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        score_mod=info_named_score_mod,
        return_lse=True,
    )
    ref, ref_lse = _manual_varlen_score_mod_ref(q, k, v, cu_q, cu_k, token_bias_info)
    _assert_close("info_varlen_score_mod_out", out, ref, atol=0.0, rtol=0.0)
    _assert_close("info_varlen_score_mod_lse", lse, ref_lse, atol=0.0, rtol=0.0)

    class UnhashableAuxScoreMod:
        def __call__(self, scores, batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
            del batch_idx, head_idx, q_idx
            return scores + aux_tensors[0][kv_idx].to(torch.float32)

        def __eq__(self, other):
            return self is other

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        score_mod=UnhashableAuxScoreMod(),
        aux_tensors=[local_bias],
        return_lse=True,
    )
    _assert_close("unhashable_varlen_aux_score_mod_out", out, ref_aux, atol=0.0, rtol=0.0)
    _assert_close("unhashable_varlen_aux_score_mod_lse", lse, ref_aux_lse, atol=0.0, rtol=0.0)

    print("validation=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
