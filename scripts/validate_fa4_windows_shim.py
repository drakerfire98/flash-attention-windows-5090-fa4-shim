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
    max_diff = (actual.float() - expected.float()).abs().max().item()
    print(f"{name}_max_diff={max_diff}")
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


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


def main() -> int:
    _add_shim_path()
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation script")

    _manual_seed()

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
    _assert_close("dense_dq", q.grad, q_ref.grad, atol=0.0, rtol=0.0)
    _assert_close("dense_dk", k.grad, k_ref.grad, atol=0.0, rtol=0.0)
    _assert_close("dense_dv", v.grad, v_ref.grad, atol=0.0, rtol=0.0)

    _manual_seed()
    q = torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.bfloat16)
    out, _ = flash_attn_func(q, k, v, window_size=(2, 1))
    ref = _manual_local_ref(q, k, v, left=2, right=1)
    _assert_close("local_window", out, ref, atol=0.02, rtol=0.0)

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

    print("validation=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
