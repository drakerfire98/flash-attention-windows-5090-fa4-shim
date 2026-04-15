"""Run backward parity probes through the isolated native FA4 bridge path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from _native_probe_setup import ensure_native_fa4_patch, install_native_probe_paths


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_windows_shim_module():
    repo_root = _repo_root()
    shim_init = repo_root / "shims" / "flash_attn" / "cute" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "native_probe_windows_flash_attn_cute_shim_probe",
        shim_init,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Windows shim module from {shim_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _max_grad_diff(native: tuple[torch.Tensor, ...], ref: tuple[torch.Tensor, ...]) -> float:
    return max((a.float() - b.float()).abs().max().item() for a, b in zip(native, ref))


def _mean_grad_diff(native: tuple[torch.Tensor, ...], ref: tuple[torch.Tensor, ...]) -> float:
    return max((a.float() - b.float()).abs().mean().item() for a, b in zip(native, ref))


def _striped_mask(batch_idx, head_idx, q_idx, kv_idx):
    del batch_idx
    return (kv_idx == 0) | ((kv_idx <= q_idx) & (((q_idx + kv_idx + head_idx) % 3) != 1))


def _varlen_softcap_score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info):
    del batch_idx
    q_global = q_idx + seqlen_info.offset_q
    k_global = kv_idx + seqlen_info.offset_k
    return (
        scores
        + (head_idx.to(scores.dtype) * 0.03)
        + ((2 * q_global - k_global).to(scores.dtype) * 0.015)
    )


def _build_paged_kv_cache(
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    *,
    page_size: int,
    page_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_dense.shape != v_dense.shape:
        raise ValueError("paged KV cache expects dense K and V to share the same shape")
    if k_dense.ndim != 4:
        raise ValueError("paged KV cache expects dense K and V shaped (batch, seqlen, heads, dim)")
    batch, seqlen_k, num_heads, head_dim = k_dense.shape
    if page_table.ndim != 2 or page_table.shape[0] != batch:
        raise ValueError("page_table must be shaped (batch, max_num_pages_per_seq)")

    num_pages = int(page_table.max().item()) + 1
    k_paged = torch.zeros(
        (num_pages, page_size, num_heads, head_dim),
        device=k_dense.device,
        dtype=k_dense.dtype,
    )
    v_paged = torch.zeros_like(k_paged)

    for batch_idx in range(batch):
        for page_offset in range(page_table.shape[1]):
            page_idx = int(page_table[batch_idx, page_offset].item())
            start = page_offset * page_size
            if start >= seqlen_k:
                continue
            end = min(seqlen_k, start + page_size)
            width = end - start
            k_paged[page_idx, :width] = k_dense[batch_idx, start:end]
            v_paged[page_idx, :width] = v_dense[batch_idx, start:end]
    return k_paged, v_paged


def _clear_compile_caches(iface) -> None:
    for attr_name in (
        "_flash_attn_fwd",
        "_flash_attn_bwd",
        "_flash_attn_fwd_combine",
        "_bwd_preprocess",
        "_bwd_postprocess_convert",
    ):
        fn = getattr(iface, attr_name, None)
        compile_cache = getattr(fn, "compile_cache", None)
        cache_dict = getattr(compile_cache, "cache", None)
        if hasattr(cache_dict, "clear"):
            cache_dict.clear()
        elif hasattr(compile_cache, "clear"):
            compile_cache.clear()


def _run_dense(native_flash_attn_func, shim_mod):
    torch.manual_seed(0)
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    native_out, native_lse = native_flash_attn_func(q, k, v, causal=True, return_lse=True)
    native_loss = native_out.float().sum() + 0.05 * native_lse.float().sum()
    native_loss.backward()
    native_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod.flash_attn_func(q_ref, k_ref, v_ref, causal=True, return_lse=True)
    ref_loss = ref_out.float().sum() + 0.05 * ref_lse.float().sum()
    ref_loss.backward()
    ref_grads = (q_ref.grad.detach().clone(), k_ref.grad.detach().clone(), v_ref.grad.detach().clone())

    print("case=dense")
    print(f"dense_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"dense_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"dense_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def _run_varlen(native_flash_attn_varlen_func, shim_mod):
    torch.manual_seed(1)
    q = torch.randn(9, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(11, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(11, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu_q = torch.tensor([0, 4, 9], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 11], device="cuda", dtype=torch.int32)

    native_out, native_lse = native_flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, causal=True, return_lse=True
    )
    native_loss = native_out.float().sum() + 0.05 * native_lse.float().sum()
    native_loss.backward()
    native_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod.flash_attn_varlen_func(
        q_ref, k_ref, v_ref, cu_q, cu_k, causal=True, return_lse=True
    )
    ref_loss = ref_out.float().sum() + 0.05 * ref_lse.float().sum()
    ref_loss.backward()
    ref_grads = (q_ref.grad.detach().clone(), k_ref.grad.detach().clone(), v_ref.grad.detach().clone())

    print("case=varlen")
    print(f"varlen_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"varlen_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"varlen_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def _run_varlen_seqused(native_flash_attn_varlen_func, shim_mod):
    torch.manual_seed(31)
    q = torch.randn(2, 6, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 7, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, 7, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    seqused_q = torch.tensor([4, 6], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor([5, 7], device="cuda", dtype=torch.int32)
    native_out, native_lse = native_flash_attn_varlen_func(
        q,
        k,
        v,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        causal=True,
        return_lse=True,
    )
    native_loss = native_out.float().sum() + 0.05 * native_lse.float().masked_fill(torch.isneginf(native_lse), 0.0).sum()
    native_loss.backward()
    native_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod.flash_attn_varlen_func(
        q_ref,
        k_ref,
        v_ref,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        causal=True,
        return_lse=True,
    )
    ref_loss = ref_out.float().sum() + 0.05 * ref_lse.float().masked_fill(torch.isneginf(ref_lse), 0.0).sum()
    ref_loss.backward()
    ref_grads = (q_ref.grad.detach().clone(), k_ref.grad.detach().clone(), v_ref.grad.detach().clone())

    print("case=varlen_seqused")
    print(f"varlen_seqused_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"varlen_seqused_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"varlen_seqused_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def _run_dense_learnable_sink(native_flash_attn_func, shim_mod):
    torch.manual_seed(13)
    q = torch.randn(1, 18, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 18, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 18, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    sink = torch.tensor([0.25, -0.5], device="cuda", dtype=torch.bfloat16)
    native_out, native_lse = native_flash_attn_func(
        q, k, v, causal=True, return_lse=True, learnable_sink=sink
    )
    native_loss = native_out.float().sum() + 0.05 * native_lse.float().sum()
    native_loss.backward()
    native_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod.flash_attn_func(
        q_ref, k_ref, v_ref, causal=True, return_lse=True, learnable_sink=sink
    )
    ref_loss = ref_out.float().sum() + 0.05 * ref_lse.float().sum()
    ref_loss.backward()
    ref_grads = (q_ref.grad.detach().clone(), k_ref.grad.detach().clone(), v_ref.grad.detach().clone())

    print("case=dense_learnable_sink")
    print(f"dense_learnable_sink_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"dense_learnable_sink_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"dense_learnable_sink_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def _run_dense_mask_mod(native_flash_attn_func, shim_mod):
    torch.manual_seed(11)
    q = torch.randn(1, 20, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 20, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 20, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    native_out, native_lse = native_flash_attn_func(
        q, k, v, causal=False, return_lse=True, mask_mod=_striped_mask
    )
    native_loss = native_out.float().sum() + 0.05 * native_lse.float().masked_fill(torch.isneginf(native_lse), 0.0).sum()
    native_loss.backward()
    native_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod.flash_attn_func(
        q_ref, k_ref, v_ref, causal=False, return_lse=True, mask_mod=_striped_mask
    )
    ref_loss = ref_out.float().sum() + 0.05 * ref_lse.float().masked_fill(torch.isneginf(ref_lse), 0.0).sum()
    ref_loss.backward()
    ref_grads = (q_ref.grad.detach().clone(), k_ref.grad.detach().clone(), v_ref.grad.detach().clone())

    print("case=dense_mask_mod")
    print(f"dense_mask_mod_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"dense_mask_mod_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"dense_mask_mod_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def _run_dense_block_sparse(native_flash_attn_func, shim_mod):
    torch.manual_seed(43)
    q = torch.randn(1, 256, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 256, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 256, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def striped_block_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del batch_idx, head_idx, seqlen_info, aux_tensors
        return (kv_idx <= q_idx) & (((q_idx + kv_idx) % 3) != 1)

    _, sparse_tensors = shim_mod.compute_block_sparsity(
        256,
        128,
        1,
        1,
        256,
        256,
        striped_block_mask,
        None,
        torch.device("cuda"),
        compute_full_blocks=True,
    )

    native_out, native_lse = native_flash_attn_func(
        q,
        k,
        v,
        causal=False,
        return_lse=True,
        mask_mod=striped_block_mask,
        full_block_cnt=sparse_tensors.full_block_cnt,
        full_block_idx=sparse_tensors.full_block_idx,
        mask_block_cnt=sparse_tensors.mask_block_cnt,
        mask_block_idx=sparse_tensors.mask_block_idx,
        block_size=sparse_tensors.block_size,
    )
    native_loss = native_out.float().sum() + 0.05 * native_lse.float().masked_fill(torch.isneginf(native_lse), 0.0).sum()
    native_loss.backward()
    native_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod.flash_attn_func(
        q_ref,
        k_ref,
        v_ref,
        causal=False,
        return_lse=True,
        mask_mod=striped_block_mask,
        full_block_cnt=sparse_tensors.full_block_cnt,
        full_block_idx=sparse_tensors.full_block_idx,
        mask_block_cnt=sparse_tensors.mask_block_cnt,
        mask_block_idx=sparse_tensors.mask_block_idx,
        block_size=sparse_tensors.block_size,
    )
    ref_loss = ref_out.float().sum() + 0.05 * ref_lse.float().masked_fill(torch.isneginf(ref_lse), 0.0).sum()
    ref_loss.backward()
    ref_grads = (q_ref.grad.detach().clone(), k_ref.grad.detach().clone(), v_ref.grad.detach().clone())

    print("case=dense_block_sparse")
    print(f"dense_block_sparse_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"dense_block_sparse_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"dense_block_sparse_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def _run_dense_softcap(native_flash_attn_func, shim_mod):
    torch.manual_seed(5)
    q = torch.randn(1, 12, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 12, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 12, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    native_out, native_lse = native_flash_attn_func(
        q, k, v, causal=True, return_lse=True, softcap=10.0
    )
    native_loss = native_out.float().sum() + 0.05 * native_lse.float().sum()
    native_loss.backward()
    native_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod.flash_attn_func(
        q_ref, k_ref, v_ref, causal=True, return_lse=True, softcap=10.0
    )
    ref_loss = ref_out.float().sum() + 0.05 * ref_lse.float().sum()
    ref_loss.backward()
    ref_grads = (q_ref.grad.detach().clone(), k_ref.grad.detach().clone(), v_ref.grad.detach().clone())

    print("case=dense_softcap")
    print(f"dense_softcap_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"dense_softcap_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"dense_softcap_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def _run_varlen_paged_kv(native_flash_attn_varlen_func, shim_mod):
    torch.manual_seed(43)
    q = torch.randn(2, 6, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k_dense = torch.randn(2, 8, 2, 16, device="cuda", dtype=torch.bfloat16)
    v_dense = torch.randn(2, 8, 2, 16, device="cuda", dtype=torch.bfloat16)
    page_table = torch.tensor([[2, 0], [1, 3]], device="cuda", dtype=torch.int32)
    k_paged_init, v_paged_init = _build_paged_kv_cache(
        k_dense,
        v_dense,
        page_size=4,
        page_table=page_table,
    )
    seqused_q = torch.tensor([5, 6], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor([6, 7], device="cuda", dtype=torch.int32)

    k_paged = k_paged_init.detach().clone().requires_grad_(True)
    v_paged = v_paged_init.detach().clone().requires_grad_(True)
    native_out, native_lse = native_flash_attn_varlen_func(
        q,
        k_paged,
        v_paged,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=page_table,
        causal=True,
        return_lse=True,
    )
    native_loss = native_out.float().sum() + 0.05 * native_lse.float().masked_fill(torch.isneginf(native_lse), 0.0).sum()
    native_loss.backward()
    native_grads = (
        q.grad.detach().clone(),
        k_paged.grad.detach().clone(),
        v_paged.grad.detach().clone(),
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_paged_ref = k_paged_init.detach().clone().requires_grad_(True)
    v_paged_ref = v_paged_init.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod.flash_attn_varlen_func(
        q_ref,
        k_paged_ref,
        v_paged_ref,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=page_table,
        causal=True,
        return_lse=True,
    )
    ref_loss = ref_out.float().sum() + 0.05 * ref_lse.float().masked_fill(torch.isneginf(ref_lse), 0.0).sum()
    ref_loss.backward()
    ref_grads = (
        q_ref.grad.detach().clone(),
        k_paged_ref.grad.detach().clone(),
        v_paged_ref.grad.detach().clone(),
    )

    print("case=varlen_paged_kv")
    print(f"varlen_paged_kv_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"varlen_paged_kv_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"varlen_paged_kv_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def _run_varlen_softcap_score_mod(native_flash_attn_varlen_func, shim_mod):
    torch.manual_seed(47)
    lengths_q = [3, 4]
    lengths_k = [5, 6]
    q = torch.randn(sum(lengths_q), 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(sum(lengths_k), 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(sum(lengths_k), 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu_q = torch.tensor([0, lengths_q[0], sum(lengths_q)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, lengths_k[0], sum(lengths_k)], device="cuda", dtype=torch.int32)

    native_out, native_lse = native_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        causal=True,
        softcap=7.5,
        score_mod=_varlen_softcap_score_mod,
        return_lse=True,
    )
    native_loss = native_out.float().sum() + 0.05 * native_lse.float().masked_fill(torch.isneginf(native_lse), 0.0).sum()
    native_loss.backward()
    native_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod.flash_attn_varlen_func(
        q_ref,
        k_ref,
        v_ref,
        cu_q,
        cu_k,
        causal=True,
        softcap=7.5,
        score_mod=_varlen_softcap_score_mod,
        return_lse=True,
    )
    ref_loss = ref_out.float().sum() + 0.05 * ref_lse.float().masked_fill(torch.isneginf(ref_lse), 0.0).sum()
    ref_loss.backward()
    ref_grads = (q_ref.grad.detach().clone(), k_ref.grad.detach().clone(), v_ref.grad.detach().clone())

    print("case=varlen_softcap_score_mod")
    print(f"varlen_softcap_score_mod_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"varlen_softcap_score_mod_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"varlen_softcap_score_mod_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def _run_varlen_block_sparse_internal(compat_replay_varlen_backward, shim_mod):
    torch.manual_seed(53)
    q = torch.randn(2, 256, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 256, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, 256, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    seqused_q = torch.tensor([160, 256], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor([192, 256], device="cuda", dtype=torch.int32)

    def striped_block_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del batch_idx, head_idx, seqlen_info, aux_tensors
        return (kv_idx <= q_idx) & (((q_idx + kv_idx) % 3) != 1)

    _, sparse_tensors = shim_mod.compute_block_sparsity(
        128,
        128,
        2,
        1,
        256,
        256,
        striped_block_mask,
        None,
        torch.device("cuda"),
        compute_full_blocks=True,
    )

    native_out, native_lse = shim_mod._attention_forward_varlen_block_sparse(
        q,
        k,
        v,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=None,
        causal=False,
        window_size=(None, None),
        learnable_sink=None,
        softcap=0.0,
        score_mod=None,
        mask_mod=striped_block_mask,
        full_block_cnt=sparse_tensors.full_block_cnt,
        full_block_idx=sparse_tensors.full_block_idx,
        mask_block_cnt=sparse_tensors.mask_block_cnt,
        mask_block_idx=sparse_tensors.mask_block_idx,
        block_size=sparse_tensors.block_size,
        aux_tensors=None,
        return_lse=True,
    )
    dout = torch.ones_like(native_out)
    dlse = 0.05 * torch.ones_like(native_lse)
    native_grads = compat_replay_varlen_backward(
        q=q.detach(),
        k=k.detach(),
        v=v.detach(),
        dout=dout,
        dlse=dlse,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=None,
        softmax_scale=None,
        causal=False,
        window_size=(None, None),
        learnable_sink=None,
        softcap=0.0,
        score_mod=None,
        mask_mod=striped_block_mask,
        aux_tensors=None,
        block_sparse_tensors=(
            sparse_tensors.mask_block_cnt,
            sparse_tensors.mask_block_idx,
            sparse_tensors.full_block_cnt,
            sparse_tensors.full_block_idx,
        ),
        block_size=sparse_tensors.block_size,
        return_lse=True,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod._attention_forward_varlen_block_sparse(
        q_ref,
        k_ref,
        v_ref,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=None,
        causal=False,
        window_size=(None, None),
        learnable_sink=None,
        softcap=0.0,
        score_mod=None,
        mask_mod=striped_block_mask,
        full_block_cnt=sparse_tensors.full_block_cnt,
        full_block_idx=sparse_tensors.full_block_idx,
        mask_block_cnt=sparse_tensors.mask_block_cnt,
        mask_block_idx=sparse_tensors.mask_block_idx,
        block_size=sparse_tensors.block_size,
        aux_tensors=None,
        return_lse=True,
    )
    ref_loss = (ref_out * dout).float().sum() + (ref_lse * dlse).float().masked_fill(torch.isneginf(ref_lse), 0.0).sum()
    ref_loss.backward()
    ref_grads = (q_ref.grad.detach().clone(), k_ref.grad.detach().clone(), v_ref.grad.detach().clone())

    print("case=varlen_block_sparse_internal")
    print(f"varlen_block_sparse_internal_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"varlen_block_sparse_internal_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"varlen_block_sparse_internal_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def _run_varlen_seqused_score_mod(native_flash_attn_varlen_func, shim_mod):
    torch.manual_seed(41)
    seqused_q = torch.tensor([3, 4], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
    q = torch.randn(2, 5, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 6, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, 6, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q_bias = torch.linspace(-0.3, 0.4, steps=int(seqused_q.sum().item()), device="cuda", dtype=torch.float32)
    kv_bias = torch.linspace(-0.2, 0.5, steps=int(seqused_k.sum().item()), device="cuda", dtype=torch.float32)

    def mixed_info_score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, info):
        del batch_idx, head_idx
        return (
            scores
            + q_bias[q_idx + info.offset_q].to(torch.float32)
            + kv_bias[kv_idx + info.offset_k].to(torch.float32)
        )

    native_out, native_lse = native_flash_attn_varlen_func(
        q,
        k,
        v,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        causal=True,
        score_mod=mixed_info_score_mod,
        return_lse=True,
    )
    native_loss = native_out.float().sum() + 0.05 * native_lse.float().masked_fill(torch.isneginf(native_lse), 0.0).sum()
    native_loss.backward()
    native_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out, ref_lse = shim_mod.flash_attn_varlen_func(
        q_ref,
        k_ref,
        v_ref,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        causal=True,
        score_mod=mixed_info_score_mod,
        return_lse=True,
    )
    ref_loss = ref_out.float().sum() + 0.05 * ref_lse.float().masked_fill(torch.isneginf(ref_lse), 0.0).sum()
    ref_loss.backward()
    ref_grads = (q_ref.grad.detach().clone(), k_ref.grad.detach().clone(), v_ref.grad.detach().clone())

    print("case=varlen_seqused_score_mod")
    print(f"varlen_seqused_score_mod_out_max_diff={(native_out.float() - ref_out.float()).abs().max().item()}")
    print(f"varlen_seqused_score_mod_grad_max_diff={_max_grad_diff(native_grads, ref_grads)}")
    print(f"varlen_seqused_score_mod_grad_mean_diff={_mean_grad_diff(native_grads, ref_grads)}")


def main() -> int:
    install_native_probe_paths()
    patched_target = ensure_native_fa4_patch()

    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
    import flash_attn.cute.interface as iface
    from cutlass.cute._compile_bridge import compat_replay_varlen_backward

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native backward probe")

    shim_mod = _load_windows_shim_module()
    print(f"patched_interface={patched_target}")
    for runner in (
        lambda: _run_dense(flash_attn_func, shim_mod),
        lambda: _run_dense_softcap(flash_attn_func, shim_mod),
        lambda: _run_dense_learnable_sink(flash_attn_func, shim_mod),
        lambda: _run_dense_mask_mod(flash_attn_func, shim_mod),
        lambda: _run_dense_block_sparse(flash_attn_func, shim_mod),
        lambda: _run_varlen(flash_attn_varlen_func, shim_mod),
        lambda: _run_varlen_seqused(flash_attn_varlen_func, shim_mod),
        lambda: _run_varlen_seqused_score_mod(flash_attn_varlen_func, shim_mod),
        lambda: _run_varlen_paged_kv(flash_attn_varlen_func, shim_mod),
        lambda: _run_varlen_softcap_score_mod(flash_attn_varlen_func, shim_mod),
        lambda: _run_varlen_block_sparse_internal(compat_replay_varlen_backward, shim_mod),
    ):
        _clear_compile_caches(iface)
        runner()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
