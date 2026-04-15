"""Run forward parity probes through the isolated native FA4 bridge path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_windows_shim_module():
    repo_root = _repo_root()
    shim_init = repo_root / "shims" / "flash_attn" / "cute" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "native_probe_windows_flash_attn_cute_shim_forward_probe",
        shim_init,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Windows shim module from {shim_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _print_case_metrics(
    name: str,
    native_out: torch.Tensor,
    ref_out: torch.Tensor,
    native_lse: torch.Tensor | None = None,
    ref_lse: torch.Tensor | None = None,
) -> None:
    diff = (native_out.float() - ref_out.float()).abs()
    print(f"case={name}")
    print(f"{name}_out_max_diff={diff.max().item()}")
    print(f"{name}_out_mean_diff={diff.mean().item()}")
    print(f"{name}_out_exact={bool(diff.max().item() == 0.0)}")
    print(f"{name}_out_finite={bool(torch.isfinite(native_out).all().item())}")
    if native_lse is not None and ref_lse is not None:
        if native_lse.shape != ref_lse.shape:
            if native_lse.ndim == 3 and ref_lse.ndim == 3:
                ref_lse = ref_lse.permute(0, 2, 1).contiguous()
            elif native_lse.ndim == 2 and ref_lse.ndim == 2:
                ref_lse = ref_lse.transpose(-1, -2).contiguous()
        both_neginf = torch.isneginf(native_lse) & torch.isneginf(ref_lse)
        native_lse_cmp = native_lse.float().masked_fill(both_neginf, 0.0)
        ref_lse_cmp = ref_lse.float().masked_fill(both_neginf, 0.0)
        lse_diff = (native_lse_cmp - ref_lse_cmp).abs()
        print(f"{name}_lse_max_diff={lse_diff.max().item()}")
        print(f"{name}_lse_mean_diff={lse_diff.mean().item()}")
        print(f"{name}_lse_exact={bool(lse_diff.max().item() == 0.0)}")
        print(f"{name}_lse_finite={bool(torch.isfinite(native_lse).logical_or(torch.isneginf(native_lse)).all().item())}")


def _dense_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        is_causal=True,
    ).permute(0, 2, 1, 3)


def _striped_mask(batch_idx, head_idx, q_idx, kv_idx):
    del batch_idx
    return (kv_idx == 0) | ((kv_idx <= q_idx) & (((q_idx + kv_idx + head_idx) % 3) != 1))


def _varlen_score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info):
    del batch_idx
    q_global = q_idx + seqlen_info.offset_q
    k_global = kv_idx + seqlen_info.offset_k
    return scores + (head_idx.to(scores.dtype) * 0.02) + ((q_global - k_global).to(scores.dtype) * 0.01)


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


def _run_dense_base(native_flash_attn_func):
    torch.manual_seed(0)
    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    native_out, native_lse = native_flash_attn_func(q, k, v, causal=True, return_lse=True)
    ref_out = _dense_reference(q, k, v)

    print(f"out_device={native_out.device}")
    print(f"out_dtype={native_out.dtype}")
    print(f"lse_type={type(native_lse).__name__}")
    if native_lse is not None:
        print(f"lse_shape={tuple(native_lse.shape)}")
    _print_case_metrics("dense_base", native_out, ref_out)
    print(f"dense_base_sum_out={native_out.float().sum().item()}")
    print(f"dense_base_sum_ref={ref_out.float().sum().item()}")


def _run_dense_softcap(native_flash_attn_func, shim_mod):
    torch.manual_seed(7)
    q = torch.randn(1, 24, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 24, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 24, 2, 64, device="cuda", dtype=torch.bfloat16)
    native_out, native_lse = native_flash_attn_func(
        q, k, v, causal=True, return_lse=True, softcap=10.0
    )
    ref_out, ref_lse = shim_mod.flash_attn_func(
        q, k, v, causal=True, return_lse=True, softcap=10.0
    )
    _print_case_metrics("dense_softcap", native_out, ref_out, native_lse, ref_lse)


def _run_dense_learnable_sink(native_flash_attn_func, shim_mod):
    torch.manual_seed(13)
    q = torch.randn(1, 18, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 18, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 18, 2, 64, device="cuda", dtype=torch.bfloat16)
    sink = torch.tensor([0.25, -0.5], device="cuda", dtype=torch.bfloat16)
    native_out, native_lse = native_flash_attn_func(
        q, k, v, causal=True, return_lse=True, learnable_sink=sink
    )
    ref_out, ref_lse = shim_mod.flash_attn_func(
        q, k, v, causal=True, return_lse=True, learnable_sink=sink
    )
    _print_case_metrics("dense_learnable_sink", native_out, ref_out, native_lse, ref_lse)


def _run_dense_mask_mod(native_flash_attn_func, shim_mod):
    torch.manual_seed(11)
    q = torch.randn(1, 20, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 20, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 20, 2, 64, device="cuda", dtype=torch.bfloat16)
    native_out, native_lse = native_flash_attn_func(
        q, k, v, causal=False, return_lse=True, mask_mod=_striped_mask
    )
    ref_out, ref_lse = shim_mod.flash_attn_func(
        q, k, v, causal=False, return_lse=True, mask_mod=_striped_mask
    )
    _print_case_metrics("dense_mask_mod", native_out, ref_out, native_lse, ref_lse)


def _run_dense_block_sparse(iface, shim_mod):
    torch.manual_seed(41)
    q = torch.randn(1, 256, 1, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 256, 1, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 256, 1, 32, device="cuda", dtype=torch.bfloat16)

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
    from types import SimpleNamespace
    from cutlass.cute._compile_bridge import NativeProbeForwardBridge

    bridge = NativeProbeForwardBridge(
        SimpleNamespace(
            is_causal=False,
            score_mod=None,
            mask_mod=striped_block_mask,
            tile_m=128,
            tile_n=128,
            q_subtile_factor=2,
        )
    )
    native_out = torch.empty_like(q)
    native_lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]), device="cuda", dtype=torch.float32)
    bridge(
        q,
        k,
        v,
        native_out,
        native_lse,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (
            sparse_tensors.mask_block_cnt,
            sparse_tensors.mask_block_idx,
            sparse_tensors.full_block_cnt,
            sparse_tensors.full_block_idx,
        ),
        None,
    )
    ref_out, ref_lse = shim_mod.flash_attn_func(
        q,
        k,
        v,
        causal=False,
        mask_mod=striped_block_mask,
        full_block_cnt=sparse_tensors.full_block_cnt,
        full_block_idx=sparse_tensors.full_block_idx,
        mask_block_cnt=sparse_tensors.mask_block_cnt,
        mask_block_idx=sparse_tensors.mask_block_idx,
        block_size=sparse_tensors.block_size,
        return_lse=True,
    )
    _print_case_metrics("dense_block_sparse", native_out, ref_out, native_lse, ref_lse)


def _run_varlen_softcap(native_flash_attn_varlen_func, shim_mod):
    torch.manual_seed(19)
    q = torch.randn(9, 2, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(11, 2, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(11, 2, 32, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 4, 9], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 11], device="cuda", dtype=torch.int32)
    native_out, native_lse = native_flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, causal=True, return_lse=True, softcap=12.0
    )
    ref_out, ref_lse = shim_mod.flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, causal=True, return_lse=True, softcap=12.0
    )
    _print_case_metrics("varlen_softcap", native_out, ref_out, native_lse, ref_lse)


def _run_varlen_seqused(native_flash_attn_varlen_func, shim_mod):
    torch.manual_seed(23)
    q = torch.randn(2, 6, 2, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 7, 2, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 7, 2, 32, device="cuda", dtype=torch.bfloat16)
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
    ref_out, ref_lse = shim_mod.flash_attn_varlen_func(
        q,
        k,
        v,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        causal=True,
        return_lse=True,
    )
    _print_case_metrics("varlen_seqused", native_out, ref_out, native_lse, ref_lse)


def _run_varlen_score_mod(native_flash_attn_varlen_func, shim_mod):
    torch.manual_seed(29)
    q = torch.randn(9, 2, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(11, 2, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(11, 2, 32, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 4, 9], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 11], device="cuda", dtype=torch.int32)
    native_out, native_lse = native_flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, causal=True, return_lse=True, score_mod=_varlen_score_mod
    )
    ref_out, ref_lse = shim_mod.flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, causal=True, return_lse=True, score_mod=_varlen_score_mod
    )
    _print_case_metrics("varlen_score_mod", native_out, ref_out, native_lse, ref_lse)


def _run_varlen_seqused_score_mod(native_flash_attn_varlen_func, shim_mod):
    torch.manual_seed(37)
    seqused_q = torch.tensor([3, 4], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
    q = torch.randn(2, 5, 2, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 6, 2, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 6, 2, 16, device="cuda", dtype=torch.bfloat16)
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
    ref_out, ref_lse = shim_mod.flash_attn_varlen_func(
        q,
        k,
        v,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        causal=True,
        score_mod=mixed_info_score_mod,
        return_lse=True,
    )
    _print_case_metrics(
        "varlen_seqused_score_mod",
        native_out,
        ref_out,
        native_lse,
        ref_lse,
    )


def main() -> int:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "native_probe_shims"))
    sys.path.insert(0, str(repo_root / "cutlass_runtime" / "src"))

    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
    import flash_attn.cute.interface as iface
    import cutlass

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native forward probe")

    shim_mod = _load_windows_shim_module()
    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")

    for runner in (
        lambda: _run_dense_base(flash_attn_func),
        lambda: _run_dense_softcap(flash_attn_func, shim_mod),
        lambda: _run_dense_learnable_sink(flash_attn_func, shim_mod),
        lambda: _run_dense_mask_mod(flash_attn_func, shim_mod),
        lambda: _run_dense_block_sparse(iface, shim_mod),
        lambda: _run_varlen_softcap(flash_attn_varlen_func, shim_mod),
        lambda: _run_varlen_seqused(flash_attn_varlen_func, shim_mod),
        lambda: _run_varlen_score_mod(flash_attn_varlen_func, shim_mod),
        lambda: _run_varlen_seqused_score_mod(flash_attn_varlen_func, shim_mod),
    ):
        _clear_compile_caches(iface)
        runner()

    compiled_values = list(iface._flash_attn_fwd.compile_cache.cache.values())
    compiled_types = sorted({type(value).__name__ for value in compiled_values}) if compiled_values else ["missing"]
    print(f"compiled_types={compiled_types}")
    if compiled_values:
        print(f"compiled_repr_sample={repr(compiled_values[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
