"""Verify persistent export/load across multiple runtime-owned native plan families."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from _native_probe_setup import install_native_probe_paths


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


def _export_reload(plan, path: Path, cute_runtime):
    manifest = plan.export_to_c(object_file_path=str(path), function_name="func")
    loaded = cute_runtime.load_module(str(path), enable_tvm_ffi=True).func
    return manifest, loaded


def _run_forward_case(iface, flash_attn_func, cute_runtime, tmpdir: Path) -> None:
    _clear_compile_caches(iface)
    torch.manual_seed(0)
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)

    expected_out, expected_lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    plan = list(iface._flash_attn_fwd.compile_cache.cache.values())[0]
    manifest, loaded = _export_reload(plan, tmpdir / "forward.o", cute_runtime)

    replay_out = torch.empty_like(expected_out)
    replay_lse = torch.empty_like(expected_lse)
    loaded(
        q.detach(), k.detach(), v.detach(), replay_out, replay_lse,
        None, None, None, None, None, None, None, None, None, None, None,
    )
    out_diff = (replay_out.float() - expected_out.float()).abs()
    lse_diff = (replay_lse.float() - expected_lse.float()).abs()
    print("case=forward")
    print(f"forward_format={manifest['format']}")
    print(f"forward_loaded_type={type(loaded).__name__}")
    print(f"forward_out_max_diff={out_diff.max().item()}")
    print(f"forward_lse_max_diff={lse_diff.max().item()}")


def _run_combine_case(iface, cute_runtime, tmpdir: Path) -> None:
    _clear_compile_caches(iface)
    torch.manual_seed(11)
    out_partial = torch.randn(3, 2, 4, 3, 8, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.randn(3, 2, 4, 3, device="cuda", dtype=torch.float32)
    lse_partial[2, 1, 3, :] = float("-inf")

    expected_out, expected_lse = iface.flash_attn_combine(out_partial, lse_partial, return_lse=True)
    plan = list(iface._flash_attn_fwd_combine.compile_cache.cache.values())[0]
    manifest, loaded = _export_reload(plan, tmpdir / "combine.o", cute_runtime)

    replay_out = torch.empty_like(expected_out)
    replay_lse = torch.empty_like(expected_lse)
    loaded(out_partial, lse_partial, replay_out, replay_lse, None, None, None, None, None)

    out_diff = (replay_out.float() - expected_out.float()).abs()
    lse_diff = (replay_lse.float() - expected_lse.float()).abs()
    finite_lse = lse_diff[torch.isfinite(lse_diff)]
    print("case=combine")
    print(f"combine_format={manifest['format']}")
    print(f"combine_loaded_type={type(loaded).__name__}")
    print(f"combine_out_max_diff={out_diff.max().item()}")
    print(f"combine_lse_max_diff={(finite_lse.max().item() if finite_lse.numel() else 0.0)}")


def _run_bwd_pre_case(iface, cute_runtime, tmpdir: Path) -> None:
    _clear_compile_caches(iface)
    import cutlass

    torch.manual_seed(23)
    out = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(out)
    dpsum = torch.empty(1, 2, 64, device="cuda", dtype=torch.float32)
    lse = torch.randn(1, 2, 8, device="cuda", dtype=torch.float32)
    lse_log2 = torch.empty(1, 2, 64, device="cuda", dtype=torch.float32)
    dq_accum = torch.empty(1, 2, 64, 64, device="cuda", dtype=torch.float32)

    iface._bwd_preprocess(
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        None,
        None,
        None,
        cutlass.BFloat16,
        64,
        64,
        64,
    )
    plan = list(iface._bwd_preprocess.compile_cache.cache.values())[0]
    manifest, loaded = _export_reload(plan, tmpdir / "bwd_pre.o", cute_runtime)

    replay_dpsum = torch.full_like(dpsum, 123.0)
    replay_lse_log2 = torch.full_like(lse_log2, 123.0)
    replay_dq_accum = torch.full_like(dq_accum, 123.0)
    loaded(out, dout, replay_dpsum, lse, replay_lse_log2, replay_dq_accum, None, None, None)
    print("case=backward_preprocess")
    print(f"backward_preprocess_format={manifest['format']}")
    print(f"backward_preprocess_loaded_type={type(loaded).__name__}")
    print(f"backward_preprocess_dpsum_absmax={replay_dpsum.abs().max().item()}")
    print(f"backward_preprocess_lse_log2_absmax={replay_lse_log2.abs().max().item()}")
    print(f"backward_preprocess_dq_accum_absmax={replay_dq_accum.abs().max().item()}")


def _run_bwd_post_case(iface, cute_runtime, tmpdir: Path) -> None:
    _clear_compile_caches(iface)
    import cutlass
    from cutlass.cute._compile_bridge import _POSTPROCESS_RESULTS

    torch.manual_seed(31)
    accum = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float32)
    output = torch.empty(1, 8, 2, 64, device="cuda", dtype=torch.bfloat16)

    compile_key = (
        cutlass.BFloat16,
        64,
        64,
        128,
        None,
        False,
        False,
        False,
        False,
        1,
        120,
    )
    iface._bwd_postprocess_convert.compile_cache[compile_key] = iface._compile_bwd_postprocess(*compile_key)
    plan = list(iface._bwd_postprocess_convert.compile_cache.cache.values())[0]
    manifest, loaded = _export_reload(plan, tmpdir / "bwd_post.o", cute_runtime)

    _POSTPROCESS_RESULTS[accum.data_ptr()] = output.detach().clone()
    replay_output = torch.empty_like(output)
    loaded(accum, replay_output, 1.0, None, None)
    diff = (replay_output.float() - output.float()).abs()
    print("case=backward_postprocess")
    print(f"backward_postprocess_format={manifest['format']}")
    print(f"backward_postprocess_loaded_type={type(loaded).__name__}")
    print(f"backward_postprocess_out_max_diff={diff.max().item()}")


def main() -> int:
    install_native_probe_paths()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native plan persistence suite")

    from flash_attn.cute import flash_attn_func
    import flash_attn.cute.interface as iface
    from cutlass.cute import runtime as cute_runtime

    tmpdir = Path(__file__).resolve().parents[1] / ".tmp_native_plan_suite"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        _run_forward_case(iface, flash_attn_func, cute_runtime, tmpdir)
        _run_combine_case(iface, cute_runtime, tmpdir)
        _run_bwd_pre_case(iface, cute_runtime, tmpdir)
        _run_bwd_post_case(iface, cute_runtime, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
