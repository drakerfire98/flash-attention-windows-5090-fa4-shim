"""Exercise upstream persistent cache flow against runtime-owned native plans."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

from _native_probe_setup import install_native_probe_paths


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / ".tmp_cute_disk_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = str(cache_dir)

    install_native_probe_paths()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native disk-cache probe")

    from flash_attn.cute import flash_attn_func
    import flash_attn.cute.interface as iface
    import cutlass
    from cutlass.cute._compile_bridge import _POSTPROCESS_RESULTS

    torch.manual_seed(0)
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)

    out1, lse1 = flash_attn_func(q, k, v, causal=True, return_lse=True)
    compile_cache = iface._flash_attn_fwd.compile_cache
    cached_values = list(compile_cache.cache.values())
    if len(cached_values) != 1:
        raise RuntimeError(f"Expected one in-memory cache entry after first compile, found {len(cached_values)}")
    print(f"first_cache_type={type(compile_cache).__name__}")
    print(f"first_cached_plan={cached_values[0]!r}")

    artifact_files = sorted(str(path.relative_to(cache_dir)) for path in cache_dir.rglob("*") if path.is_file())
    print(f"artifact_files={artifact_files}")

    compile_cache.cache.clear()
    out2, lse2 = flash_attn_func(q, k, v, causal=True, return_lse=True)
    reloaded_values = list(compile_cache.cache.values())
    if len(reloaded_values) != 1:
        raise RuntimeError(f"Expected one in-memory cache entry after disk reload, found {len(reloaded_values)}")
    print(f"reloaded_cached_plan={reloaded_values[0]!r}")

    out_diff = (out2.float() - out1.float()).abs()
    lse_diff = (lse2.float() - lse1.float()).abs()
    print(f"disk_reload_out_max_diff={out_diff.max().item()}")
    print(f"disk_reload_lse_max_diff={lse_diff.max().item()}")
    print(f"disk_reload_exact={bool(out_diff.max().item() == 0.0 and lse_diff.max().item() == 0.0)}")

    torch.manual_seed(11)
    out_partial = torch.randn(3, 2, 4, 3, 8, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.randn(3, 2, 4, 3, device="cuda", dtype=torch.float32)
    iface.flash_attn_combine(out_partial, lse_partial, return_lse=True)
    combine_cache = iface._flash_attn_fwd_combine.compile_cache
    print(f"combine_cache_type={type(combine_cache).__name__}")
    combine_artifacts = sorted(
        str(path.relative_to(cache_dir))
        for path in cache_dir.rglob("*")
        if path.is_file() and "fwd_combine" in str(path)
    )
    print(f"combine_artifact_files={combine_artifacts}")
    combine_cache.cache.clear()
    iface.flash_attn_combine(out_partial, lse_partial, return_lse=True)
    print(f"combine_reloaded_cached_plan={list(combine_cache.cache.values())[0]!r}")

    torch.manual_seed(23)
    out = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(out)
    dpsum = torch.empty(1, 2, 64, device="cuda", dtype=torch.float32)
    lse = torch.randn(1, 2, 8, device="cuda", dtype=torch.float32)
    lse_log2 = torch.empty(1, 2, 64, device="cuda", dtype=torch.float32)
    dq_accum = torch.empty(1, 2, 64, 64, device="cuda", dtype=torch.float32)
    iface._bwd_preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, None, None, None, cutlass.BFloat16, 64, 64, 64)
    bwd_pre_cache = iface._bwd_preprocess.compile_cache
    print(f"bwd_pre_cache_type={type(bwd_pre_cache).__name__}")
    bwd_pre_artifacts = sorted(
        str(path.relative_to(cache_dir))
        for path in cache_dir.rglob("*")
        if path.is_file() and "bwd_pre" in str(path)
    )
    print(f"bwd_pre_artifact_files={bwd_pre_artifacts}")
    bwd_pre_cache.cache.clear()
    replay_dpsum = torch.full_like(dpsum, 123.0)
    replay_lse_log2 = torch.full_like(lse_log2, 123.0)
    replay_dq_accum = torch.full_like(dq_accum, 123.0)
    iface._bwd_preprocess(
        out,
        dout,
        replay_dpsum,
        lse,
        replay_lse_log2,
        replay_dq_accum,
        None,
        None,
        None,
        cutlass.BFloat16,
        64,
        64,
        64,
    )
    print(f"bwd_pre_reload_dpsum_absmax={replay_dpsum.abs().max().item()}")
    print(f"bwd_pre_reload_lse_log2_absmax={replay_lse_log2.abs().max().item()}")
    print(f"bwd_pre_reload_dq_accum_absmax={replay_dq_accum.abs().max().item()}")

    bwd_post_key = (
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
    iface._bwd_postprocess_convert.compile_cache[bwd_post_key] = iface._compile_bwd_postprocess(*bwd_post_key)
    bwd_post_cache = iface._bwd_postprocess_convert.compile_cache
    print(f"bwd_post_cache_type={type(bwd_post_cache).__name__}")
    bwd_post_artifacts = sorted(
        str(path.relative_to(cache_dir))
        for path in cache_dir.rglob("*")
        if path.is_file() and "bwd_post" in str(path)
    )
    print(f"bwd_post_artifact_files={bwd_post_artifacts}")
    bwd_post_cache.cache.clear()
    accum = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float32)
    output = torch.empty(1, 8, 2, 64, device="cuda", dtype=torch.bfloat16)
    _POSTPROCESS_RESULTS[accum.data_ptr()] = output.detach().clone()
    replay_output = torch.empty_like(output)
    bwd_post_cache[bwd_post_key](accum, replay_output, 1.0, None, None)
    print(f"bwd_post_reload_out_max_diff={(replay_output.float() - output.float()).abs().max().item()}")

    shutil.rmtree(cache_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
