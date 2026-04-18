"""Verify persistent export/load for importable callable-backed native plans."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from _native_probe_setup import install_native_probe_paths
from native_plan_callable_defs import dense_score_bias, striped_mask


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


def main() -> int:
    install_native_probe_paths()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the callable native-plan probe")

    from flash_attn.cute import flash_attn_func
    import flash_attn.cute.interface as iface
    from cutlass.cute import runtime as cute_runtime

    tmpdir = Path(__file__).resolve().parents[1] / ".tmp_native_callable_plan"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        _clear_compile_caches(iface)
        torch.manual_seed(7)
        q = torch.randn(1, 20, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 20, 2, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 20, 2, 64, device="cuda", dtype=torch.bfloat16)

        expected_out, expected_lse = flash_attn_func(
            q, k, v, causal=False, return_lse=True, mask_mod=striped_mask
        )
        mask_plan = list(iface._flash_attn_fwd.compile_cache.cache.values())[0]
        mask_manifest, loaded_mask_plan = _export_reload(mask_plan, tmpdir / "mask_mod.o", cute_runtime)
        replay_out = torch.empty_like(expected_out)
        replay_lse = torch.empty_like(expected_lse)
        loaded_mask_plan(
            q.detach(), k.detach(), v.detach(), replay_out, replay_lse,
            None, None, None, None, None, None, None, None, None, None, None,
        )
        mask_out_diff = (replay_out.float() - expected_out.float()).abs()
        mask_lse_diff = (replay_lse.float() - expected_lse.float()).abs()
        print("case=mask_mod")
        print(f"mask_mod_format={mask_manifest['format']}")
        print(f"mask_mod_out_max_diff={mask_out_diff.max().item()}")
        print(f"mask_mod_lse_max_diff={mask_lse_diff.max().item()}")

        _clear_compile_caches(iface)
        expected_out2, expected_lse2 = iface._flash_attn_fwd(
            q, k, v, causal=True, return_lse=True, score_mod=dense_score_bias
        )
        score_plan = list(iface._flash_attn_fwd.compile_cache.cache.values())[0]
        score_manifest, loaded_score_plan = _export_reload(score_plan, tmpdir / "score_mod.o", cute_runtime)
        replay_out2 = torch.empty_like(expected_out2)
        replay_lse2 = torch.empty_like(expected_lse2)
        loaded_score_plan(
            q.detach(), k.detach(), v.detach(), replay_out2, replay_lse2,
            None, None, None, None, None, None, None, None, None, None, None,
        )
        score_out_diff = (replay_out2.float() - expected_out2.float()).abs()
        score_lse_diff = (replay_lse2.float() - expected_lse2.float()).abs()
        print("case=score_mod")
        print(f"score_mod_format={score_manifest['format']}")
        print(f"score_mod_out_max_diff={score_out_diff.max().item()}")
        print(f"score_mod_lse_max_diff={score_lse_diff.max().item()}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
