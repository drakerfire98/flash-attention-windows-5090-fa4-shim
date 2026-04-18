"""Verify persistent export/load for simple local closure-backed native plans."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from _native_probe_setup import install_native_probe_paths


def make_local_sliding_window_mask(window_left: int, window_right: int):
    def local_mask(batch_idx, head_idx, q_idx, kv_idx):
        del batch_idx, head_idx
        return (kv_idx >= (q_idx - window_left)) & (kv_idx <= (q_idx + window_right))

    return local_mask


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


def main() -> int:
    install_native_probe_paths()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the closure native-plan probe")

    from flash_attn.cute import flash_attn_func
    from cutlass.cute import runtime as cute_runtime
    import flash_attn.cute.interface as iface

    tmpdir = Path(__file__).resolve().parents[1] / ".tmp_native_closure_plan"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        _clear_compile_caches(iface)
        torch.manual_seed(21)
        q = torch.randn(1, 18, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 18, 2, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 18, 2, 64, device="cuda", dtype=torch.bfloat16)
        mask_mod = make_local_sliding_window_mask(2, 3)

        expected_out, expected_lse = flash_attn_func(
            q, k, v, causal=False, return_lse=True, mask_mod=mask_mod
        )
        plan = list(iface._flash_attn_fwd.compile_cache.cache.values())[0]
        manifest = plan.export_to_c(object_file_path=str(tmpdir / "closure_mask.o"), function_name="func")
        loaded_plan = cute_runtime.load_module(str(tmpdir / "closure_mask.o"), enable_tvm_ffi=True).func

        replay_out = torch.empty_like(expected_out)
        replay_lse = torch.empty_like(expected_lse)
        loaded_plan(
            q.detach(), k.detach(), v.detach(), replay_out, replay_lse,
            None, None, None, None, None, None, None, None, None, None, None,
        )
        out_diff = (replay_out.float() - expected_out.float()).abs()
        lse_diff = (replay_lse.float() - expected_lse.float()).abs()
        print("case=closure_mask_mod")
        print(f"closure_mask_format={manifest['format']}")
        print(f"closure_mask_out_max_diff={out_diff.max().item()}")
        print(f"closure_mask_lse_max_diff={lse_diff.max().item()}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
