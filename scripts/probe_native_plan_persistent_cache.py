"""Verify manifest-backed export/load for runtime-owned native plans on Windows."""

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


def main() -> int:
    install_native_probe_paths()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native plan persistence probe")

    from flash_attn.cute import flash_attn_func
    import flash_attn.cute.interface as iface
    from cutlass.cute import runtime as cute_runtime

    _clear_compile_caches(iface)
    torch.manual_seed(0)
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)

    expected_out, expected_lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    cached_values = list(getattr(iface._flash_attn_fwd.compile_cache, "cache", {}).values())
    if len(cached_values) != 1:
        raise RuntimeError(f"Expected exactly one compiled plan, found {len(cached_values)}")
    plan = cached_values[0]
    print(f"compiled_plan_type={type(plan).__name__}")
    print(f"compiled_plan_repr={plan!r}")

    tmpdir = Path(__file__).resolve().parents[1] / ".tmp_native_plan_probe"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        artifact_path = tmpdir / "dense_fwd_plan.o"
        manifest = plan.export_to_c(object_file_path=str(artifact_path), function_name="func")
        print(f"artifact_path={artifact_path}")
        print(f"artifact_format={manifest['format']}")
        loaded_module = cute_runtime.load_module(str(artifact_path), enable_tvm_ffi=True)
        loaded_plan = loaded_module.func
        print(f"loaded_plan_type={type(loaded_plan).__name__}")
        print(f"loaded_plan_repr={loaded_plan!r}")

        replay_out = torch.empty_like(expected_out)
        replay_lse = torch.empty_like(expected_lse)
        loaded_plan(
            q.detach(),
            k.detach(),
            v.detach(),
            replay_out,
            replay_lse,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        out_diff = (replay_out.float() - expected_out.float()).abs()
        lse_diff = (replay_lse.float() - expected_lse.float()).abs()
        print(f"replay_out_max_diff={out_diff.max().item()}")
        print(f"replay_lse_max_diff={lse_diff.max().item()}")
        print(f"replay_exact={bool(out_diff.max().item() == 0.0 and lse_diff.max().item() == 0.0)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
