"""Run forward-combine parity probes through the isolated native FA4 bridge path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from _native_probe_setup import install_native_probe_paths, native_flash_attn_interface_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_windows_shim_module():
    repo_root = _repo_root()
    shim_init = repo_root / "shims" / "flash_attn" / "cute" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "native_probe_windows_flash_attn_cute_shim_combine_probe",
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
    native_lse: torch.Tensor | None,
    ref_lse: torch.Tensor | None,
) -> None:
    out_diff = (native_out.float() - ref_out.float()).abs()
    print(f"case={name}")
    print(f"{name}_out_max_diff={out_diff.max().item()}")
    print(f"{name}_out_mean_diff={out_diff.mean().item()}")
    print(f"{name}_out_exact={bool(out_diff.max().item() == 0.0)}")
    if native_lse is not None and ref_lse is not None:
        lse_diff = (native_lse.float() - ref_lse.float()).abs()
        finite_lse_diff = lse_diff[torch.isfinite(lse_diff)]
        max_lse = finite_lse_diff.max().item() if finite_lse_diff.numel() > 0 else 0.0
        mean_lse = finite_lse_diff.mean().item() if finite_lse_diff.numel() > 0 else 0.0
        print(f"{name}_lse_max_diff={max_lse}")
        print(f"{name}_lse_mean_diff={mean_lse}")
        print(f"{name}_lse_exact={bool(max_lse == 0.0)}")


def _clear_compile_cache(iface) -> None:
    compile_cache = getattr(getattr(iface, "_flash_attn_fwd_combine", None), "compile_cache", None)
    cache_dict = getattr(compile_cache, "cache", None)
    if hasattr(cache_dict, "clear"):
        cache_dict.clear()
    elif hasattr(compile_cache, "clear"):
        compile_cache.clear()


def _run_batched(iface, shim_mod):
    torch.manual_seed(101)
    out_partial = torch.randn(3, 2, 4, 3, 8, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.randn(3, 2, 4, 3, device="cuda", dtype=torch.float32)
    lse_partial[2, 1, 3, :] = float("-inf")
    native_out, native_lse = iface.flash_attn_combine(out_partial, lse_partial, return_lse=True)
    ref_out, ref_lse = shim_mod.flash_attn_combine(out_partial, lse_partial, return_lse=True)
    _print_case_metrics("combine_batched", native_out, ref_out, native_lse, ref_lse)


def _run_varlen(iface, shim_mod):
    torch.manual_seed(131)
    out_partial = torch.randn(4, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.randn(4, 7, 2, device="cuda", dtype=torch.float32)
    lse_partial[:, 5, 1] = float("-inf")
    native_out, native_lse = iface.flash_attn_combine(out_partial, lse_partial, return_lse=True)
    ref_out, ref_lse = shim_mod.flash_attn_combine(out_partial, lse_partial, return_lse=True)
    _print_case_metrics("combine_varlen", native_out, ref_out, native_lse, ref_lse)


def _run_dynamic_batched(iface, shim_mod):
    torch.manual_seed(151)
    out_partial = torch.randn(4, 2, 5, 3, 8, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.randn(4, 2, 5, 3, device="cuda", dtype=torch.float32)
    lse_partial[:, 1, 4, :] = float("-inf")
    seqused = torch.tensor([3, 5], device="cuda", dtype=torch.int32)
    dynamic_splits = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    native_out = torch.empty(2, 5, 3, 8, device="cuda", dtype=torch.bfloat16)
    native_lse = torch.empty(2, 5, 3, device="cuda", dtype=torch.float32)
    iface._flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        native_out,
        native_lse,
        None,
        seqused,
        dynamic_splits,
        None,
        None,
    )
    ref_out, ref_lse = shim_mod.flash_attn_combine(
        out_partial,
        lse_partial,
        seqused=seqused,
        num_splits_dynamic_ptr=dynamic_splits,
        return_lse=True,
    )
    _print_case_metrics("combine_dynamic_batched", native_out, ref_out, native_lse, ref_lse)


def _run_dynamic_varlen(iface, shim_mod):
    torch.manual_seed(173)
    out_partial = torch.randn(4, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.randn(4, 7, 2, device="cuda", dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    seqused = torch.tensor([3, 2], device="cuda", dtype=torch.int32)
    dynamic_splits = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    varlen_batch_idx = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    native_out = torch.empty(7, 2, 8, device="cuda", dtype=torch.bfloat16)
    native_lse = torch.empty(7, 2, device="cuda", dtype=torch.float32)
    iface._flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        native_out,
        native_lse,
        cu_seqlens,
        seqused,
        dynamic_splits,
        varlen_batch_idx,
        None,
    )
    ref_out, ref_lse = shim_mod.flash_attn_combine(
        out_partial,
        lse_partial,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        varlen_batch_idx=varlen_batch_idx,
        num_splits_dynamic_ptr=dynamic_splits,
        return_lse=True,
    )
    _print_case_metrics("combine_dynamic_varlen", native_out, ref_out, native_lse, ref_lse)


def main() -> int:
    install_native_probe_paths()
    interface_target = native_flash_attn_interface_path()

    import flash_attn.cute.interface as iface
    import cutlass
    from cutlass.cute._native_backend import native_combine_backend_status

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native combine probe")

    shim_mod = _load_windows_shim_module()
    print(f"native_interface={interface_target}")
    print(f"loaded_interface={getattr(iface, '__file__', '<unknown>')}")
    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")
    print(f"native_combine_backend_pre={native_combine_backend_status()}")

    for runner in (
        lambda: _run_batched(iface, shim_mod),
        lambda: _run_varlen(iface, shim_mod),
        lambda: _run_dynamic_batched(iface, shim_mod),
        lambda: _run_dynamic_varlen(iface, shim_mod),
    ):
        _clear_compile_cache(iface)
        runner()
        compiled_values = list(iface._flash_attn_fwd_combine.compile_cache.cache.values())
        compiled_types = (
            sorted({type(value).__name__ for value in compiled_values})
            if compiled_values
            else ["missing"]
        )
        print(f"compiled_types={compiled_types}")
        if compiled_values:
            print(f"compiled_repr_sample={repr(compiled_values[0])}")
        print(f"native_combine_backend_post={native_combine_backend_status()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
