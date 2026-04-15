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
    print(f"{name}_out_finite={bool(torch.isfinite(native_out).all().item())}")
    if native_lse is not None and ref_lse is not None:
        if native_lse.shape != ref_lse.shape:
            if native_lse.ndim == 3 and ref_lse.ndim == 3:
                ref_lse = ref_lse.permute(0, 2, 1).contiguous()
            elif native_lse.ndim == 2 and ref_lse.ndim == 2:
                ref_lse = ref_lse.transpose(-1, -2).contiguous()
        lse_diff = (native_lse.float() - ref_lse.float()).abs()
        print(f"{name}_lse_max_diff={lse_diff.max().item()}")
        print(f"{name}_lse_mean_diff={lse_diff.mean().item()}")
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


def main() -> int:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "native_probe_shims"))

    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
    import flash_attn.cute.interface as iface
    import cutlass

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native forward probe")

    shim_mod = _load_windows_shim_module()
    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")

    _run_dense_base(flash_attn_func)
    _run_dense_softcap(flash_attn_func, shim_mod)
    _run_dense_mask_mod(flash_attn_func, shim_mod)
    _run_varlen_softcap(flash_attn_varlen_func, shim_mod)

    compiled_values = list(iface._flash_attn_fwd.compile_cache.cache.values())
    compiled_types = sorted({type(value).__name__ for value in compiled_values}) if compiled_values else ["missing"]
    print(f"compiled_types={compiled_types}")
    if compiled_values:
        print(f"compiled_repr_sample={repr(compiled_values[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
