"""Run backward parity probes through the isolated native FA4 bridge path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


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


def main() -> int:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "native_probe_shims"))

    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native backward probe")

    shim_mod = _load_windows_shim_module()
    _run_dense(flash_attn_func, shim_mod)
    _run_varlen(flash_attn_varlen_func, shim_mod)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
