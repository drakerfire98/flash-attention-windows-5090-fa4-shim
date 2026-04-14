"""Probe FA4 imports with the local cutlass shim ahead of site-packages."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


def _run_forward_smoke(flash_attn_func) -> int:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)

    out = flash_attn_func(q, k, v, causal=True)
    if isinstance(out, tuple):
        main_out = out[0]
        aux = out[1] if len(out) > 1 else None
    else:
        main_out = out
        aux = None

    q_ref = q.permute(0, 2, 1, 3)
    k_ref = k.permute(0, 2, 1, 3)
    v_ref = v.permute(0, 2, 1, 3)
    ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    ref = ref.permute(0, 2, 1, 3).to(main_out.dtype)

    max_diff = (main_out.float() - ref.float()).abs().max().item()
    mean_diff = (main_out.float() - ref.float()).abs().mean().item()

    print("forward_smoke: success")
    print(f"output_type: {type(out).__name__}")
    print(f"main_out_shape: {tuple(main_out.shape)}")
    print(f"main_out_dtype: {main_out.dtype}")
    print(f"aux_type: {type(aux).__name__}")
    print(f"out_finite: {bool(torch.isfinite(main_out).all().item())}")
    print(f"max_diff_vs_sdpa: {max_diff}")
    print(f"mean_diff_vs_sdpa: {mean_diff}")
    return 0


def _run_backward_smoke(flash_attn_func) -> int:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out, _ = flash_attn_func(q, k, v, causal=True)
    loss = out.float().sum()
    loss.backward()

    ref = F.scaled_dot_product_attention(
        q_ref.permute(0, 2, 1, 3),
        k_ref.permute(0, 2, 1, 3),
        v_ref.permute(0, 2, 1, 3),
        is_causal=True,
    ).permute(0, 2, 1, 3)
    ref.float().sum().backward()

    dq_diff = (q.grad.float() - q_ref.grad.float()).abs().max().item()
    dk_diff = (k.grad.float() - k_ref.grad.float()).abs().max().item()
    dv_diff = (v.grad.float() - v_ref.grad.float()).abs().max().item()

    print("backward_smoke: success")
    print(f"max_dq_diff_vs_sdpa: {dq_diff}")
    print(f"max_dk_diff_vs_sdpa: {dk_diff}")
    print(f"max_dv_diff_vs_sdpa: {dv_diff}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-forward",
        action="store_true",
        help="Run a tiny CUDA forward pass and compare it to torch SDPA.",
    )
    parser.add_argument(
        "--run-backward",
        action="store_true",
        help="Run a tiny CUDA backward pass and compare grads to torch SDPA.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    shim_root = repo_root / "shims"

    sys.path.insert(0, str(shim_root))

    print(f"python: {sys.executable}")
    print(f"shim_root: {shim_root}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")

    try:
        import cutlass  # noqa: F401
        print(f"cutlass module: {getattr(cutlass, '__file__', '<shim>')}")

        import flash_attn.cute as flash_attn_cute
        from flash_attn.cute import flash_attn_func  # noqa: F401

        print("fa4_import: success")
        print(f"flash_attn.cute module: {getattr(flash_attn_cute, '__file__', '<shim>')}")
        print(f"flash_attn_func: {flash_attn_func}")
        if args.run_forward:
            return _run_forward_smoke(flash_attn_func)
        if args.run_backward:
            return _run_backward_smoke(flash_attn_func)
        return 0
    except Exception as exc:
        print(f"fa4_import: failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
