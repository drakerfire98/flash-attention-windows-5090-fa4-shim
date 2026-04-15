"""Run a tiny native FA4 forward probe through the isolated native shims."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "native_probe_shims"))

    from flash_attn.cute import flash_attn_func
    import flash_attn.cute.interface as iface
    import cutlass

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native forward probe")

    torch.manual_seed(0)
    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)

    out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    ref = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        is_causal=True,
    ).permute(0, 2, 1, 3)

    diff = (out.float() - ref.float()).abs()
    compiled_values = list(iface._flash_attn_fwd.compile_cache.cache.values())
    compiled_type = type(compiled_values[0]).__name__ if compiled_values else "missing"
    compiled_repr = repr(compiled_values[0]) if compiled_values else "<missing>"

    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")
    print(f"out_device={out.device}")
    print(f"out_dtype={out.dtype}")
    print(f"out_finite={bool(torch.isfinite(out).all().item())}")
    print(f"lse_type={type(lse).__name__}")
    if lse is not None:
        print(f"lse_shape={tuple(lse.shape)}")
        print(f"lse_finite={bool(torch.isfinite(lse).all().item())}")
    print(f"max_diff={diff.max().item()}")
    print(f"mean_diff={diff.mean().item()}")
    print(f"sum_out={out.float().sum().item()}")
    print(f"sum_ref={ref.float().sum().item()}")
    print(f"compiled_type={compiled_type}")
    print(f"compiled_repr={compiled_repr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
